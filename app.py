from flask import Flask, render_template, request, redirect, send_file, url_for, session, jsonify
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename
from agents.file_handling_agent import handle_zip_submission, build_file_hierarchy
from agents.evaluation_agent import EvaluationAgent
#from agents.feedback_agent import generate_feedback_csv
from tools.database_utils import DatabaseManager
from tools.assessment_utils import AssessmentManager
from tools.clip_vector_store import CLIPVectorStore
from typing import Dict, List
import os
import shutil
import functools
import time
import uuid
import sqlite3
import fitz  # PyMuPDF
from rapidocr_onnxruntime import RapidOCR
import threading
import numpy as np
import faiss
import open_clip
from PIL import Image
import torch

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SECRET_KEY'] = os.urandom(24)
socketio = SocketIO(app)

KNOWLEDGE_FOLDER = 'knowledge'
PROMPT_TEMPLATE_PATH = 'prompt.txt'
CONTEXT_TEMPLATE_PATH = 'context.txt'
os.makedirs(KNOWLEDGE_FOLDER, exist_ok=True)

# Ensure prompt.txt and context.txt exist with defaults
DEFAULT_PROMPT = "Enter your evaluation prompt here."
DEFAULT_CONTEXT = "Enter your evaluation context here."
if not os.path.exists(PROMPT_TEMPLATE_PATH):
    with open(PROMPT_TEMPLATE_PATH, 'w', encoding='utf-8') as f:
        f.write(DEFAULT_PROMPT)
if not os.path.exists(CONTEXT_TEMPLATE_PATH):
    with open(CONTEXT_TEMPLATE_PATH, 'w', encoding='utf-8') as f:
        f.write(DEFAULT_CONTEXT)

# Initialize managers and agents
assessment_manager = AssessmentManager()
db_manager = DatabaseManager()

# Initialize evaluation agent
evaluation_agent = EvaluationAgent()

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Global Chatbot RAG Cache ---
chatbot_rag = None
chatbot_rag_last_updated = 0

# --- Knowledge Documents Table (SQLite) ---
def init_knowledge_table():
    conn = sqlite3.connect('storage/evaluations.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS knowledge_docs (
        id TEXT PRIMARY KEY,
        filename TEXT,
        filepath TEXT,
        upload_date TEXT
    )''')
    conn.commit()
    conn.close()

init_knowledge_table()

# --- Utility: Save knowledge doc and add to multimodal RAG ---
def process_knowledge_pdf(pdf_path, doc_id, target_dir=None):
    """
    Extract text and images from the PDF, run OCR on images, and add all content to the vectorstore.
    If target_dir is provided, use it as the vectorstore directory; otherwise, use storage/rag_store.
    """
    import tempfile
    import shutil
    import os
    import torch

    # 1. Extract text from PDF
    text_chunks = []
    doc = fitz.open(pdf_path)
    for page in doc:
        text = page.get_text()
        if text.strip():
            text_chunks.append(text)

    # 2. Extract images and run OCR
    ocr = RapidOCR()
    image_paths = []
    for page_num in range(len(doc)):
        for img_index, img in enumerate(doc.get_page_images(page_num)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            # Save image to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
                tmp_img.write(image_bytes)
                tmp_img_path = tmp_img.name
            image_paths.append(tmp_img_path)

    # 3. Vector store path
    vectorstore_dir = target_dir if target_dir else "storage/rag_store"
    os.makedirs(vectorstore_dir, exist_ok=True)
    index_path = os.path.join(vectorstore_dir, "clip.index")
    store = CLIPVectorStore(index_path)

    # 4. Add text chunks
    for chunk in text_chunks:
        emb = store.embed_text(chunk)
        store.add(emb, {"source": pdf_path, "type": "text"})

    # 5. Add image embeddings
    for img_path in image_paths:
        emb = store.embed_image(img_path)
        store.add(emb, {"source": pdf_path, "type": "image"})
        os.unlink(img_path)

def agentic_index_documents(project_path, project_vectorstore_dir):
    """
    Index ALL files in the project directory (recursively), regardless of extension.
    Emits status messages via socketio for UI feedback.
    """
    import glob
    all_files = []
    for root, _, files in os.walk(project_path):
        for file in files:
            if file.endswith(('.docx', '.pdf')):
                all_files.append(os.path.join(root, file))
    print(f"[agentic_index_documents] All files found: {all_files}")
    socketio.emit('status_update', {
        'project': os.path.basename(project_path),
        'status': f'Indexing {len(all_files)} file(s) in project...'
    })
    indexed_files = []
    for file_path in all_files:
        try:
            socketio.emit('status_update', {
                'project': os.path.basename(project_path),
                'status': f'Indexing: {os.path.basename(file_path)}'
            })
            print(f'[agentic_index_documents] Indexing: {file_path}')
            process_knowledge_pdf(file_path, doc_id=None, target_dir=project_vectorstore_dir)
            indexed_files.append(file_path)
        except Exception as e:
            print(f'[agentic_index_documents] Error indexing {file_path}: {e}')
            socketio.emit('status_update', {
                'project': os.path.basename(project_path),
                'status': f'Error indexing {os.path.basename(file_path)}: {e}'
            })
    print(f"[agentic_index_documents] Indexed files: {indexed_files}")
    # Update the project's clip_vectorstore_path in the database
    db_manager.set_clip_vectorstore_path(os.path.basename(project_path), os.path.join(project_vectorstore_dir, 'clip.index'))
    socketio.emit('status_update', {
        'project': os.path.basename(project_path),
        'status': 'Document agent indexing complete.'
    })

@app.route('/')
def index():
    submissions = db_manager.get_all_projects()
    reports = []
    for sub in submissions:
        # Flatten file hierarchy to get all files with relative paths
        def flatten_files(node, rel_path=""):
            files = []
            if node.get('type') == 'file':
                files.append({
                    'name': node['name'],
                    'path': os.path.join(sub['project_name'], rel_path, node['name']).replace('\\', '/')
                })
            elif node.get('type') == 'directory':
                for child in node.get('children', []):
                    child_rel = os.path.join(rel_path, node['name']) if rel_path else node['name']
                    files.extend(flatten_files(child, child_rel))
            return files
        file_list = []
        if sub.get('file_hierarchy'):
            file_list = flatten_files(sub['file_hierarchy'], rel_path="")
        reports.append({
            'id': sub['project_name'],
            'folder_name': sub['project_name'],
            'upload_date': sub.get('timestamp', ''),
            'files': file_list,
            'feedback': sub['feedback']
        })
    # Get all knowledge docs
    conn = sqlite3.connect('storage/evaluations.db')
    c = conn.cursor()
    c.execute("SELECT id, filename, upload_date FROM knowledge_docs")
    knowledge_docs = c.fetchall()
    conn.close()
    return render_template('index.html', submissions=submissions, knowledge_docs=knowledge_docs, reports=reports)

# Serve uploaded files from storage/projects_rag
@app.route('/uploads/<path:filepath>')
def serve_uploaded_file(filepath):
    abs_path = os.path.join('storage', 'projects_rag', filepath)
    if os.path.exists(abs_path):
        return send_file(abs_path, as_attachment=False)
    return redirect(url_for('index'))

@app.route('/open_file/<project>/<path:filepath>')
def open_file(project, filepath):
    # Serve a file from storage/projects_rag/<project>/<filepath>
    abs_path = os.path.join('storage', 'projects_rag', project, filepath)
    if os.path.exists(abs_path):
        return send_file(abs_path, as_attachment=False)
    return redirect(url_for('index'))

@app.route('/view-assessment')
def view_assessment():
    brief_path = os.path.join('storage/assessment_brief', 'assessment_brief.pdf')
    if os.path.exists(brief_path):
        return send_file(brief_path, as_attachment=False)
    return redirect('/')

@app.route('/upload-assessment', methods=['POST'])
def upload_assessment():
    if 'file' not in request.files:
        return redirect('/')
    
    assessment_file = request.files['file']
    if assessment_file.filename == '':
        return redirect('/')
    
    if assessment_file and assessment_file.filename.endswith('.pdf'):
        # Save the assessment brief and create RAG
        socketio.emit('status', {'message': "Saving assessment brief..."})
        filepath = assessment_manager.save_assessment_brief(assessment_file)
        
        socketio.emit('status', {'message': "Creating assessment RAG..."})
        assessment_manager.create_rag(filepath)
        socketio.emit('status', {'message': "Assessment brief updated and criteria generated successfully!"})
    
    return redirect('/')

@app.route('/upload', methods=['POST'])
def upload_file():
    print('Upload endpoint triggered')
    if 'file' not in request.files:
        print('No file part in request.files')
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    print(f'File received: {file.filename}')
    if file.filename == '':
        print('No selected file')
        return jsonify({'error': 'No selected file'}), 400
    if not file.filename.endswith('.zip'):
        print('File is not a ZIP archive')
        return jsonify({'error': 'File must be a ZIP archive'}), 400
    import tempfile
    import shutil
    try:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        print(f"File saved to: {filepath}")  # Debug log
        # Extract ZIP to a temp directory
        temp_dir = tempfile.mkdtemp(prefix='java_eval_')
        shutil.unpack_archive(filepath, temp_dir)
        print(f"Extracted ZIP to: {temp_dir}")
        # List all top-level folders/files in temp_dir
        top_level_items = os.listdir(temp_dir)
        projects = []
        threads = []
        for item in top_level_items:
            src_path = os.path.join(temp_dir, item)
            if os.path.isdir(src_path):
                dest_project_path = os.path.join('storage', 'projects_rag', item)
                if os.path.exists(dest_project_path):
                    shutil.rmtree(dest_project_path)
                shutil.move(src_path, dest_project_path)
                print(f"Moved project to: {dest_project_path}")
                # Build file hierarchy for DB
                file_hierarchy = build_file_hierarchy(dest_project_path)
                projects.append({
                    'name': item,
                    'path': dest_project_path,
                    'hierarchy': file_hierarchy
                })
                db_manager.save_project(
                    project_name=item,
                    file_hierarchy=file_hierarchy,
                    project_path=dest_project_path
                )
                # --- Agentic document and code indexing on upload (threaded) ---
                def thread_target(project_path=dest_project_path):
                    try:
                        agentic_index_documents(project_path, project_path)
                    except Exception as e:
                        print(f'[upload_file] Error in agentic_index_documents thread: {e}')
                t = threading.Thread(target=thread_target)
                threads.append(t)
                t.start()
            elif os.path.isfile(src_path):
                # Optionally handle single files as projects (rare)
                dest_file_path = os.path.join('storage', 'projects_rag', item)
                shutil.move(src_path, dest_file_path)
                print(f"Moved file to: {dest_file_path}")
        # Wait for all indexing threads to finish before cleaning up temp files
        for t in threads:
            t.join()
        # Store projects in session for later use
        session['projects'] = projects
        print('Upload and processing successful')
        shutil.rmtree(temp_dir)
        # Return JSON if AJAX/fetch, else redirect to index
        if request.is_json or request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({
                'message': 'File uploaded and processed successfully',
                'projects': projects
            })
        else:
            return redirect(url_for('index'))
    except Exception as e:
        print(f"Upload error: {str(e)}")  # Debug log
        return jsonify({'error': str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze_project():
    print('[DEBUG] /analyze endpoint called')
    data = request.json
    print(f'[DEBUG] Incoming request data: {data}')
    project_name = data.get('project_name')
    if not project_name:
        print('[ERROR] No project name provided in request')
        return jsonify({'error': 'No project name provided'}), 400

    # Retrieve project and knowledge vectorstore paths
    project = db_manager.get_project(project_name)
    print(f'[DEBUG] Project lookup result: {project}')
    if not project:
        print(f'[ERROR] Project not found: {project_name}')
        return jsonify({'error': 'Project not found'}), 404

    project_vectorstore_path = project.get('clip_vectorstore_path')
    knowledge_vectorstore_path = db_manager.get_knowledge_clip_vectorstore_path()
    print(f'[DEBUG] Project vectorstore path: {project_vectorstore_path}')
    print(f'[DEBUG] Knowledge vectorstore path: {knowledge_vectorstore_path}')
    if not project_vectorstore_path or not knowledge_vectorstore_path:
        print('[ERROR] Vectorstore paths missing')
        return jsonify({'error': 'Vectorstore paths missing'}), 400

    # Load context prompt (adjust path as needed)
    try:
        with open(CONTEXT_TEMPLATE_PATH, 'r', encoding='utf-8') as f:
            context_prompt = f.read()
        print('[DEBUG] Loaded context prompt successfully')
    except Exception as e:
        print(f'[ERROR] Failed to load context prompt: {e}')
        context_prompt = ""

    # Run analysis
    print('[DEBUG] Calling evaluation_agent.analyze...')
    feedback = evaluation_agent.analyze(
        knowledge_vectorstore_path=knowledge_vectorstore_path,
        project_vectorstore_path=project_vectorstore_path,
        context_prompt=context_prompt
    )
    print(f'[DEBUG] Feedback generated: {feedback[:500]}...')  # Print first 500 chars

    # Save feedback to database
    print('[DEBUG] Saving feedback to database...')
    db_manager.update_feedback(
        project_name=project_name,
        feedback=feedback
    )
    print('[DEBUG] Feedback saved successfully')

    return jsonify({'feedback': feedback})

@app.route('/evaluations/<project_name>', methods=['GET'])
def get_evaluation(project_name):
    try:
        evaluation = db_manager.get_project(project_name)
        if evaluation:
            return jsonify(evaluation)
        return jsonify({'error': 'Evaluation not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/evaluations', methods=['GET'])
def get_all_evaluations():
    try:
        evaluations = db_manager.get_all_projects()
        return jsonify(evaluations)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/feedback')
def feedback():
    return render_template('feedback.html')

@app.route('/export/json')
def export_json_route():
    export_to_json('data/feedback.csv', 'data/feedback.json')
    return send_file('data/feedback.json', as_attachment=True)

@app.route('/export/pdf')
def export_pdf_route():
    export_to_pdf('data/feedback.csv', 'data/feedback.pdf')
    return send_file('data/feedback.pdf', as_attachment=True)

@app.route('/reset', methods=['POST'])
def reset():
    """Reset all data and clear uploads."""
    try:
        # Clear database
        db_manager.clear_all_data()
        
        # Clear upload directory
        upload_dir = app.config['UPLOAD_FOLDER']
        for filename in os.listdir(upload_dir):
            file_path = os.path.join(upload_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Error deleting {file_path}: {str(e)}')
        
        # Clear RAG storage
        rag_dir = 'storage/projects_rag'
        if os.path.exists(rag_dir):
            shutil.rmtree(rag_dir)
            os.makedirs(rag_dir)

        # --- Clear all knowledge documents and DB records ---
        # Delete all files in knowledge/ directory
        knowledge_dir = KNOWLEDGE_FOLDER
        for filename in os.listdir(knowledge_dir):
            file_path = os.path.join(knowledge_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Error deleting knowledge file {file_path}: {str(e)}')
        # Remove all records from knowledge_docs table
        conn = sqlite3.connect('data/evaluations.db')
        c = conn.cursor()
        c.execute('DELETE FROM knowledge_docs')
        conn.commit()
        conn.close()
        # --- End knowledge doc cleanup ---
        
        # Return JSON if AJAX/fetch, else redirect
        if request.is_json or request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'status': 'success'})
        else:
            return redirect(url_for('index'))
    except Exception as e:
        if request.is_json or request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'status': 'fail', 'message': str(e)}), 500
        else:
            return redirect(url_for('index'))

@app.route('/view_knowledge/<doc_id>')
def view_knowledge(doc_id):
    conn = sqlite3.connect('data/evaluations.db')
    c = conn.cursor()
    c.execute("SELECT filepath FROM knowledge_docs WHERE id=?", (doc_id,))
    row = c.fetchone()
    conn.close()
    if row and os.path.exists(row[0]):
        return send_file(row[0], as_attachment=False)
    return redirect(url_for('index'))

@app.route('/upload_brief', methods=['POST'])
def upload_brief():
    import time as _time
    start_total = _time.time()
    if 'brief' not in request.files:
        print('[upload_brief] No brief in request.files')
        return redirect(url_for('index'))
    file = request.files['brief']
    if file.filename == '':
        print('[upload_brief] No filename in file')
        return redirect(url_for('index'))
    if file and file.filename.lower().endswith('.pdf'):
        filename = secure_filename(file.filename)
        save_path = os.path.join(KNOWLEDGE_FOLDER, filename)
        print(f'[upload_brief] Saving file to {save_path}')
        start_upload = _time.time()
        file.save(save_path)
        end_upload = _time.time()
        print(f'[upload_brief] File save time: {end_upload - start_upload:.2f} seconds')
        def vectorise_and_generate():
            try:
                start_vectorise = _time.time()
                print('[vectorise_and_generate] Start')
                # 1. Vectorise and append to multimodal RAG (append, do not overwrite)
                doc_id = None
                conn = sqlite3.connect('data/evaluations.db')
                c = conn.cursor()
                c.execute("SELECT id FROM knowledge_docs WHERE filename=?", (filename,))
                row = c.fetchone()
                if row:
                    doc_id = row[0]
                else:
                    doc_id = str(uuid.uuid4())
                conn.close()
                print(f'[vectorise_and_generate] Using doc_id: {doc_id}')
                try:
                    rag_store_dir = os.path.join(KNOWLEDGE_FOLDER, 'rag_store')
                    os.makedirs(rag_store_dir, exist_ok=True)
                    process_knowledge_pdf(save_path, doc_id, target_dir=rag_store_dir)
                    print('[vectorise_and_generate] process_knowledge_pdf success')
                    # Update knowledge table with vectorstore path
                    db_manager.set_knowledge_clip_vectorstore_path(os.path.join(rag_store_dir, 'clip.index'))
                except Exception as e:
                    print(f'[vectorise_and_generate] process_knowledge_pdf failed: {e}')
                    raise
                end_vectorise = _time.time()
                print(f'[vectorise_and_generate] Vectorisation time: {end_vectorise - start_vectorise:.2f} seconds')
                # 2. Generate context using the prompt template and Gemma3 LLM via RAG
                from tools.assessment_utils import AssessmentManager
                assessment_manager = AssessmentManager()
                try:
                    with open(PROMPT_TEMPLATE_PATH, 'r', encoding='utf-8') as f:
                        prompt_template = f.read()
                    from langchain_community.document_loaders import PyPDFLoader
                    loader_pdf = PyPDFLoader(save_path)
                    documents = loader_pdf.load()
                    brief_content = "\n".join(doc.page_content for doc in documents)
                    prompt = prompt_template.replace('{brief_content}', brief_content)
                    print('[vectorise_and_generate] Invoking LLM')
                    context_result = assessment_manager.llm.invoke(prompt)
                    print('[vectorise_and_generate] LLM invoke success')
                    with open(CONTEXT_TEMPLATE_PATH, 'w', encoding='utf-8') as f:
                        f.write(context_result)
                except Exception as e:
                    print(f'[vectorise_and_generate] LLM/context failed: {e}')
                    raise
                # 3. Save or update knowledge doc metadata
                start_db = _time.time()
                try:
                    conn = sqlite3.connect('data/evaluations.db')
                    c = conn.cursor()
                    if row:
                        c.execute("UPDATE knowledge_docs SET filepath=?, upload_date=? WHERE id=?",
                                  (save_path, time.strftime('%Y-%m-%d %H:%M:%S'), doc_id))
                    else:
                        c.execute("INSERT INTO knowledge_docs (id, filename, filepath, upload_date) VALUES (?, ?, ?, ?)",
                                  (doc_id, filename, save_path, time.strftime('%Y-%m-%d %H:%M:%S')))
                    conn.commit()
                    conn.close()
                    end_db = _time.time()
                    print(f'[vectorise_and_generate] DB update success, time: {end_db - start_db:.2f} seconds')
                except Exception as e:
                    print(f'[vectorise_and_generate] DB update failed: {e}')
                    raise
                end_total = _time.time()
                print(f'[vectorise_and_generate] Total operation time (vectorise+LLM+DB): {end_total - start_vectorise:.2f} seconds')
            except Exception as e:
                print(f'[vectorise_and_generate] Exception: {e}')
        thread = threading.Thread(target=vectorise_and_generate)
        thread.start()
        thread.join()  # Wait for thread to finish before allowing file access
        end_total = _time.time()
        print(f'[upload_brief] Total time from upload to return: {end_total - start_total:.2f} seconds')
    return redirect(url_for('index'))

@app.route('/delete_knowledge/<doc_id>', methods=['POST'])
def delete_knowledge(doc_id):
    conn = sqlite3.connect('data/evaluations.db')
    c = conn.cursor()
    c.execute("SELECT filepath FROM knowledge_docs WHERE id=?", (doc_id,))
    row = c.fetchone()
    if row:
        try:
            os.remove(row[0])
        except Exception:
            pass
    c.execute("DELETE FROM knowledge_docs WHERE id=?", (doc_id,))
    conn.commit()
    conn.close()
    return redirect(url_for('index'))

@app.route('/delete_report/<project_name>', methods=['POST'])
def delete_report(project_name):
    print(f'[delete_report] Called for project: {project_name}')
    project_dir = os.path.join('storage', 'projects_rag', project_name)
    if os.path.exists(project_dir):
        print(f'[delete_report] Removing directory: {project_dir}')
        shutil.rmtree(project_dir)
    else:
        print(f'[delete_report] Directory not found: {project_dir}')
    db_manager.clear_project_data(project_name)
    print(f'[delete_report] Cleared database entries for: {project_name}')
    return redirect(url_for('index'))

@app.route('/get_template', methods=['GET'])
def get_template():
    try:
        with open(PROMPT_TEMPLATE_PATH, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception:
        content = ""
    return jsonify({'template_content': content})

@app.route('/save_template', methods=['POST'])
def save_template():
    data = request.get_json()
    content = data.get('template_content', '')
    try:
        with open(PROMPT_TEMPLATE_PATH, 'w', encoding='utf-8') as f:
            f.write(content)
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_context', methods=['GET'])
def get_context():
    try:
        with open(CONTEXT_TEMPLATE_PATH, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception:
        content = ""
    return jsonify({'context_content': content})

@app.route('/save_context', methods=['POST'])
def save_context():
    data = request.get_json()
    content = data.get('context_content', '')
    try:
        with open(CONTEXT_TEMPLATE_PATH, 'w', encoding='utf-8') as f:
            f.write(content)
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/save_feedback/<project_name>', methods=['POST'])
def save_feedback(project_name):
    data = request.get_json()
    feedback = data.get('feedback', '')
    try:
        db_manager.update_feedback(
            project_name=project_name,
            feedback=feedback
        )
        return jsonify({'status': 'success', 'refresh': True})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/db_schema')
def db_schema():
    db_path = 'storage/evaluations.db'
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    tables = [row[0] for row in cursor.fetchall()]
    schema = {'tables': []}
    for table in tables:
        cursor.execute(f'PRAGMA table_info({table})')
        columns = [row[1] for row in cursor.fetchall()]
        schema['tables'].append({'name': table, 'columns': columns})
    conn.close()
    return jsonify(schema)

@app.route('/db_table_rows/<table>')
def db_table_rows(table):
    db_path = 'storage/evaluations.db'
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        cursor.execute(f'SELECT * FROM {table} LIMIT 50')
        rows = cursor.fetchall()
        columns = rows[0].keys() if rows else []
        data_rows = [dict(row) for row in rows]
        return jsonify({'columns': columns, 'rows': data_rows})
    except Exception as e:
        return jsonify({'columns': [], 'rows': [], 'error': str(e)})
    finally:
        conn.close()

def load_chatbot_rag(force_reload=False):
    global chatbot_rag, chatbot_rag_last_updated
    # If already loaded and not forced, return cached
    if chatbot_rag and not force_reload:
        return chatbot_rag

    db = DatabaseManager()
    knowledge_path = db.get_knowledge_clip_vectorstore_path()
    project_paths = [p['clip_vectorstore_path'] for p in db.get_all_projects() if p['clip_vectorstore_path']]
    all_vectorstore_paths = [knowledge_path] + project_paths

    all_vectorstores = []
    for path in all_vectorstore_paths:
        if path and os.path.exists(path):
            vs = CLIPVectorStore(path)
            print(f"[DEBUG] Vectorstore: {path}")
            for meta in getattr(vs, 'metadatas', []):
                print(meta.get('source', ''))
            all_vectorstores.append(vs)

    all_feedback = [p['feedback'] for p in db.get_all_projects() if p['feedback']]

    class UnifiedRAG:
        def __init__(self):
            self.vectorstores = []  # List of CLIPVectorStore
            self.text_chunks = []   # List of dicts: {'text': ..., 'metadata': ...}
            self.feedback_embeddings = []  # List of np.ndarray

        def add_vectorstore(self, vs):
            self.vectorstores.append(vs)

        def add_text(self, text, metadata=None):
            self.text_chunks.append({'text': text, 'metadata': metadata or {}})
            # If possible, embed the feedback now for fast retrieval
            if hasattr(CLIPVectorStore, 'embed_text'):
                try:
                    emb = CLIPVectorStore.embed_text(text) if isinstance(CLIPVectorStore.embed_text, staticmethod) else CLIPVectorStore('').embed_text(text)
                    self.feedback_embeddings.append(emb)
                except Exception as e:
                    print(f'[UnifiedRAG] Error embedding feedback: {e}')
            else:
                self.feedback_embeddings.append(None)

        def retrieve(self, query, top_k=5):
            results = []
            # 1. Embed the query (if possible)
            query_emb = None
            if hasattr(CLIPVectorStore, 'embed_text'):
                try:
                    query_emb = CLIPVectorStore.embed_text(query) if isinstance(CLIPVectorStore.embed_text, staticmethod) else CLIPVectorStore('').embed_text(query)
                except Exception as e:
                    print(f'[UnifiedRAG] Error embedding query: {e}')
            # 2. Retrieve from all vectorstores (if they have a search/similarity method)
            vectorstore_results = []
            for vs in self.vectorstores:
                if hasattr(vs, 'search'):
                    try:
                        vectorstore_results.extend(vs.search(query, top_k=top_k))
                    except Exception as e:
                        print(f'[UnifiedRAG] Error searching vectorstore: {e}')
                elif hasattr(vs, 'similarity_search'):
                    try:
                        vectorstore_results.extend(vs.similarity_search(query, k=top_k))
                    except Exception as e:
                        print(f'[UnifiedRAG] Error in similarity_search: {e}')
            # 3. Retrieve from feedback text chunks using embedding similarity
            feedback_results = []
            if query_emb is not None and self.feedback_embeddings:
                # Compute cosine similarity between query_emb and each feedback embedding
                similarities = []
                for i, emb in enumerate(self.feedback_embeddings):
                    if emb is not None:
                        sim = float(np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb) + 1e-8))
                        similarities.append((sim, i))
                # Sort by similarity
                similarities.sort(reverse=True)
                for sim, idx in similarities[:top_k]:
                    chunk = self.text_chunks[idx]
                    feedback_results.append({'text': chunk['text'], 'metadata': chunk['metadata'], 'similarity': sim})
            else:
                # Fallback: keyword match, then top_k
                lowered_query = query.lower()
                for chunk in self.text_chunks:
                    if lowered_query in chunk['text'].lower():
                        feedback_results.append({'text': chunk['text'], 'metadata': chunk['metadata'], 'similarity': 1.0})
                if len(feedback_results) < top_k:
                    for chunk in self.text_chunks:
                        if chunk not in feedback_results:
                            feedback_results.append({'text': chunk['text'], 'metadata': chunk['metadata'], 'similarity': 0.0})
                        if len(feedback_results) >= top_k:
                            break
            # 4. Merge and rank all results by similarity (if available), deduplicate by text
            all_results = []
            # Vectorstore results may not have similarity, so assign a default
            for r in vectorstore_results:
                all_results.append({'text': r.get('text', ''), 'metadata': r.get('metadata', {}), 'similarity': r.get('similarity', 1.0)})
            all_results.extend(feedback_results)
            # Deduplicate by text, keep highest similarity
            dedup = {}
            for r in all_results:
                t = r['text']
                if t not in dedup or r['similarity'] > dedup[t]['similarity']:
                    dedup[t] = r
            ranked = sorted(dedup.values(), key=lambda x: x['similarity'], reverse=True)
            return ranked[:top_k]

    unified_rag = UnifiedRAG()
    for vs in all_vectorstores:
        unified_rag.add_vectorstore(vs)
    for feedback in all_feedback:
        unified_rag.add_text(feedback, metadata={'type': 'feedback'})

    chatbot_rag = unified_rag
    chatbot_rag_last_updated = time.time()
    return chatbot_rag

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    question = data.get('question', '').strip()
    if not question:
        return jsonify({'error': 'No question provided'}), 400

    # Load the unified RAG (persistent, refreshed as needed)
    rag = load_chatbot_rag()

    # Retrieve relevant context (pseudo-code, adapt to your RAG API)
    # For example, you might have: rag.retrieve(question, top_k=5)
    context_chunks = rag.retrieve(question, top_k=5)
    context_text = "\n\n".join(chunk['text'] for chunk in context_chunks)

    # Compose prompt for LLM
    prompt = (
        f"You are an expert assistant. Use the following context from assessment briefs, student submissions, and feedback to answer the user's question.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )

    # Call your LLM (e.g., assessment_manager.llm.invoke or similar)
    answer = assessment_manager.llm.invoke(prompt)

    # Convert newlines to <br> for better frontend formatting
    answer_html = answer.replace('\n', '<br>') if answer else ''

    return jsonify({'answer': answer_html})

if __name__ == '__main__':
    db_manager._init_db()  # Initialize database tables
    socketio.run(app, debug=True)