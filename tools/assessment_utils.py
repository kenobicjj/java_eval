import os
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
import shutil
import json
from pathlib import Path
import re
import time
import gc
import csv
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from tools.clip_vector_store import CLIPVectorStore
import numpy as np

RAG_STORE_DIR = "storage/rag_store"
PROJECTS_RAG_DIR = "storage/projects_rag"
STORAGE_DIR = "storage"

CRITERIA_EXTRACTION_TEMPLATE = """
You are an expert in analyzing assessment briefs and creating evaluation criteria. Based on the following assessment brief content, create a structured set of evaluation criteria.

Assessment Brief Content:
{brief_content}

Additional Instructions:
{preamble}

Create evaluation criteria focusing on:
1. Code Structure and Organization
2. Code Quality and Best Practices
3. Functionality and Requirements
4. Error Handling and Robustness

Consider these aspects:
{postamble}

Return a JSON structure with categories and specific criteria. Use exactly this format (do not include any other text):

{{
    "code_structure": {{
        "organization": [
            "Classes follow single responsibility principle",
            "Clear separation of concerns"
        ],
        "design": [
            "Appropriate use of inheritance and interfaces",
            "Modular code organization"
        ]
    }},
    "code_quality": {{
        "practices": [
            "Consistent naming conventions",
            "Proper documentation"
        ],
        "maintainability": [
            "Code is well-commented",
            "Methods are concise and focused"
        ]
    }},
    "functionality": {{
        "requirements": [
            "All required features are implemented",
            "Features work as specified"
        ],
        "robustness": [
            "Proper error handling",
            "Input validation"
        ]
    }}
}}

Ensure your response contains only the JSON structure with specific, measurable criteria for Java code evaluation.
"""

class AssessmentManager:
    def __init__(self, criteria_path: str = None):
        """
        Initialize the assessment manager.
        
        Args:
            criteria_path: Path to the JSON file containing assessment criteria
        """
        if criteria_path is None:
            self.criteria_path = os.path.join(STORAGE_DIR, "assessment_criteria.json")
        else:
            self.criteria_path = criteria_path
            
        self.vector_store = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self._ensure_directories()
        
        # Initialize Ollama with Gemma model
        self.llm = OllamaLLM(
            model="gemma3",
            temperature=0.1,
            num_ctx=4096,
            repeat_penalty=1.1,
            num_predict=2048,
            stop=["}"],
            system="You are an expert code evaluator that analyzes assessment briefs and creates structured evaluation criteria. Always respond with valid JSON."
        )
        
        self.criteria_prompt = PromptTemplate(
            template=CRITERIA_EXTRACTION_TEMPLATE,
            input_variables=["brief_content", "preamble", "postamble"]
        )
        self.criteria = self._load_criteria()

    def _ensure_directories(self):
        """Ensure necessary directories exist."""
        os.makedirs(RAG_STORE_DIR, exist_ok=True)
        os.makedirs(PROJECTS_RAG_DIR, exist_ok=True)
        os.makedirs(STORAGE_DIR, exist_ok=True)

    def _load_criteria(self) -> Dict:
        """Load assessment criteria from JSON file."""
        # Use CLIPVectorStore for vector store
        brief_dir = os.path.dirname(self.criteria_path)
        rag_dir = os.path.join(brief_dir, "rag_store")
        index_path = os.path.join(rag_dir, "clip.index")
        if os.path.exists(index_path):
            try:
                self.vector_store = CLIPVectorStore(index_path)
            except Exception as e:
                print(f"Error loading CLIP vector store from {index_path}: {e}")
        if not os.path.exists(self.criteria_path):
            return {}  # Return empty dict if no criteria file exists
        
        with open(self.criteria_path, 'r') as f:
            return json.load(f)
       
        # Generate criteria from the brief
        self._generate_criteria_from_brief(filepath, preamble, postamble)
        
        return filepath

    def _generate_criteria_from_brief(self, pdf_path: str, preamble: str, postamble: str):
        """Generate evaluation criteria by analyzing the assessment brief."""
        try:
            # Load and process the PDF
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            brief_content = "\n".join(doc.page_content for doc in documents)
            
            # Set default values for preamble and postamble if not provided
            preamble = preamble or "Focus on code quality, maintainability, and adherence to requirements."
            postamble = postamble or "Consider industry best practices and common Java coding standards."
            
            # Create the chain using the new pattern
            chain = self.criteria_prompt | self.llm
            
            # Generate criteria
            result = chain.invoke({
                "brief_content": brief_content,
                "preamble": preamble,
                "postamble": postamble
            })
            
            # Clean up the result
            result_str = result.strip()
            
            # Parse and save the criteria
            try:
                criteria = json.loads(result_str)
                with open(self.criteria_path, 'w') as f:
                    json.dump(criteria, f, indent=4)
                self.criteria = criteria
                print("Successfully generated and saved criteria")
            except json.JSONDecodeError as e:
                print(f"Error parsing generated criteria: {e}")
                print(f"Generated content: {result_str}")
                self._create_default_criteria()
        except Exception as e:
            print(f"Error generating criteria: {e}")
            self._create_default_criteria()

    def _create_default_criteria(self):
        """Create and save default criteria if generation fails."""
        default_criteria = {
            "code_structure": {
                "organization": [
                    "Classes follow single responsibility principle",
                    "Clear separation of concerns"
                ],
                "design": [
                    "Appropriate use of inheritance and interfaces",
                    "Modular code organization"
                ]
            },
            "code_quality": {
                "practices": [
                    "Consistent naming conventions",
                    "Proper documentation"
                ],
                "maintainability": [
                    "Code is well-commented",
                    "Methods are concise and focused"
                ]
            },
            "functionality": {
                "requirements": [
                    "All required features are implemented",
                    "Features work as specified"
                ],
                "robustness": [
                    "Proper error handling",
                    "Input validation"
                ]
            }
        }
        with open(self.criteria_path, 'w') as f:
            json.dump(default_criteria, f, indent=4)
        self.criteria = default_criteria

    def _cleanup_vector_store(self):
        """Clean up the existing vector store."""
        try:
            self.vector_store = None
            
            # Force garbage collection to release file handles
            gc.collect()
            time.sleep(0.5)
            
            # Remove the RAG store directory if it exists, with retry for WinError 32
            if os.path.exists(RAG_STORE_DIR):
                for attempt in range(5):
                    try:
                        shutil.rmtree(RAG_STORE_DIR)
                        break
                    except Exception as e:
                        if hasattr(e, 'winerror') and e.winerror == 32:
                            print(f"WinError 32: File in use, retrying... (attempt {attempt+1})")
                            time.sleep(1)
                            gc.collect()
                        else:
                            print(f"Warning: Could not remove RAG store directory: {e}")
                            break
        except Exception as e:
            print(f"Warning: Error during vector store cleanup: {e}")

    def create_rag(self, pdf_path: str):
        """Create RAG from the assessment brief PDF and store it alongside the knowledge file."""
        # Clean up existing vector store
        self._cleanup_vector_store()
        # Store vectorstore next to the PDF
        rag_dir = os.path.join(os.path.dirname(pdf_path), "rag_store")
        os.makedirs(rag_dir, exist_ok=True)
        # Load and split the PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        splits = self.text_splitter.split_documents(documents)
        index_path = os.path.join(rag_dir, "clip.index")
        store = CLIPVectorStore(index_path)
        for doc in splits:
            emb = store.embed_text(doc.page_content)
            store.add(emb, {"source": pdf_path, "type": "text", **doc.metadata})
        self.vector_store = store

    def store_project_in_rag(self, project_path: str, project_name: str):
        """Store a project's Java files in a dedicated RAG collection."""
        project_rag_dir = os.path.join(PROJECTS_RAG_DIR, project_name)
        
        # Clear existing project RAG if it exists
        if os.path.exists(project_rag_dir):
            shutil.rmtree(project_rag_dir)
        os.makedirs(project_rag_dir)

        documents = []
        # Process each Java file in the project
        for root, _, files in os.walk(project_path):
            for file in files:
                if file.endswith('.java'):
                    file_path = os.path.join(root, file)
                    try:
                        loader = TextLoader(file_path)
                        file_docs = loader.load()
                        # Add metadata about the file
                        for doc in file_docs:
                            doc.metadata['file_name'] = file
                            doc.metadata['project_name'] = project_name
                        documents.extend(file_docs)
                    except Exception as e:
                        print(f"Error loading file {file_path}: {str(e)}")

        if documents:
            splits = self.text_splitter.split_documents(documents)
            index_path = os.path.join(project_rag_dir, "clip.index")
            store = CLIPVectorStore(index_path)
            for doc in splits:
                emb = store.embed_text(doc.page_content)
                store.add(emb, {"source": file_path, "type": "text", **doc.metadata})
            self.vector_store = store
            return project_rag_dir
        return None

    def compare_project(self, project_path: str, project_name: str, structure_context: str = None) -> Dict:
        """
        Compare a project against the assessment criteria using RAG and static analysis.
        
        Args:
            project_path: Path to the project directory
            project_name: Name of the project
            structure_context: Optional context from structure analysis
            
        Returns:
            Dictionary containing:
                - criteria_matches: Dictionary of criteria matches
                - summary: Summary of the evaluation
        """
        # Try to find a rag_store next to the assessment brief in knowledge/
        knowledge_dir = os.path.abspath('knowledge')
        rag_dir = os.path.join(knowledge_dir, 'rag_store')
        index_path = os.path.join(rag_dir, 'clip.index')
        if not self.vector_store and os.path.exists(index_path):
            try:
                self.vector_store = CLIPVectorStore(index_path)
            except Exception as e:
                print(f"Error loading CLIP vector store from {index_path}: {e}")
        if not self.vector_store:
            raise ValueError("Assessment criteria RAG not initialized. Please upload an assessment brief first.")
        
        # Load and analyze project files
        project_analysis = self._analyze_project(project_path)
        
        # Build context for evaluation
        context = self._build_evaluation_context(
            project_name=project_name,
            structure_context=structure_context,
            project_analysis=project_analysis
        )
        
        # Match against assessment criteria
        criteria_matches = self._match_criteria(context)
        
        # Generate detailed summary
        summary = self._generate_detailed_summary(
            project_name=project_name,
            structure_context=structure_context,
            project_analysis=project_analysis,
            criteria_matches=criteria_matches
        )
        
        return {
            'criteria_matches': criteria_matches,
            'summary': summary,
            'static_analysis': project_analysis
        }

    def _analyze_project(self, project_path: str) -> Dict:
        """Perform static analysis of the project."""
        analysis = {
            'files': {},
            'metrics': {
                'total_lines': 0,
                'comment_lines': 0,
                'code_lines': 0,
                'class_count': 0,
                'method_count': 0,
                'complexity': 0
            },
            'patterns': {
                'design_patterns': [],
                'anti_patterns': []
            },
            'issues': []
        }
        
        for root, _, files in os.walk(project_path):
            for file in files:
                if file.endswith('.java'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            file_analysis = self._analyze_java_file(content)
                            analysis['files'][file] = file_analysis
                            
                            # Update overall metrics
                            for key in ['total_lines', 'comment_lines', 'code_lines', 
                                      'class_count', 'method_count', 'complexity']:
                                analysis['metrics'][key] += file_analysis['metrics'][key]
                            
                            # Collect patterns and issues
                            analysis['patterns']['design_patterns'].extend(
                                file_analysis['patterns']['design_patterns']
                            )
                            analysis['patterns']['anti_patterns'].extend(
                                file_analysis['patterns']['anti_patterns']
                            )
                            analysis['issues'].extend(file_analysis['issues'])
                    except Exception as e:
                        analysis['issues'].append({
                            'file': file,
                            'type': 'error',
                            'message': f"Error analyzing file: {str(e)}"
                        })
        
        return analysis

    def _analyze_java_file(self, content: str) -> Dict:
        """Analyze a single Java file."""
        analysis = {
            'metrics': {
                'total_lines': len(content.splitlines()),
                'comment_lines': 0,
                'code_lines': 0,
                'class_count': 0,
                'method_count': 0,
                'complexity': 0
            },
            'patterns': {
                'design_patterns': [],
                'anti_patterns': []
            },
            'issues': []
        }
        
        # Count comments and code lines
        in_block_comment = False
        lines = content.splitlines()
        for line in lines:
            line = line.strip()
            if line.startswith('/*'):
                in_block_comment = True
                analysis['metrics']['comment_lines'] += 1
            elif line.endswith('*/'):
                in_block_comment = False
                analysis['metrics']['comment_lines'] += 1
            elif in_block_comment:
                analysis['metrics']['comment_lines'] += 1
            elif line.startswith('//'):
                analysis['metrics']['comment_lines'] += 1
            elif line:
                analysis['metrics']['code_lines'] += 1
        
        # Count classes and methods
        analysis['metrics']['class_count'] = len(re.findall(r'\bclass\s+\w+', content))
        analysis['metrics']['method_count'] = len(re.findall(r'\b(public|private|protected)\s+\w+\s+\w+\s*\(', content))
        
        # Calculate complexity (simplified)
        analysis['metrics']['complexity'] = len(re.findall(r'\b(if|while|for|catch)\b', content))
        
        # Detect patterns
        patterns = {
            'Singleton': r'private\s+static\s+\w+\s+instance',
            'Factory': r'(create|get)\w+Instance',
            'Observer': r'(add|remove)Listener',
            'Builder': r'build\(\)',
        }
        
        anti_patterns = {
            'God Class': analysis['metrics']['method_count'] > 20,
            'Long Method': bool(re.search(r'{[^}]{1000,}}', content)),
            'Magic Numbers': bool(re.search(r'\b\d{4,}\b', content))
        }
        
        for pattern, regex in patterns.items():
            if re.search(regex, content):
                analysis['patterns']['design_patterns'].append(pattern)
        
        for pattern, condition in anti_patterns.items():
            if condition:
                analysis['patterns']['anti_patterns'].append(pattern)
        
        # Identify potential issues
        if analysis['metrics']['comment_lines'] / max(analysis['metrics']['total_lines'], 1) < 0.1:
            analysis['issues'].append({
                'type': 'warning',
                'message': 'Low comment density'
            })
        
        if analysis['metrics']['complexity'] / max(analysis['metrics']['method_count'], 1) > 5:
            analysis['issues'].append({
                'type': 'warning',
                'message': 'High average method complexity'
            })
        
        return analysis

    def _build_evaluation_context(self, project_name: str, structure_context: str, 
                                project_analysis: Dict) -> str:
        """Build a comprehensive context for evaluation."""
        context_parts = [
            f"Project: {project_name}",
            
            "Structure Analysis:",
            structure_context or "No structure analysis provided.",
            
            "Static Analysis:",
            f"- Total Files: {len(project_analysis['files'])}",
            f"- Total Lines: {project_analysis['metrics']['total_lines']}",
            f"- Code Lines: {project_analysis['metrics']['code_lines']}",
            f"- Comment Lines: {project_analysis['metrics']['comment_lines']}",
            f"- Classes: {project_analysis['metrics']['class_count']}",
            f"- Methods: {project_analysis['metrics']['method_count']}",
            f"- Average Complexity: {project_analysis['metrics']['complexity'] / max(project_analysis['metrics']['method_count'], 1):.2f}",
            
            "Design Patterns Found:",
            ", ".join(project_analysis['patterns']['design_patterns']) or "None detected",
            
            "Anti-Patterns Found:",
            ", ".join(project_analysis['patterns']['anti_patterns']) or "None detected",
            
            "Issues Found:",
            "\n".join(f"- {issue['message']}" for issue in project_analysis['issues']) or "No issues found"
        ]
        
        return "\n".join(context_parts)

    def _match_criteria(self, context: str) -> Dict:
        """Match project against assessment criteria."""
        matches = []
        if not self.vector_store:
            return matches
        emb = self.vector_store.embed_text(context)
        # Use FAISS to get top 5 similar items
        D, I = self.vector_store.index.search(np.array([emb]).astype(np.float32), 5)
        for idx, dist in zip(I[0], D[0]):
            if idx < len(self.vector_store.metadatas):
                meta = self.vector_store.metadatas[idx]
                if dist < 0.8:  # Lower distance = better match
                    matches.append({
                        'criterion': meta.get('source', ''),
                        'score': 1 - dist,
                        'category': meta.get('category', 'general')
                    })
        return matches

    def _generate_detailed_summary(self, project_name: str, structure_context: str,
                                 project_analysis: Dict, criteria_matches: List) -> str:
        """Generate a detailed evaluation summary."""
        summary_parts = []
        
        # Project Overview
        summary_parts.append(f"Project Evaluation: {project_name}")
        summary_parts.append("\nStructure Analysis:")
        summary_parts.append(structure_context or "No structure analysis available.")
        
        # Code Metrics
        summary_parts.append("\nCode Metrics:")
        metrics = project_analysis['metrics']
        summary_parts.append(f"- Total Files: {len(project_analysis['files'])}")
        summary_parts.append(f"- Lines of Code: {metrics['code_lines']}")
        summary_parts.append(f"- Comment Lines: {metrics['comment_lines']}")
        summary_parts.append(f"- Classes: {metrics['class_count']}")
        summary_parts.append(f"- Methods: {metrics['method_count']}")
        comment_ratio = metrics['comment_lines'] / max(metrics['total_lines'], 1)
        summary_parts.append(f"- Documentation Ratio: {comment_ratio:.1%}")
        
        # Design Patterns
        if project_analysis['patterns']['design_patterns']:
            summary_parts.append("\nDesign Patterns Identified:")
            for pattern in project_analysis['patterns']['design_patterns']:
                summary_parts.append(f"- {pattern}")
        
        # Issues and Recommendations
        if project_analysis['issues']:
            summary_parts.append("\nAreas for Improvement:")
            for issue in project_analysis['issues']:
                summary_parts.append(f"- {issue['message']}")
        
        # Criteria Matching
        summary_parts.append("\nAssessment Criteria Alignment:")
        for match in criteria_matches:
            score_text = "Strong" if match['score'] > 0.8 else "Moderate" if match['score'] > 0.6 else "Weak"
            summary_parts.append(f"- {score_text} alignment with: {match['criterion']}")
        
        return "\n".join(summary_parts)

    def evaluate_project(self, project_path: str) -> Dict:
        """
        Evaluate a Java project against assessment criteria using RAG.
        
        Args:
            project_path: Path to the Java project
            
        Returns:
            Dictionary containing evaluation results
        """
        # Get all Java files in the project
        java_files = list(Path(project_path).rglob("*.java"))
        
        # Initialize results dictionary
        results = {
            "criteria_matches": {},
            "summary": ""
        }
        
        # Read and analyze all Java files
        code_content = []
        for file_path in java_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                code_content.append(f.read())
        
        # Combine all code content
        full_code = "\n".join(code_content)
        
        # Evaluate against each criterion category
        for category, subcategories in self.criteria.items():
            results["criteria_matches"][category] = {}
            
            for subcategory, checks in subcategories.items():
                matches = []
                
                for check in checks:
                    # Use regex and pattern matching to evaluate criteria
                    match_score = self._evaluate_criterion(full_code, check)
                    matches.append({
                        "criterion": check,
                        "score": match_score,
                        "details": self._get_match_details(full_code, check)
                    })
                
                results["criteria_matches"][category][subcategory] = matches
        
        # Generate summary
        results["summary"] = self._generate_summary(results["criteria_matches"])
        
        return results
    
    def _evaluate_criterion(self, code: str, criterion: str) -> float:
        """
        Evaluate code against a specific criterion.
        
        Args:
            code: Source code content
            criterion: Criterion to evaluate against
            
        Returns:
            Score between 0 and 1 indicating how well the criterion is met
        """
        # Simple pattern matching for demonstration
        # In a real implementation, this would use more sophisticated NLP/ML techniques
        patterns = {
            "class_organization": r"class\s+\w+(\s+extends\s+\w+)?(\s+implements\s+\w+)?",
            "method_design": r"(public|private|protected)\s+\w+\s+\w+\s*\([^)]*\)",
            "naming_conventions": r"[a-z][a-zA-Z0-9]*|[A-Z][a-zA-Z0-9]*",
            "error_handling": r"try|catch|throw|throws",
            "core_features": r"@Override|interface|implements",
            "robustness": r"if|switch|while|for"
        }
        
        # Find the most relevant pattern for the criterion
        pattern = None
        for key, regex in patterns.items():
            if key.lower() in criterion.lower():
                pattern = regex
                break
        
        if pattern:
            matches = len(re.findall(pattern, code))
            # Normalize score between 0 and 1
            return min(1.0, matches / 10)
        
        return 0.5  # Default score if no pattern matches
    
    def _get_match_details(self, code: str, criterion: str) -> str:
        """
        Get detailed information about how a criterion was matched.
        
        Args:
            code: Source code content
            criterion: Criterion that was evaluated
            
        Returns:
            String containing match details
        """
        # This would be more sophisticated in a real implementation
        return f"Evaluated criterion: {criterion}"
    
    def _load_project_files(self, project_path: str) -> str:
        """Load all project files and return their content as a single string."""
        project_content = ""
        for root, _, files in os.walk(project_path):
            for file in files:
                if file.endswith(('.java', '.txt', '.md')):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        project_content += f.read() + "\n"
        return project_content

    def __del__(self):
        """Cleanup when the object is destroyed."""
        self._cleanup_vector_store() 

def generate_feedback_csv(evaluations: List[Dict], csv_path: str = "data/feedback.csv"):
    """
    Generate a feedback CSV file from a list of evaluation results.

    Args:
        evaluations: List of dictionaries containing evaluation results.
        csv_path: Path to save the generated CSV file.
    """
    if not evaluations:
        print("No evaluations provided for feedback CSV.")
        return

    # Determine CSV columns from the first evaluation
    columns = ["project_name", "summary"]
    # Optionally add more columns based on your evaluation structure

    with open(csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        writer.writeheader()
        for eval in evaluations:
            row = {
                "project_name": eval.get("project_name", ""),
                "summary": eval.get("summary", ""),
            }
            writer.writerow(row)
    print(f"Feedback CSV generated at {csv_path}")