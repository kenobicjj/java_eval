import os
import zipfile
import shutil
from typing import List, Dict
import tempfile

def build_file_hierarchy(directory: str) -> Dict:
    """Build a hierarchical representation of the directory structure."""
    hierarchy = {'type': 'directory', 'name': os.path.basename(directory), 'children': []}
    
    try:
        for item in sorted(os.listdir(directory)):
            path = os.path.join(directory, item)
            if os.path.isdir(path):
                hierarchy['children'].append(build_file_hierarchy(path))
            else:
                hierarchy['children'].append({
                    'type': 'file',
                    'name': item,
                    'extension': os.path.splitext(item)[1][1:] if os.path.splitext(item)[1] else ''
                })
    except Exception as e:
        print(f"Error building hierarchy for {directory}: {str(e)}")
    
    return hierarchy

def handle_zip_submission(zip_path: str) -> List[Dict]:
    """
    Process a ZIP file containing Java projects.
    
    Args:
        zip_path: Path to the ZIP file
        
    Returns:
        List of dictionaries containing project information:
            - name: Project name
            - path: Path to project directory
            - hierarchy: File hierarchy structure
    """
    print(f"Processing ZIP file: {zip_path}")  # Debug log
    
    # Create a temporary directory for extraction
    temp_dir = tempfile.mkdtemp(prefix='java_eval_')
    print(f"Created temp directory: {temp_dir}")  # Debug log
    
    try:
        # Extract the ZIP file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
        print("ZIP file extracted successfully")  # Debug log

        projects = []
        # Walk through the extracted contents to find Java projects
        # for root, dirs, files in os.walk(temp_dir):
        java_files = [f for f in files if f.endswith('.java')]
            
        if java_files:
                # Check if this is a project root (no parent directory has Java files)
                parent_has_java = False
                parent = os.path.dirname(root)
                while parent and parent != temp_dir:
                    if any(f.endswith('.java') for f in os.listdir(parent)):
                        parent_has_java = True
                        break
                    parent = os.path.dirname(parent)
                
                if not parent_has_java:
                    project_name = os.path.basename(root)
                    print(f"Found Java project: {project_name}")  # Debug log
                    
                    # Build file hierarchy
                    hierarchy = build_file_hierarchy(root)
                    print(f"Built hierarchy for: {project_name}")  # Debug log
                    
                    projects.append({
                        'name': project_name,
                        'path': root,
                        'hierarchy': hierarchy
                    })
        
        if not projects:
            print("No Java projects found in ZIP file")  # Debug log
        else:
            print(f"Found {len(projects)} Java projects")  # Debug log
            
            # Clean up the original ZIP file
        os.remove(zip_path)
        print("Removed original ZIP file")  # Debug log
            
        return projects
        
    except Exception as e:
        print(f"Error processing ZIP file: {str(e)}")  # Debug log
        # Clean up temp directory in case of error
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise