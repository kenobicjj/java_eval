import sqlite3
import json
import os
from typing import Dict, List
from datetime import datetime

class DatabaseManager:
    def __init__(self, db_path: str = 'storage/evaluations.db'):
        """Initialize the database manager."""
        os.makedirs('storage', exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize the database tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Unified projects table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS projects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_name TEXT UNIQUE NOT NULL,
                    file_hierarchy TEXT,
                    project_path TEXT,
                    clip_vectorstore_path TEXT,
                    feedback TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create knowledge table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS knowledge (
                    id INTEGER PRIMARY KEY,
                    clip_vectorstore_path TEXT
                )
            ''')
            
            conn.commit()
            print("Database tables initialized successfully")  # Debug log

    def save_project(self, project_name: str, file_hierarchy: Dict = None, project_path: str = None,
                     feedback: str = None, clip_vectorstore_path: str = None):
        """Save a project to the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Convert dictionaries to JSON strings
                hierarchy_json = json.dumps(file_hierarchy) if file_hierarchy else None
                
                # Check if project already exists
                cursor.execute('SELECT id FROM projects WHERE project_name = ?', (project_name,))
                existing = cursor.fetchone()
                
                if existing:
                    # Update existing project
                    cursor.execute('''
                        UPDATE projects 
                        SET file_hierarchy = ?, project_path = ?, clip_vectorstore_path = ?, 
                            feedback = ?, timestamp = CURRENT_TIMESTAMP
                        WHERE project_name = ?
                    ''', (hierarchy_json, project_path, clip_vectorstore_path, feedback, project_name))
                    print(f"Updated project: {project_name}")  # Debug log
                else:
                    # Insert new project
                    cursor.execute('''
                        INSERT INTO projects (project_name, file_hierarchy, project_path, clip_vectorstore_path, 
                                              feedback)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (project_name, hierarchy_json, project_path, clip_vectorstore_path, feedback))
                    print(f"Inserted new project: {project_name}")  # Debug log
                
                conn.commit()
                return True
                
        except Exception as e:
            print(f"Error saving project: {str(e)}")  # Debug log
            raise

    def get_project(self, project_name: str) -> Dict:
        """Retrieve a project by name."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT * FROM projects WHERE project_name = ?
                ''', (project_name,))
                
                row = cursor.fetchone()
                if row:
                    project = {
                        'project_name': row['project_name'],
                        'file_hierarchy': json.loads(row['file_hierarchy']) if row['file_hierarchy'] else {},
                        'project_path': row['project_path'],
                        'clip_vectorstore_path': row['clip_vectorstore_path'],
                        'feedback': row['feedback'],
                        'timestamp': row['timestamp']
                    }
                    print(f"Retrieved project: {project_name}")  # Debug log
                    return project
                    
                return None
                
        except Exception as e:
            print(f"Error retrieving project: {str(e)}")  # Debug log
            return None

    def get_all_projects(self) -> List[Dict]:
        """Retrieve all projects from the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT * FROM projects
                    ORDER BY timestamp DESC
                ''')
                
                projects = []
                for row in cursor.fetchall():
                    project = {
                        'project_name': row['project_name'],
                        'file_hierarchy': json.loads(row['file_hierarchy']) if row['file_hierarchy'] else {},
                        'project_path': row['project_path'],
                        'clip_vectorstore_path': row['clip_vectorstore_path'],
                        'feedback': row['feedback'],
                        'timestamp': row['timestamp']
                    }
                    projects.append(project)
                
                print(f"Retrieved {len(projects)} projects")  # Debug log
                return projects
                
        except Exception as e:
            print(f"Error retrieving projects: {str(e)}")  # Debug log
            return []

    def clear_all_data(self):
        """Clear all data from the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM projects')
                cursor.execute('DELETE FROM knowledge')
                conn.commit()
                print("Cleared all data from database")  # Debug log
                
        except Exception as e:
            print(f"Error clearing database: {str(e)}")  # Debug log
            raise

    def clear_project_data(self, project_name: str):
        """Delete all database entries for a given project."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM projects WHERE project_name = ?', (project_name,))
                conn.commit()
                print(f"Cleared all data for project: {project_name}")
        except Exception as e:
            print(f"Error clearing project data: {str(e)}")
            raise

    def set_clip_vectorstore_path(self, project_name: str, clip_vectorstore_path: str):
        """Set the CLIP vectorstore path for a project."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE projects SET clip_vectorstore_path = ? WHERE project_name = ?
                ''', (clip_vectorstore_path, project_name))
                conn.commit()
                print(f"Set CLIP vectorstore path for {project_name}: {clip_vectorstore_path}")
        except Exception as e:
            print(f"Error setting CLIP vectorstore path: {str(e)}")
            raise

    def get_clip_vectorstore_path(self, project_name: str) -> str:
        """Get the CLIP vectorstore path for a project."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT clip_vectorstore_path FROM projects WHERE project_name = ?
                ''', (project_name,))
                row = cursor.fetchone()
                return row[0] if row and row[0] else None
        except Exception as e:
            print(f"Error getting CLIP vectorstore path: {str(e)}")
            return None

    def set_knowledge_clip_vectorstore_path(self, path: str):
        """Set the knowledge CLIP vectorstore path in the knowledge table."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO knowledge (id, clip_vectorstore_path) VALUES (1, ?)
                ''', (path,))
                conn.commit()
                print(f"Set knowledge CLIP vectorstore path: {path}")
        except Exception as e:
            print(f"Error setting knowledge CLIP vectorstore path: {str(e)}")
            raise

    def get_knowledge_clip_vectorstore_path(self) -> str:
        """Get the knowledge CLIP vectorstore path from the knowledge table."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT clip_vectorstore_path FROM knowledge WHERE id = 1
                ''')
                row = cursor.fetchone()
                return row[0] if row and row[0] else None
        except Exception as e:
            print(f"Error getting knowledge CLIP vectorstore path: {str(e)}")
            return None

    def update_feedback(self, project_name: str, feedback: str):
        """Update only the feedback and timestamp for a project."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE projects SET feedback = ?, timestamp = CURRENT_TIMESTAMP WHERE project_name = ?
                ''', (feedback, project_name))
                conn.commit()
                print(f"Updated feedback for project: {project_name}")
        except Exception as e:
            print(f"Error updating feedback: {str(e)}")
            raise