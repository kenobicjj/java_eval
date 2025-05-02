from tools.db_utils import get_db_connection

print("Attempting to connect to database...")
try:
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Test query
    cur.execute("SELECT * FROM submissions LIMIT 1")
    print("Connection and query successful!")
    
    cur.close()
    conn.close()
except Exception as e:
    print(f"Error: {e}") 