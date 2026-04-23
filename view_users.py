import sqlite3

def view_users():
    conn = sqlite3.connect('bioquery.db')
    cursor = conn.cursor()
    cursor.execute('SELECT id, username, email, created_at FROM users')
    users = cursor.fetchall()
    
    print("\n--- Registered Users ---")
    for user in users:
        print(f"ID: {user[0]} | Username: {user[1]} | Email: {user[2]} | Joined: {user[3]}")
    
    conn.close()

if __name__ == "__main__":
    view_users()