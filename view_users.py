import sqlite3

import os
DATA_DIR = os.environ.get("DATA_DIR", ".")
db_path = os.path.join(DATA_DIR, 'bioquery.db')
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
users = conn.execute('SELECT * FROM users').fetchall()

for u in users:
    print(f"ID: {u['id']} | Username: {u['username']} | Email: {u['email']} | Joined: {u['created_at']}")

conn.close()