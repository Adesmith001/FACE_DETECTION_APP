"""
database_setup.py
-----------------
Creates a local SQLite database for storing users, images,
and detected emotions from the Emotion Detection App.
"""

import sqlite3

# Create or connect to database
conn = sqlite3.connect('emotion_users.db')
cursor = conn.cursor()

# Create table
cursor.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    usage_mode TEXT CHECK(usage_mode IN ('online', 'offline')),
    image_path TEXT,
    detected_emotion TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
''')

conn.commit()
conn.close()

print("[INFO] Database 'emotion_users.db' created successfully!")
