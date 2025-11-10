"""
app.py
------
Flask web app for Real-Time Emotion Detection using DeepFace and OpenCV.
Includes SQLite database storage for users and their detected emotions.
"""

from flask import Flask, render_template, Response
import cv2
from deepface import DeepFace
import sqlite3
import os
from datetime import datetime

# ----------------------------
# FLASK APP INITIALIZATION
# ----------------------------
app = Flask(__name__)
cap = cv2.VideoCapture(0)

# ----------------------------
# DATABASE SETUP
# ----------------------------
DB_PATH = "emotion_users.db"

def init_db():
    """Create database if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            usage_mode TEXT CHECK(usage_mode IN ('online','offline')),
            image_path TEXT,
            detected_emotion TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def save_user_result(name, usage_mode, image_path, emotion):
    """Save detected emotion result to database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO users (name, usage_mode, image_path, detected_emotion)
        VALUES (?, ?, ?, ?)
    ''', (name, usage_mode, image_path, emotion))
    conn.commit()
    conn.close()

# Initialize the database
init_db()

# ----------------------------
# VIDEO STREAM FUNCTION
# ----------------------------
def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Analyze emotion using DeepFace
            try:
                result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                emotion = result[0]['dominant_emotion']
            except:
                emotion = "No face detected"

            # Display emotion text on the frame
            cv2.putText(frame, f'Mood: {emotion}', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            # Save current frame as image (optional)
            os.makedirs("captured", exist_ok=True)
            image_filename = f"captured/{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(image_filename, frame)

            # Save result to database
            save_user_result("Anonymous", "online", image_filename, emotion)

            # Encode frame for streaming
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# ----------------------------
# FLASK ROUTES
# ----------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/users')
def users():
    """Display stored user data in a simple HTML table."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    conn.close()

    html = """
    <html>
    <head>
        <title>Emotion App Users</title>
        <style>
            body { font-family: Arial; background-color: #111; color: #fff; text-align:center; }
            table { margin: auto; border-collapse: collapse; width: 80%; background: #222; }
            th, td { border: 1px solid #444; padding: 10px; }
            th { background-color: #00ff99; color: #111; }
            tr:nth-child(even) { background-color: #333; }
            a { color: #00ff99; text-decoration: none; }
        </style>
    </head>
    <body>
        <h2>📋 Users Who Used the Emotion App</h2>
        <table>
            <tr>
                <th>ID</th><th>Name</th><th>Mode</th><th>Image</th><th>Emotion</th><th>Time</th>
            </tr>
    """
    for row in rows:
        html += f"""
            <tr>
                <td>{row[0]}</td>
                <td>{row[1]}</td>
                <td>{row[2]}</td>
                <td><a href='/{row[3]}' target='_blank'>View</a></td>
                <td>{row[4]}</td>
                <td>{row[5]}</td>
            </tr>
        """
    html += "</table></body></html>"
    return html

# ----------------------------
# MAIN ENTRY POINT
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)
