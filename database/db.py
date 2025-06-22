import sqlite3
import os
from hashlib import sha256
import pandas as pd

DB_PATH = "mood_journal.db"

def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def create_tables():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE,
        password TEXT)''')
    cur.execute('''CREATE TABLE IF NOT EXISTS user_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        prompt TEXT,
        emotion TEXT,
        suggestion TEXT,
        toxicity_score REAL,
        timestamp TEXT,
        FOREIGN KEY(user_id) REFERENCES users(id))''')
    conn.commit()
    conn.close()

def register_user(email, password):
    try:
        conn = get_conn()
        conn.execute("INSERT INTO users (email, password) VALUES (?, ?)",
                     (email, sha256(password.encode()).hexdigest()))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False

def login_user(email, password):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE email = ? AND password = ?",
                (email, sha256(password.encode()).hexdigest()))
    user = cur.fetchone()
    conn.close()
    return user[0] if user else None

def log_user_prompt(user_id, prompt, emotion, suggestion, toxicity_score, timestamp):
    conn = get_conn()
    conn.execute("INSERT INTO user_logs (user_id, prompt, emotion, suggestion, toxicity_score, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                 (user_id, prompt, emotion, suggestion, toxicity_score, timestamp))
    conn.commit()
    conn.close()

def get_user_logs(user_id):
    conn = get_conn()
    df = pd.read_sql_query("SELECT * FROM user_logs WHERE user_id = ?", conn, params=(user_id,))
    conn.close()
    return df
