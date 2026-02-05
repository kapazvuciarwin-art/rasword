from flask import Flask, render_template, request, jsonify
import sqlite3
import random
import os

app = Flask(__name__)
DATABASE = 'vocabulary.db'

def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS words (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            japanese_word TEXT NOT NULL,
            part_of_speech TEXT,
            sentence1 TEXT,
            sentence2 TEXT,
            chinese_meaning TEXT NOT NULL,
            jlpt_level TEXT
        )
    ''')
    conn.commit()
    conn.close()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/words', methods=['GET'])
def get_words():
    conn = get_db()
    words = conn.execute('SELECT * FROM words ORDER BY id DESC').fetchall()
    conn.close()
    return jsonify([dict(w) for w in words])

@app.route('/api/words', methods=['POST'])
def add_word():
    data = request.json
    conn = get_db()
    conn.execute('''
        INSERT INTO words (japanese_word, part_of_speech, sentence1, sentence2, chinese_meaning, jlpt_level)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (data['japanese_word'], data['part_of_speech'], data['sentence1'], 
          data['sentence2'], data['chinese_meaning'], data['jlpt_level']))
    conn.commit()
    conn.close()
    return jsonify({'success': True})

@app.route('/api/words/batch', methods=['POST'])
def add_words_batch():
    data = request.json
    words = data.get('words', [])
    conn = get_db()
    for word in words:
        conn.execute('''
            INSERT INTO words (japanese_word, part_of_speech, sentence1, sentence2, chinese_meaning, jlpt_level)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (word['japanese_word'], word['part_of_speech'], word['sentence1'],
              word['sentence2'], word['chinese_meaning'], word['jlpt_level']))
    conn.commit()
    conn.close()
    return jsonify({'success': True, 'count': len(words)})

@app.route('/api/words/<int:word_id>', methods=['DELETE'])
def delete_word(word_id):
    conn = get_db()
    conn.execute('DELETE FROM words WHERE id = ?', (word_id,))
    conn.commit()
    conn.close()
    return jsonify({'success': True})

@app.route('/api/quiz', methods=['GET'])
def get_quiz():
    conn = get_db()
    words = conn.execute('SELECT * FROM words').fetchall()
    conn.close()
    
    if len(words) < 10:
        return jsonify({'error': '資料庫中至少需要10個單字才能開始練習'}), 400
    
    # 選擇一個題目
    question_word = random.choice(words)
    
    # 選擇9個其他單字作為干擾選項
    other_words = [w for w in words if w['id'] != question_word['id']]
    distractors = random.sample(other_words, min(9, len(other_words)))
    
    # 組合選項（正確答案 + 干擾選項）
    options = [{'id': question_word['id'], 'chinese_meaning': question_word['chinese_meaning']}]
    for d in distractors:
        options.append({'id': d['id'], 'chinese_meaning': d['chinese_meaning']})
    
    random.shuffle(options)
    
    return jsonify({
        'question': {
            'id': question_word['id'],
            'japanese_word': question_word['japanese_word'],
            'part_of_speech': question_word['part_of_speech'],
            'sentence1': question_word['sentence1'],
            'sentence2': question_word['sentence2'],
            'jlpt_level': question_word['jlpt_level'],
            'correct_meaning': question_word['chinese_meaning']
        },
        'options': options
    })

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5000, debug=True)
