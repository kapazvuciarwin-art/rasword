from flask import Flask, render_template, request, jsonify
import sqlite3
import random
import os
import json
import time
import threading
import requests
from datetime import datetime, timedelta
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

# OpenRouter API 設定
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# 免費模型優先順序（會自動嘗試直到找到可用的）
OPENROUTER_FREE_MODELS = [
    "deepseek/deepseek-r1:free",
    "deepseek/deepseek-chat:free",
    "qwen/qwen3-14b:free",
    "qwen/qwen3-32b:free",
    "google/gemma-3-27b-it:free",
    "meta-llama/llama-4-scout:free",
    "meta-llama/llama-3.2-3b-instruct:free",
    "mistralai/mistral-7b-instruct:free",
    "microsoft/phi-3-mini-128k-instruct:free",
]

# Groq API（OpenAI 相容介面）
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
# 預設 Groq 模型，可在前端自訂
DEFAULT_GROQ_MODEL = os.getenv("GROQ_DEFAULT_MODEL", "llama-3.3-70b-versatile")

app = Flask(__name__)

ALLOWED_QUIZ_SOURCES = {"manual", "lyrics", "transcript", "batch", "batch_ai"}

# API 速率限制器
class RateLimiter:
    def __init__(self, max_calls=5, period=59):
        self.max_calls = max_calls
        self.period = period  # 秒
        self.calls = defaultdict(list)  # api_key -> [timestamps]
        self.lock = threading.Lock()
    
    def _clean_old_calls(self, api_key):
        """清除過期的呼叫記錄"""
        now = time.time()
        self.calls[api_key] = [t for t in self.calls[api_key] if now - t < self.period]
    
    def get_wait_time(self, api_key):
        """取得需要等待的時間（秒），0 表示可以立即呼叫"""
        with self.lock:
            self._clean_old_calls(api_key)
            
            if len(self.calls[api_key]) < self.max_calls:
                return 0
            
            # 計算需要等待多久
            oldest_call = min(self.calls[api_key])
            wait_time = self.period - (time.time() - oldest_call)
            return max(0, wait_time)
    
    def can_call(self, api_key):
        """檢查是否可以呼叫"""
        return self.get_wait_time(api_key) == 0
    
    def record_call(self, api_key):
        """記錄一次呼叫"""
        with self.lock:
            self.calls[api_key].append(time.time())
    
    def wait_and_call(self, api_key):
        """等待直到可以呼叫，然後記錄"""
        wait_time = self.get_wait_time(api_key)
        if wait_time > 0:
            time.sleep(wait_time)
        self.record_call(api_key)
        return wait_time
    
    def get_status(self, api_key):
        """取得目前狀態"""
        with self.lock:
            self._clean_old_calls(api_key)
            used = len(self.calls[api_key])
            remaining = self.max_calls - used
            wait_time = self.get_wait_time(api_key)
            return {
                'used': used,
                'remaining': remaining,
                'max': self.max_calls,
                'period': self.period,
                'wait_time': round(wait_time, 1)
            }

# 全域速率限制器
rate_limiter = RateLimiter(max_calls=5, period=59)

# Gemini API 設定
DEFAULT_GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# 模型優先順序（從最新到穩定）
MODEL_PRIORITY = [
    'gemini-3-pro-preview',
    'gemini-3-flash-preview', 
    'gemini-2.5-flash',
    'gemini-2.0-flash',
]

def get_genai_module():
    import google.generativeai as genai_module
    return genai_module

def get_working_model(api_key):
    """嘗試找到可用的模型"""
    genai_module = get_genai_module()
    genai_module.configure(api_key=api_key)
    
    for model_name in MODEL_PRIORITY:
        try:
            model = genai_module.GenerativeModel(model_name)
            # 簡單測試模型是否可用
            response = model.generate_content("test")
            return model_name
        except Exception as e:
            print(f"模型 {model_name} 不可用: {e}")
            continue
    
    return MODEL_PRIORITY[-1]  # 預設使用最後一個
DATABASE = 'vocabulary.db'

def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


def _normalize_source_filter(source_value):
    source = (source_value or "").strip().lower()
    if not source or source == "all":
        return None
    if source in ALLOWED_QUIZ_SOURCES:
        return source
    return "__invalid__"

def init_db():
    conn = get_db()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS words (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            japanese_word TEXT NOT NULL,
            part_of_speech TEXT,
            sentence1 TEXT,
            sentence2 TEXT,
            sentence3 TEXT,
            chinese_meaning TEXT NOT NULL,
            chinese_short TEXT,
            jlpt_level TEXT,
            kana_form TEXT,
            kanji_form TEXT,
            common_form TEXT DEFAULT 'kanji',
            source_title TEXT,
            source_lyric_id INTEGER,
            quiz_correct INTEGER DEFAULT 0,
            quiz_wrong INTEGER DEFAULT 0,
            quiz_next_review TEXT,
            typing_correct INTEGER DEFAULT 0,
            typing_wrong INTEGER DEFAULT 0,
            typing_next_review TEXT
        )
    ''')
    # 資料庫升級 - 新增欄位
    cursor = conn.execute('PRAGMA table_info(words)')
    columns = [row[1] for row in cursor.fetchall()]
    
    if 'chinese_short' not in columns:
        conn.execute('ALTER TABLE words ADD COLUMN chinese_short TEXT')
    if 'kana_form' not in columns:
        conn.execute('ALTER TABLE words ADD COLUMN kana_form TEXT')
    if 'kanji_form' not in columns:
        conn.execute('ALTER TABLE words ADD COLUMN kanji_form TEXT')
    if 'common_form' not in columns:
        conn.execute('ALTER TABLE words ADD COLUMN common_form TEXT DEFAULT "kanji"')
    # SRS 欄位
    if 'quiz_correct' not in columns:
        conn.execute('ALTER TABLE words ADD COLUMN quiz_correct INTEGER DEFAULT 0')
    if 'quiz_wrong' not in columns:
        conn.execute('ALTER TABLE words ADD COLUMN quiz_wrong INTEGER DEFAULT 0')
    if 'quiz_next_review' not in columns:
        conn.execute('ALTER TABLE words ADD COLUMN quiz_next_review TEXT')
    if 'typing_correct' not in columns:
        conn.execute('ALTER TABLE words ADD COLUMN typing_correct INTEGER DEFAULT 0')
    if 'typing_wrong' not in columns:
        conn.execute('ALTER TABLE words ADD COLUMN typing_wrong INTEGER DEFAULT 0')
    if 'typing_next_review' not in columns:
        conn.execute('ALTER TABLE words ADD COLUMN typing_next_review TEXT')
    if 'source' not in columns:
        conn.execute('ALTER TABLE words ADD COLUMN source TEXT DEFAULT "manual"')
    # 提示次數欄位
    if 'quiz_hint_count' not in columns:
        conn.execute('ALTER TABLE words ADD COLUMN quiz_hint_count INTEGER DEFAULT 0')
    if 'quiz_hint_sessions' not in columns:
        conn.execute('ALTER TABLE words ADD COLUMN quiz_hint_sessions INTEGER DEFAULT 0')
    if 'typing_hint_count' not in columns:
        conn.execute('ALTER TABLE words ADD COLUMN typing_hint_count INTEGER DEFAULT 0')
    if 'typing_hint_sessions' not in columns:
        conn.execute('ALTER TABLE words ADD COLUMN typing_hint_sessions INTEGER DEFAULT 0')
    if 'sentence3' not in columns:
        conn.execute('ALTER TABLE words ADD COLUMN sentence3 TEXT')
    if 'source_title' not in columns:
        conn.execute('ALTER TABLE words ADD COLUMN source_title TEXT')
    if 'source_lyric_id' not in columns:
        conn.execute('ALTER TABLE words ADD COLUMN source_lyric_id INTEGER')
    
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
    try:
        data = request.json
        if not data:
            return jsonify({'success': False, 'error': '無效的請求資料'}), 400
        
        japanese_word = data.get('japanese_word', '').strip()
        if not japanese_word:
            return jsonify({'success': False, 'error': '日文詞不能為空'}), 400
        
        conn = get_db()
        source = data.get('source', 'manual')  # manual, batch, transcript
        conn.execute('''
            INSERT INTO words (japanese_word, part_of_speech, sentence1, sentence2, sentence3, chinese_meaning, chinese_short, jlpt_level, kana_form, kanji_form, common_form, source, source_title, source_lyric_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (japanese_word, data.get('part_of_speech', ''), data.get('sentence1', ''), 
              data.get('sentence2', ''), data.get('sentence3', ''), data.get('chinese_meaning', ''), data.get('chinese_short', ''), 
              data.get('jlpt_level', ''), data.get('kana_form', ''), data.get('kanji_form', ''),
              data.get('common_form', 'kanji'), source, data.get('source_title', ''), data.get('source_lyric_id')))
        conn.commit()
        conn.close()
        return jsonify({'success': True})
    except Exception as e:
        print(f"新增單字錯誤: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/words/batch', methods=['POST'])
def add_words_batch():
    data = request.json
    words = data.get('words', [])
    conn = get_db()
    for word in words:
        source = word.get('source', 'batch')
        conn.execute('''
            INSERT INTO words (japanese_word, part_of_speech, sentence1, sentence2, chinese_meaning, chinese_short, jlpt_level, kana_form, kanji_form, common_form, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (word['japanese_word'], word['part_of_speech'], word['sentence1'],
              word['sentence2'], word['chinese_meaning'], word.get('chinese_short', ''), 
              word['jlpt_level'], word.get('kana_form', ''), word.get('kanji_form', ''),
              word.get('common_form', 'kanji'), source))
    conn.commit()
    conn.close()
    return jsonify({'success': True, 'count': len(words)})

@app.route('/api/words/check', methods=['POST'])
def check_word_exists():
    """檢查單字是否已存在"""
    try:
        data = request.json
        if not data:
            return jsonify({'exists': False, 'error': '無效的請求'})
        
        japanese_word = data.get('japanese_word', '').strip()
        
        if not japanese_word:
            return jsonify({'exists': False})
        
        conn = get_db()
        # 檢查 japanese_word、kana_form、kanji_form 是否有符合的
        existing = conn.execute('''
            SELECT id, japanese_word, kana_form, kanji_form, chinese_short 
            FROM words 
            WHERE japanese_word = ? OR kana_form = ? OR kanji_form = ?
        ''', (japanese_word, japanese_word, japanese_word)).fetchone()
        conn.close()
        
        if existing:
            return jsonify({
                'exists': True,
                'word': {
                    'id': existing['id'],
                    'japanese_word': existing['japanese_word'],
                    'kana_form': existing['kana_form'],
                    'kanji_form': existing['kanji_form'],
                    'chinese_short': existing['chinese_short']
                }
            })
        
        return jsonify({'exists': False})
    except Exception as e:
        print(f"檢查單字錯誤: {e}")
        return jsonify({'exists': False, 'error': str(e)}), 500

@app.route('/api/words/check-batch', methods=['POST'])
def check_words_batch():
    """批次檢查單字是否已存在"""
    data = request.json
    words = data.get('words', [])
    
    if not words:
        return jsonify({'results': []})
    
    conn = get_db()
    results = []
    
    for word in words:
        word = word.strip()
        if not word:
            continue
        
        existing = conn.execute('''
            SELECT id, japanese_word, kana_form, kanji_form, chinese_short 
            FROM words 
            WHERE japanese_word = ? OR kana_form = ? OR kanji_form = ?
        ''', (word, word, word)).fetchone()
        
        if existing:
            results.append({
                'word': word,
                'exists': True,
                'existing': {
                    'japanese_word': existing['japanese_word'],
                    'chinese_short': existing['chinese_short']
                }
            })
        else:
            results.append({
                'word': word,
                'exists': False
            })
    
    conn.close()
    return jsonify({'results': results})

@app.route('/api/words/<int:word_id>', methods=['DELETE'])
def delete_word(word_id):
    conn = get_db()
    conn.execute('DELETE FROM words WHERE id = ?', (word_id,))
    conn.commit()
    conn.close()
    return jsonify({'success': True})

def calculate_srs_weight(word, quiz_type):
    """計算 SRS 權重，權重越高越容易被選中"""
    if quiz_type == 'quiz':
        correct = word['quiz_correct'] or 0
        wrong = word['quiz_wrong'] or 0
        # sqlite3.Row 沒有 .get() 方法，使用 try-except
        try:
            hint_count = word['quiz_hint_count'] or 0
        except (KeyError, IndexError):
            hint_count = 0
        try:
            hint_sessions = word['quiz_hint_sessions'] or 0
        except (KeyError, IndexError):
            hint_sessions = 0
        next_review = word['quiz_next_review']
    else:  # typing
        correct = word['typing_correct'] or 0
        wrong = word['typing_wrong'] or 0
        # sqlite3.Row 沒有 .get() 方法，使用 try-except
        try:
            hint_count = word['typing_hint_count'] or 0
        except (KeyError, IndexError):
            hint_count = 0
        try:
            hint_sessions = word['typing_hint_sessions'] or 0
        except (KeyError, IndexError):
            hint_sessions = 0
        next_review = word['typing_next_review']
    
    # 基礎權重
    weight = 10.0
    
    # 錯誤越多，權重越高
    weight += wrong * 5
    
    # 正確越多，權重越低（但不低於 1）
    weight -= correct * 2
    
    # 【新增】提示次數權重
    # 使用過提示的次數越多，表示越不熟練，權重越高
    weight += hint_sessions * 3  # 每次使用過提示的答題 +3 權重
    
    # 累計提示次數也有影響（但權重較低）
    weight += hint_count * 0.5   # 每使用一次提示 +0.5 權重
    
    # 【新增】提示依賴度懲罰
    # 如果經常使用提示，即使答對也要多複習
    if hint_sessions > 0:
        hint_dependency_ratio = hint_count / max(hint_sessions, 1)  # 平均每次答題用幾個提示
        if hint_dependency_ratio > 2:  # 平均每次用超過2個提示
            weight += 5  # 額外懲罰
    
    # 從未答過的題目給予較高權重
    if correct == 0 and wrong == 0:
        weight += 15
    
    # 檢查是否到了複習時間
    if next_review:
        try:
            review_time = datetime.fromisoformat(next_review)
            now = datetime.now()
            if now >= review_time:
                # 已經到複習時間，增加權重
                hours_overdue = (now - review_time).total_seconds() / 3600
                weight += min(hours_overdue, 20)  # 最多加 20
            else:
                # 還沒到複習時間，大幅降低權重
                weight = max(1, weight - 20)
        except:
            pass
    
    return max(1, weight)

def calculate_next_review(correct_count, wrong_count, is_correct, hint_count=0):
    """計算下次複習時間"""
    # 基礎間隔（分鐘）
    if is_correct:
        # 正確：根據連續正確次數增加間隔
        intervals = [10, 30, 60, 240, 480, 1440, 2880, 5760, 10080]  # 10分, 30分, 1時, 4時, 8時, 1天, 2天, 4天, 7天
        index = min(correct_count, len(intervals) - 1)
        minutes = intervals[index]
        
        # 【新增】如果答對但用了提示，縮短間隔
        if hint_count > 0:
            # 用提示答對，間隔減半（但至少10分鐘）
            minutes = max(10, minutes // 2)
    else:
        # 錯誤：很快再次出現
        minutes = 5
    
    return datetime.now() + timedelta(minutes=minutes)

@app.route('/api/record-answer', methods=['POST'])
def record_answer():
    """記錄答題結果"""
    data = request.json
    word_id = data.get('word_id')
    quiz_type = data.get('quiz_type')  # 'quiz' 或 'typing'
    is_correct = data.get('is_correct', False)
    hint_count = data.get('hint_count', 0)  # 本次使用的提示次數
    
    if not word_id or not quiz_type:
        return jsonify({'error': '缺少必要參數'}), 400
    
    conn = get_db()
    
    # 獲取當前統計
    word = conn.execute('SELECT * FROM words WHERE id = ?', (word_id,)).fetchone()
    if not word:
        conn.close()
        return jsonify({'error': '找不到單字'}), 404
    
    if quiz_type == 'quiz':
        correct = (word['quiz_correct'] or 0) + (1 if is_correct else 0)
        wrong = (word['quiz_wrong'] or 0) + (0 if is_correct else 1)
        
        # 更新提示統計
        try:
            current_hint_count = (word['quiz_hint_count'] or 0) + hint_count
        except (KeyError, IndexError):
            current_hint_count = hint_count
        try:
            current_hint_sessions = (word['quiz_hint_sessions'] or 0) + (1 if hint_count > 0 else 0)
        except (KeyError, IndexError):
            current_hint_sessions = (1 if hint_count > 0 else 0)
        
        next_review = calculate_next_review(correct, wrong, is_correct, hint_count).isoformat()
        
        conn.execute('''
            UPDATE words SET 
                quiz_correct = ?, 
                quiz_wrong = ?, 
                quiz_next_review = ?,
                quiz_hint_count = ?,
                quiz_hint_sessions = ?
            WHERE id = ?
        ''', (correct, wrong, next_review, current_hint_count, current_hint_sessions, word_id))
    else:  # typing
        correct = (word['typing_correct'] or 0) + (1 if is_correct else 0)
        wrong = (word['typing_wrong'] or 0) + (0 if is_correct else 1)
        
        # 更新提示統計
        try:
            current_hint_count = (word['typing_hint_count'] or 0) + hint_count
        except (KeyError, IndexError):
            current_hint_count = hint_count
        try:
            current_hint_sessions = (word['typing_hint_sessions'] or 0) + (1 if hint_count > 0 else 0)
        except (KeyError, IndexError):
            current_hint_sessions = (1 if hint_count > 0 else 0)
        
        next_review = calculate_next_review(correct, wrong, is_correct, hint_count).isoformat()
        
        conn.execute('''
            UPDATE words SET 
                typing_correct = ?, 
                typing_wrong = ?, 
                typing_next_review = ?,
                typing_hint_count = ?,
                typing_hint_sessions = ?
            WHERE id = ?
        ''', (correct, wrong, next_review, current_hint_count, current_hint_sessions, word_id))
    
    conn.commit()
    conn.close()
    
    return jsonify({'success': True, 'next_review': next_review})


@app.route('/api/quiz/kana-hint', methods=['POST'])
def record_quiz_kana_hint():
    """在選擇題按下「顯示假名」時，提升該單字在打字題的出題權重"""
    data = request.json or {}
    word_id = data.get('word_id')
    if not word_id:
        return jsonify({'error': '缺少 word_id'}), 400

    conn = get_db()
    word = conn.execute(
        'SELECT id, typing_hint_count, typing_hint_sessions FROM words WHERE id = ?',
        (word_id,),
    ).fetchone()
    if not word:
        conn.close()
        return jsonify({'error': '找不到單字'}), 404

    try:
        hint_count = (word['typing_hint_count'] or 0) + 1
    except (KeyError, IndexError):
        hint_count = 1
    try:
        hint_sessions = (word['typing_hint_sessions'] or 0) + 1
    except (KeyError, IndexError):
        hint_sessions = 1

    # 拉近打字複習時間，避免被未到期機制壓低權重
    next_review = datetime.now().isoformat()
    conn.execute(
        '''
        UPDATE words
        SET typing_hint_count = ?, typing_hint_sessions = ?, typing_next_review = ?
        WHERE id = ?
        ''',
        (hint_count, hint_sessions, next_review, word_id),
    )
    conn.commit()
    conn.close()
    return jsonify({'success': True})

@app.route('/api/quiz', methods=['GET'])
def get_quiz():
    source_filter = _normalize_source_filter(request.args.get('source'))
    if source_filter == "__invalid__":
        return jsonify({'error': '無效的來源篩選'}), 400

    conn = get_db()
    if source_filter:
        words = conn.execute('SELECT * FROM words WHERE source = ?', (source_filter,)).fetchall()
    else:
        words = conn.execute('SELECT * FROM words').fetchall()
    conn.close()
    
    if len(words) < 2:
        source_text = f'（來源：{source_filter}）' if source_filter else ''
        return jsonify({'error': f'可用單字不足，至少需要 2 個單字才能開始選擇題{source_text}'}), 400
    
    # 計算每個單字的 SRS 權重
    weighted_words = [(w, calculate_srs_weight(w, 'quiz')) for w in words]
    total_weight = sum(weight for _, weight in weighted_words)
    
    # 根據權重隨機選擇
    r = random.uniform(0, total_weight)
    cumulative = 0
    question_word = words[0]
    for word, weight in weighted_words:
        cumulative += weight
        if r <= cumulative:
            question_word = word
            break
    
    # 選擇9個其他單字作為干擾選項
    other_words = [w for w in words if w['id'] != question_word['id']]
    distractors = random.sample(other_words, min(9, len(other_words)))
    
    # 組合選項（正確答案 + 干擾選項）- 使用簡短解釋作為選項
    def get_option_text(word):
        return word['chinese_short'] if word['chinese_short'] else word['chinese_meaning']
    
    options = [{'id': question_word['id'], 'chinese_meaning': get_option_text(question_word)}]
    for d in distractors:
        options.append({'id': d['id'], 'chinese_meaning': get_option_text(d)})
    
    random.shuffle(options)
    
    return jsonify({
        'question': {
            'id': question_word['id'],
            'japanese_word': question_word['japanese_word'],
            'part_of_speech': question_word['part_of_speech'],
            'sentence1': question_word['sentence1'],
            'sentence2': question_word['sentence2'],
            'sentence3': question_word['sentence3'],
            'jlpt_level': question_word['jlpt_level'],
            'correct_meaning': question_word['chinese_meaning'],
            'kana_form': question_word['kana_form'],
            'kanji_form': question_word['kanji_form'],
            'common_form': question_word['common_form'],
            'source': question_word['source'],
            'source_title': question_word['source_title'],
            'source_lyric_id': question_word['source_lyric_id'],
        },
        'options': options
    })

def get_best_api_key(api_keys_str):
    """從多個 API Key 中選擇最佳的（可用額度最多的）"""
    if not api_keys_str:
        return DEFAULT_GEMINI_API_KEY, None
    
    # 支援多個 key（用逗號分隔）
    keys = [k.strip() for k in api_keys_str.split(',') if k.strip()]
    
    if not keys:
        return DEFAULT_GEMINI_API_KEY, None
    
    # 找出可用額度最多的 key
    best_key = None
    best_remaining = -1
    
    for key in keys:
        status = rate_limiter.get_status(key)
        if status['remaining'] > best_remaining:
            best_remaining = status['remaining']
            best_key = key
    
    # 如果所有 key 都用完了，返回等待時間最短的
    if best_remaining == 0:
        min_wait = float('inf')
        for key in keys:
            status = rate_limiter.get_status(key)
            if status['wait_time'] < min_wait:
                min_wait = status['wait_time']
                best_key = key
    
    return best_key, keys

def call_openrouter_api(api_key, prompt, model=None):
    """呼叫 OpenRouter API，自動嘗試多個免費模型"""
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:5000",
        "X-Title": "Rasword"
    }
    
    # 如果指定了模型就只用那個，否則自動嘗試免費模型
    models_to_try = [model] if model else OPENROUTER_FREE_MODELS
    
    last_error = None
    
    for try_model in models_to_try:
        try:
            print(f"[OpenRouter] 嘗試模型: {try_model}")
            
            data = {
                "model": try_model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.7
            }
            
            response = requests.post(OPENROUTER_API_URL, headers=headers, json=data, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                if 'error' not in result:
                    content = result['choices'][0]['message']['content']
                    print(f"[OpenRouter] 成功使用模型: {try_model}")
                    return content, try_model
            
            # 記錄錯誤但繼續嘗試下一個
            error_detail = response.text
            print(f"[OpenRouter] 模型 {try_model} 失敗: {error_detail}")
            last_error = f"{try_model}: {error_detail}"
            
        except Exception as e:
            print(f"[OpenRouter] 模型 {try_model} 例外: {e}")
            last_error = f"{try_model}: {str(e)}"
            continue
    
    raise Exception(f"所有模型都失敗了。最後錯誤: {last_error}")


def call_groq_api(api_key, prompt, model_name=None):
    """
    呼叫 Groq Chat Completions API（OpenAI 相容）。
    目前只用一個模型（可由前端或環境變數決定）。
    """
    if not api_key:
        raise ValueError("缺少 Groq API Key")
    model = (model_name or DEFAULT_GROQ_MODEL or "").strip()
    if not model:
        raise ValueError("Groq 模型名稱未設定")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
    }
    r = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=60)
    if r.status_code != 200:
        try:
            err_json = r.json()
            msg = err_json.get("error", {}).get("message") or r.text
        except Exception:
            msg = r.text
        raise RuntimeError(f"Groq API 錯誤（HTTP {r.status_code}）: {msg}")
    data = r.json()
    try:
        text = data["choices"][0]["message"]["content"]
    except Exception as e:
        raise RuntimeError(f"Groq 回應格式錯誤: {e}")
    return (text or "").strip(), model

def generate_word_prompt(japanese_word):
    """生成單字資訊的 prompt"""
    return f"""請分析這個日文詞彙「{japanese_word}」，並以 JSON 格式回傳以下資訊：
1. part_of_speech: 詞性（如：名詞、動詞、形容詞、副詞等）
2. sentence1: 一個常用的日文例句（使用這個詞）
3. sentence2: 另一個常用的日文例句（使用這個詞）
4. chinese_short: 繁體中文簡短解釋（1-3個字，只寫最核心的意思，例如：吃、喝、貓、漂亮）
5. chinese_meaning: 繁體中文完整解釋
6. jlpt_level: 用日文解釋這個詞的意思（像日日辭典一樣，純日文定義）
7. kana_form: 這個詞的純假名寫法（平假名或片假名）
8. kanji_form: 這個詞的漢字寫法（如果有的話，沒有就留空）
9. common_form: 最常用的寫法是哪種？回答 "kanji"（漢字常用）、"hiragana"（平假名常用）、"katakana"（片假名常用，常見於副詞、擬聲詞、外來語等）或 "both"（兩者都常用）

只回傳純 JSON，不要有其他文字或 markdown 格式。範例格式：
{{"part_of_speech": "動詞", "sentence1": "ご飯を食べる", "sentence2": "朝ご飯を食べました", "chinese_short": "吃", "chinese_meaning": "吃、進食", "jlpt_level": "食べ物を口に入れて、かんで、飲み込むこと。", "kana_form": "たべる", "kanji_form": "食べる", "common_form": "kanji"}}"""

@app.route('/api/rate-limit-status', methods=['POST'])
def get_rate_limit_status():
    """取得 API 速率限制狀態"""
    data = request.json
    custom_api_keys = data.get('api_key', '').strip()
    
    # 支援多個 key
    if custom_api_keys:
        keys = [k.strip() for k in custom_api_keys.split(',') if k.strip()]
    else:
        keys = [DEFAULT_GEMINI_API_KEY] if DEFAULT_GEMINI_API_KEY else []
    
    if not keys:
        return jsonify({'error': '未設定 API Key'}), 400
    
    # 返回所有 key 的狀態
    all_status = []
    total_remaining = 0
    
    for key in keys:
        status = rate_limiter.get_status(key)
        key_id = hash(key) % 1000000
        status['key_id'] = key_id
        status['key_preview'] = key[:10] + '...' + key[-4:] if len(key) > 14 else key
        all_status.append(status)
        total_remaining += status['remaining']
    
    return jsonify({
        'keys': all_status,
        'total_remaining': total_remaining,
        'total_max': len(keys) * 5,
        'key_count': len(keys)
    })

@app.route('/api/generate', methods=['POST'])
def generate_word_info():
    data = request.json
    japanese_word = data.get('japanese_word', '').strip()
    custom_api_keys = data.get('api_key', '').strip()
    api_provider = data.get('api_provider', 'gemini')  # 'gemini'、'openrouter'、'groq'
    openrouter_key = data.get('openrouter_key', '').strip()
    openrouter_model = data.get('openrouter_model', '').strip()
    groq_key = data.get('groq_key', '').strip()
    groq_model = data.get('groq_model', '').strip()
    
    if not japanese_word:
        return jsonify({'error': '請輸入日文詞'}), 400
    
    prompt = generate_word_prompt(japanese_word)
    
    # 使用 OpenRouter API
    if api_provider == 'openrouter':
        if not openrouter_key:
            return jsonify({'error': '請設定 OpenRouter API Key'}), 400
        
        try:
            # 自動嘗試免費模型（不傳 model 參數）
            result_text, used_model = call_openrouter_api(openrouter_key, prompt)
            
            # 清理可能的 markdown 格式
            if result_text.startswith('```'):
                lines = result_text.split('\n')
                result_text = '\n'.join(lines[1:-1])
            
            result = json.loads(result_text)
            result['japanese_word'] = japanese_word
            result['model_used'] = f"OpenRouter: {used_model}"
            result.setdefault('kana_form', '')
            result.setdefault('kanji_form', '')
            result.setdefault('common_form', 'kanji')
            
            return jsonify(result)
            
        except json.JSONDecodeError as e:
            return jsonify({'error': f'AI 回應格式錯誤: {str(e)}', 'raw': result_text}), 500
        except Exception as e:
            return jsonify({'error': f'生成失敗: {str(e)}'}), 500
    
    # 使用 Groq API
    if api_provider == 'groq':
        if not groq_key:
            return jsonify({'error': '請設定 Groq API Key'}), 400
        try:
            result_text, used_model = call_groq_api(groq_key, prompt, groq_model or None)
            # 清理可能的 markdown 格式
            if result_text.startswith('```'):
                lines = result_text.split('\n')
                result_text = '\n'.join(lines[1:-1])
            result = json.loads(result_text)
            result['japanese_word'] = japanese_word
            result['model_used'] = f"Groq: {used_model}"
            result.setdefault('kana_form', '')
            result.setdefault('kanji_form', '')
            result.setdefault('common_form', 'kanji')
            return jsonify(result)
        except json.JSONDecodeError as e:
            return jsonify({'error': f'Groq 回應 JSON 解析錯誤: {str(e)}', 'raw': result_text}), 500
        except Exception as e:
            return jsonify({'error': f'Groq 生成失敗: {str(e)}'}), 500
    
    # 使用 Gemini API（原有邏輯）
    api_key, all_keys = get_best_api_key(custom_api_keys)
    
    if not api_key:
        return jsonify({'error': '請設定 API Key（在設定中填入或聯繫管理員）'}), 400
    
    # 檢查速率限制
    wait_time = rate_limiter.get_wait_time(api_key)
    if wait_time > 0:
        # 如果有多個 key，嘗試找其他可用的
        if all_keys and len(all_keys) > 1:
            for key in all_keys:
                if rate_limiter.get_wait_time(key) == 0:
                    api_key = key
                    wait_time = 0
                    break
        
        if wait_time > 0:
            return jsonify({
                'error': f'所有 API Key 都在冷卻中，請等待 {round(wait_time, 1)} 秒',
                'wait_time': wait_time,
                'rate_limited': True
            }), 429
    
    # 記錄此次呼叫
    rate_limiter.record_call(api_key)
    
    try:
        genai_module = get_genai_module()
        genai_module.configure(api_key=api_key)
        
        # 嘗試不同模型
        last_error = None
        for model_name in MODEL_PRIORITY:
            try:
                model = genai_module.GenerativeModel(model_name)
                
                response = model.generate_content(prompt)
                result_text = response.text.strip()
                
                # 清理可能的 markdown 格式
                if result_text.startswith('```'):
                    lines = result_text.split('\n')
                    result_text = '\n'.join(lines[1:-1])
                
                result = json.loads(result_text)
                result['japanese_word'] = japanese_word
                result['model_used'] = model_name
                # 確保新欄位存在
                result.setdefault('kana_form', '')
                result.setdefault('kanji_form', '')
                result.setdefault('common_form', 'kanji')
                
                return jsonify(result)
                
            except json.JSONDecodeError as e:
                return jsonify({'error': f'AI 回應格式錯誤: {str(e)}', 'raw': result_text}), 500
            except Exception as e:
                last_error = str(e)
                print(f"模型 {model_name} 失敗: {e}")
                continue
        
        return jsonify({'error': f'所有模型都失敗了: {last_error}'}), 500
        
    except Exception as e:
        return jsonify({'error': f'生成失敗: {str(e)}'}), 500

@app.route('/api/generate-batch', methods=['POST'])
def generate_batch():
    """批次 AI 生成並儲存單字"""
    data = request.json
    words_text = data.get('words', '').strip()
    custom_api_keys = data.get('api_key', '').strip()
    api_provider = data.get('api_provider', 'gemini')  # 'gemini'、'openrouter'、'groq'
    openrouter_key = data.get('openrouter_key', '').strip()
    openrouter_model = data.get('openrouter_model', '').strip()
    groq_key = data.get('groq_key', '').strip()
    groq_model = data.get('groq_model', '').strip()
    
    if not words_text:
        return jsonify({'error': '請輸入要生成的詞彙'}), 400
    
    # 分割詞彙（每行一個）
    word_list = [w.strip() for w in words_text.split('\n') if w.strip()]
    
    if not word_list:
        return jsonify({'error': '沒有有效的詞彙'}), 400
    
    results = []
    errors = []
    total_wait_time = 0
    
    # 使用 OpenRouter API
    if api_provider == 'openrouter':
        if not openrouter_key:
            return jsonify({'error': '請設定 OpenRouter API Key'}), 400
        
        conn = get_db()
        used_model = None
        
        for japanese_word in word_list:
            try:
                prompt = generate_word_prompt(japanese_word)
                # 自動嘗試免費模型
                result_text, used_model = call_openrouter_api(openrouter_key, prompt)
                
                # 清理可能的 markdown 格式
                if result_text.startswith('```'):
                    lines = result_text.split('\n')
                    result_text = '\n'.join(lines[1:-1])
                
                result = json.loads(result_text)
                
                # 儲存到資料庫
                conn.execute('''
                    INSERT INTO words (japanese_word, part_of_speech, sentence1, sentence2, chinese_meaning, chinese_short, jlpt_level, kana_form, kanji_form, common_form, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (japanese_word, result.get('part_of_speech', ''), result.get('sentence1', ''),
                      result.get('sentence2', ''), result.get('chinese_meaning', ''),
                      result.get('chinese_short', ''), result.get('jlpt_level', ''),
                      result.get('kana_form', ''), result.get('kanji_form', ''),
                      result.get('common_form', 'kanji'), 'batch_ai'))
                
                results.append({'word': japanese_word, 'success': True})
                
            except json.JSONDecodeError as e:
                errors.append({'word': japanese_word, 'error': f'JSON 解析錯誤: {str(e)}'})
            except Exception as e:
                errors.append({'word': japanese_word, 'error': str(e)})
        
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'total': len(word_list),
            'completed': len(results),
            'failed': len(errors),
            'results': results,
            'errors': errors,
            'total_wait_time': 0,
            'model_used': used_model
        })
    
    # 使用 Groq API：逐個詞呼叫，同步儲存
    if api_provider == 'groq':
        if not groq_key:
            return jsonify({'error': '請設定 Groq API Key'}), 400
        conn = get_db()
        used_model = None
        for japanese_word in word_list:
            try:
                prompt = generate_word_prompt(japanese_word)
                result_text, used_model = call_groq_api(groq_key, prompt, groq_model or None)
                if result_text.startswith('```'):
                    lines = result_text.split('\n')
                    result_text = '\n'.join(lines[1:-1])
                result = json.loads(result_text)
                conn.execute(
                    '''
                    INSERT INTO words (japanese_word, part_of_speech, sentence1, sentence2, chinese_meaning, chinese_short, jlpt_level, kana_form, kanji_form, common_form, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''',
                    (
                        japanese_word,
                        result.get('part_of_speech', ''),
                        result.get('sentence1', ''),
                        result.get('sentence2', ''),
                        result.get('chinese_meaning', ''),
                        result.get('chinese_short', ''),
                        result.get('jlpt_level', ''),
                        result.get('kana_form', ''),
                        result.get('kanji_form', ''),
                        result.get('common_form', 'kanji'),
                        'batch_ai',
                    ),
                )
                results.append({'word': japanese_word, 'success': True})
            except json.JSONDecodeError as e:
                errors.append({'word': japanese_word, 'error': f'Groq JSON 解析錯誤: {str(e)}'})
            except Exception as e:
                errors.append({'word': japanese_word, 'error': f'Groq 生成失敗: {str(e)}'})
        conn.commit()
        conn.close()
        return jsonify({
            'success': True,
            'total': len(word_list),
            'completed': len(results),
            'failed': len(errors),
            'results': results,
            'errors': errors,
            'total_wait_time': 0,
            'model_used': used_model,
        })
    
    # 使用 Gemini API（原有邏輯）
    if custom_api_keys:
        all_keys = [k.strip() for k in custom_api_keys.split(',') if k.strip()]
    else:
        all_keys = [DEFAULT_GEMINI_API_KEY] if DEFAULT_GEMINI_API_KEY else []
    
    if not all_keys:
        return jsonify({'error': '請設定 API Key'}), 400
    
    try:
        genai_module = get_genai_module()
        
        # 使用第一個可用的 key 來測試模型
        test_key = all_keys[0]
        genai_module.configure(api_key=test_key)
        
        # 找到可用的模型
        working_model_name = None
        for model_name in MODEL_PRIORITY:
            try:
                model = genai_module.GenerativeModel(model_name)
                model.generate_content("test")
                working_model_name = model_name
                break
            except:
                continue
        
        if not working_model_name:
            return jsonify({'error': '無法連接到 AI 模型'}), 500
        
        conn = get_db()
        
        for japanese_word in word_list:
            try:
                # 選擇最佳的 API key（可用額度最多的）
                best_key = None
                best_remaining = -1
                min_wait = float('inf')
                
                for key in all_keys:
                    status = rate_limiter.get_status(key)
                    if status['remaining'] > best_remaining:
                        best_remaining = status['remaining']
                        best_key = key
                    if status['wait_time'] < min_wait:
                        min_wait = status['wait_time']
                
                # 如果所有 key 都用完了，等待最短的
                if best_remaining == 0:
                    time.sleep(min_wait)
                    total_wait_time += min_wait
                    # 重新選擇
                    for key in all_keys:
                        if rate_limiter.get_wait_time(key) == 0:
                            best_key = key
                            break
                
                # 記錄呼叫
                rate_limiter.record_call(best_key)
                
                # 使用選中的 key
                genai_module.configure(api_key=best_key)
                working_model = genai_module.GenerativeModel(working_model_name)
                
                prompt = generate_word_prompt(japanese_word)

                response = working_model.generate_content(prompt)
                result_text = response.text.strip()
                
                # 清理可能的 markdown 格式
                if result_text.startswith('```'):
                    lines = result_text.split('\n')
                    result_text = '\n'.join(lines[1:-1])
                
                result = json.loads(result_text)
                
                # 儲存到資料庫
                conn.execute('''
                    INSERT INTO words (japanese_word, part_of_speech, sentence1, sentence2, chinese_meaning, chinese_short, jlpt_level, kana_form, kanji_form, common_form, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (japanese_word, result.get('part_of_speech', ''), result.get('sentence1', ''),
                      result.get('sentence2', ''), result.get('chinese_meaning', ''),
                      result.get('chinese_short', ''), result.get('jlpt_level', ''),
                      result.get('kana_form', ''), result.get('kanji_form', ''),
                      result.get('common_form', 'kanji'), 'batch_ai'))
                
                results.append({'word': japanese_word, 'success': True})
                
            except json.JSONDecodeError as e:
                errors.append({'word': japanese_word, 'error': f'JSON 解析錯誤: {str(e)}'})
            except Exception as e:
                errors.append({'word': japanese_word, 'error': str(e)})
        
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'total': len(word_list),
            'completed': len(results),
            'failed': len(errors),
            'results': results,
            'errors': errors,
            'total_wait_time': round(total_wait_time, 1)
        })
        
    except Exception as e:
        return jsonify({'error': f'批次生成失敗: {str(e)}'}), 500

@app.route('/api/typing-quiz', methods=['GET'])
def get_typing_quiz():
    """打字測驗 - 顯示漢字，回答假名"""
    source_filter = _normalize_source_filter(request.args.get('source'))
    if source_filter == "__invalid__":
        return jsonify({'error': '無效的來源篩選'}), 400

    conn = get_db()
    # 只選擇有漢字寫法和假名寫法的單字
    if source_filter:
        words = conn.execute('''
            SELECT * FROM words 
            WHERE kanji_form IS NOT NULL AND kanji_form != ''
            AND kana_form IS NOT NULL AND kana_form != ''
            AND source = ?
        ''', (source_filter,)).fetchall()
    else:
        words = conn.execute('''
            SELECT * FROM words 
            WHERE kanji_form IS NOT NULL AND kanji_form != '' 
            AND kana_form IS NOT NULL AND kana_form != ''
        ''').fetchall()
    conn.close()
    
    if len(words) < 1:
        source_text = f'（來源：{source_filter}）' if source_filter else ''
        return jsonify({'error': f'沒有符合條件的單字（需要有漢字和假名寫法）{source_text}'}), 400
    
    # 計算每個單字的 SRS 權重
    weighted_words = [(w, calculate_srs_weight(w, 'typing')) for w in words]
    total_weight = sum(weight for _, weight in weighted_words)
    
    # 根據權重隨機選擇
    r = random.uniform(0, total_weight)
    cumulative = 0
    question_word = words[0]
    for word, weight in weighted_words:
        cumulative += weight
        if r <= cumulative:
            question_word = word
            break
    
    return jsonify({
        'question': {
            'id': question_word['id'],
            'kanji_form': question_word['kanji_form'],
            'kana_form': question_word['kana_form'],
            'part_of_speech': question_word['part_of_speech'],
            'sentence1': question_word['sentence1'],
            'sentence2': question_word['sentence2'],
            'sentence3': question_word['sentence3'],
            'jlpt_level': question_word['jlpt_level'],
            'chinese_short': question_word['chinese_short'],
            'chinese_meaning': question_word['chinese_meaning'],
            'source': question_word['source'],
            'source_title': question_word['source_title'],
            'source_lyric_id': question_word['source_lyric_id'],
        }
    })

@app.route('/api/typing-quiz/check', methods=['POST'])
def check_typing_answer():
    """檢查打字測驗答案"""
    data = request.json
    user_answer = data.get('answer', '').strip()
    correct_answer = data.get('correct', '').strip()
    
    # 標準化比較（去除空白）
    is_correct = user_answer == correct_answer
    
    return jsonify({
        'correct': is_correct,
        'user_answer': user_answer,
        'correct_answer': correct_answer
    })

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5000, debug=True)
