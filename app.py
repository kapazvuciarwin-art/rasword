from flask import Flask, render_template, request, jsonify
import sqlite3
import random
import os
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

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
            chinese_short TEXT,
            jlpt_level TEXT,
            kana_form TEXT,
            kanji_form TEXT,
            common_form TEXT DEFAULT 'kanji',
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
        INSERT INTO words (japanese_word, part_of_speech, sentence1, sentence2, chinese_meaning, chinese_short, jlpt_level, kana_form, kanji_form, common_form)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (data['japanese_word'], data['part_of_speech'], data['sentence1'], 
          data['sentence2'], data['chinese_meaning'], data.get('chinese_short', ''), 
          data['jlpt_level'], data.get('kana_form', ''), data.get('kanji_form', ''),
          data.get('common_form', 'kanji')))
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
            INSERT INTO words (japanese_word, part_of_speech, sentence1, sentence2, chinese_meaning, chinese_short, jlpt_level, kana_form, kanji_form, common_form)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (word['japanese_word'], word['part_of_speech'], word['sentence1'],
              word['sentence2'], word['chinese_meaning'], word.get('chinese_short', ''), 
              word['jlpt_level'], word.get('kana_form', ''), word.get('kanji_form', ''),
              word.get('common_form', 'kanji')))
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

def calculate_srs_weight(word, quiz_type):
    """計算 SRS 權重，權重越高越容易被選中"""
    if quiz_type == 'quiz':
        correct = word['quiz_correct'] or 0
        wrong = word['quiz_wrong'] or 0
        next_review = word['quiz_next_review']
    else:  # typing
        correct = word['typing_correct'] or 0
        wrong = word['typing_wrong'] or 0
        next_review = word['typing_next_review']
    
    # 基礎權重
    weight = 10.0
    
    # 錯誤越多，權重越高
    weight += wrong * 5
    
    # 正確越多，權重越低（但不低於 1）
    weight -= correct * 2
    
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

def calculate_next_review(correct_count, wrong_count, is_correct):
    """計算下次複習時間"""
    # 基礎間隔（分鐘）
    if is_correct:
        # 正確：根據連續正確次數增加間隔
        intervals = [10, 30, 60, 240, 480, 1440, 2880, 5760, 10080]  # 10分, 30分, 1時, 4時, 8時, 1天, 2天, 4天, 7天
        index = min(correct_count, len(intervals) - 1)
        minutes = intervals[index]
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
        next_review = calculate_next_review(correct, wrong, is_correct).isoformat()
        
        conn.execute('''
            UPDATE words SET quiz_correct = ?, quiz_wrong = ?, quiz_next_review = ?
            WHERE id = ?
        ''', (correct, wrong, next_review, word_id))
    else:  # typing
        correct = (word['typing_correct'] or 0) + (1 if is_correct else 0)
        wrong = (word['typing_wrong'] or 0) + (0 if is_correct else 1)
        next_review = calculate_next_review(correct, wrong, is_correct).isoformat()
        
        conn.execute('''
            UPDATE words SET typing_correct = ?, typing_wrong = ?, typing_next_review = ?
            WHERE id = ?
        ''', (correct, wrong, next_review, word_id))
    
    conn.commit()
    conn.close()
    
    return jsonify({'success': True, 'next_review': next_review})

@app.route('/api/quiz', methods=['GET'])
def get_quiz():
    conn = get_db()
    words = conn.execute('SELECT * FROM words').fetchall()
    conn.close()
    
    if len(words) < 10:
        return jsonify({'error': '資料庫中至少需要10個單字才能開始練習'}), 400
    
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
            'jlpt_level': question_word['jlpt_level'],
            'correct_meaning': question_word['chinese_meaning'],
            'kana_form': question_word['kana_form'],
            'kanji_form': question_word['kanji_form'],
            'common_form': question_word['common_form']
        },
        'options': options
    })

@app.route('/api/generate', methods=['POST'])
def generate_word_info():
    data = request.json
    japanese_word = data.get('japanese_word', '').strip()
    custom_api_key = data.get('api_key', '').strip()
    
    # 使用自訂 API key 或預設的
    api_key = custom_api_key if custom_api_key else DEFAULT_GEMINI_API_KEY
    
    if not api_key:
        return jsonify({'error': '請設定 API Key（在設定中填入或聯繫管理員）'}), 400
    
    if not japanese_word:
        return jsonify({'error': '請輸入日文詞'}), 400
    
    try:
        genai_module = get_genai_module()
        genai_module.configure(api_key=api_key)
        
        # 嘗試不同模型
        last_error = None
        for model_name in MODEL_PRIORITY:
            try:
                model = genai_module.GenerativeModel(model_name)
                
                prompt = f"""請分析這個日文詞彙「{japanese_word}」，並以 JSON 格式回傳以下資訊：
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
    custom_api_key = data.get('api_key', '').strip()
    
    api_key = custom_api_key if custom_api_key else DEFAULT_GEMINI_API_KEY
    
    if not api_key:
        return jsonify({'error': '請設定 API Key'}), 400
    
    if not words_text:
        return jsonify({'error': '請輸入要生成的詞彙'}), 400
    
    # 分割詞彙（每行一個）
    word_list = [w.strip() for w in words_text.split('\n') if w.strip()]
    
    if not word_list:
        return jsonify({'error': '沒有有效的詞彙'}), 400
    
    results = []
    errors = []
    
    try:
        genai_module = get_genai_module()
        genai_module.configure(api_key=api_key)
        
        # 找到可用的模型
        working_model = None
        for model_name in MODEL_PRIORITY:
            try:
                working_model = genai_module.GenerativeModel(model_name)
                # 測試模型
                working_model.generate_content("test")
                break
            except:
                continue
        
        if not working_model:
            return jsonify({'error': '無法連接到 AI 模型'}), 500
        
        conn = get_db()
        
        for japanese_word in word_list:
            try:
                prompt = f"""請分析這個日文詞彙「{japanese_word}」，並以 JSON 格式回傳以下資訊：
1. part_of_speech: 詞性（如：名詞、動詞、形容詞、副詞等）
2. sentence1: 一個常用的日文例句（使用這個詞）
3. sentence2: 另一個常用的日文例句（使用這個詞）
4. chinese_short: 繁體中文簡短解釋（1-3個字，只寫最核心的意思，例如：吃、喝、貓、漂亮）
5. chinese_meaning: 繁體中文完整解釋
6. jlpt_level: 用日文解釋這個詞的意思（像日日辭典一樣，純日文定義）
7. kana_form: 這個詞的純假名寫法（平假名或片假名）
8. kanji_form: 這個詞的漢字寫法（如果有的話，沒有就留空）
9. common_form: 最常用的寫法是哪種？回答 "kanji"（漢字常用）、"hiragana"（平假名常用）、"katakana"（片假名常用，常見於副詞、擬聲詞、外來語等）或 "both"（兩者都常用）

只回傳純 JSON，不要有其他文字或 markdown 格式。"""

                response = working_model.generate_content(prompt)
                result_text = response.text.strip()
                
                # 清理可能的 markdown 格式
                if result_text.startswith('```'):
                    lines = result_text.split('\n')
                    result_text = '\n'.join(lines[1:-1])
                
                result = json.loads(result_text)
                
                # 儲存到資料庫
                conn.execute('''
                    INSERT INTO words (japanese_word, part_of_speech, sentence1, sentence2, chinese_meaning, chinese_short, jlpt_level, kana_form, kanji_form, common_form)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (japanese_word, result.get('part_of_speech', ''), result.get('sentence1', ''),
                      result.get('sentence2', ''), result.get('chinese_meaning', ''),
                      result.get('chinese_short', ''), result.get('jlpt_level', ''),
                      result.get('kana_form', ''), result.get('kanji_form', ''),
                      result.get('common_form', 'kanji')))
                
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
            'errors': errors
        })
        
    except Exception as e:
        return jsonify({'error': f'批次生成失敗: {str(e)}'}), 500

@app.route('/api/typing-quiz', methods=['GET'])
def get_typing_quiz():
    """打字測驗 - 顯示漢字，回答假名"""
    conn = get_db()
    # 只選擇有漢字寫法和假名寫法的單字
    words = conn.execute('''
        SELECT * FROM words 
        WHERE kanji_form IS NOT NULL AND kanji_form != '' 
        AND kana_form IS NOT NULL AND kana_form != ''
    ''').fetchall()
    conn.close()
    
    if len(words) < 1:
        return jsonify({'error': '資料庫中沒有足夠的單字（需要有漢字和假名寫法的單字）'}), 400
    
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
            'jlpt_level': question_word['jlpt_level'],
            'chinese_short': question_word['chinese_short'],
            'chinese_meaning': question_word['chinese_meaning']
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
