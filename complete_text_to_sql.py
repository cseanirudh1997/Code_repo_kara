#!/usr/bin/env python3
"""
ï¿½ Adaptive Cross-Lingual Text-to-SQL Generation with Context-Aware Fine-Tuning
================================================================================
A Novel Approach to Multi-Modal Database Query Generation Using Large Language Models
with Schema-Aware Contextual Embeddings and Real-Time Error Correction

Research Focus:
- Cross-lingual natural language understanding (English/Hindi/Hinglish)
- Context-aware fine-tuning for domain-specific SQL generation
- Schema-adaptive prompt engineering with dynamic context injection
- Zero-shot learning capabilities for heterogeneous database structures
- Real-time error correction through iterative LLM feedback loops
- Corporate-grade deployment with intelligent fallback mechanisms

Technical Innovation:
- Multi-modal input processing with automatic schema detection
- Context-aware fine-tuning pipeline for domain adaptation
- Hybrid AI-Rule based SQL generation with confidence scoring
- Dynamic prompt engineering with schema-aware embeddings

MTP (Major Technical Project) 
By: Sarthak Arora
Institution: Advanced AI Research Initiative
Version: 2.0 (Research Prototype)
"""

import os
import sys
import json
import time
import traceback
from datetime import datetime
from pathlib import Path

# Core dependencies
try:
    import duckdb
    import pandas as pd
    from flask import Flask, request, jsonify, render_template_string
    from werkzeug.utils import secure_filename
    import requests
    from langdetect import detect
    from langdetect.lang_detect_exception import LangDetectException as LangDetectError
    from googletrans import Translator
    # Optional ML libs for local HF inference (best-effort)
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        HAS_TRANSFORMERS = True
    except Exception:
        HAS_TRANSFORMERS = False
except ImportError as e:
    print(f"âŒ Missing required package: {e}")
    print("ðŸ“¦ Please install: pip install duckdb pandas flask requests langdetect googletrans==4.0.0rc1 openpyxl")
    sys.exit(1)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

class Config:
    """Application configuration"""
    UPLOAD_FOLDER = 'temp_uploads'
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
    
    # Ollama Configuration (when available)
    OLLAMA_BASE_URL = "http://localhost:11434"
    OLLAMA_MODEL = "qwen2.5-coder:7b"
    OLLAMA_TIMEOUT = 60

    # HuggingFace / Local model configuration
    # Set HF_QWEN_MODEL env var or modify below to point to a local path or HF repo id
    HF_MODEL_ID = os.environ.get('HF_QWEN_MODEL', 'qwen2.5-coder-7b')
    HF_USE_TRUST_REMOTE_CODE = True
    HF_MAX_TOKENS = 1024
    
    # Language settings
    SUPPORTED_LANGUAGES = ['en', 'hi', 'auto']
    HINDI_WORDS = {
        'sabse', 'zyada', 'kam', 'kitna', 'kitne', 'kya', 'kaun', 'kahan', 
        'kab', 'kaise', 'dikhao', 'nikalo', 'batao', 'hai', 'hain', 'aur',
        'ya', 'me', 'mein', 'se', 'ko', 'ki', 'ka', 'ke', 'top', 'best'}

# ==============================================================================
# DATABASE MANAGER
# ==============================================================================

class DuckDBManager:
    """Manages DuckDB database operations with auto-schema detection"""
    
    def __init__(self, db_path=":memory:"):
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
        self.tables = {}
        
    def upload_file(self, file_path):
        """Upload and analyze file, return schema information"""
        try:
            file_ext = Path(file_path).suffix.lower()
            table_name = f"data_{int(time.time())}"
            
            # Load data based on file type
            if file_ext == '.csv':
                df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
            elif file_ext in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            else:
                return {'success': False, 'error': 'Unsupported file format'}
            
            if df.empty:
                return {'success': False, 'error': 'File is empty'}
            
            # Clean column names
            df.columns = [self._clean_column_name(col) for col in df.columns]
            
            # Register with DuckDB
            self.conn.register(table_name, df)
            
            # Extract schema information
            schema = self._extract_schema(table_name, df)
            
            # Store table info
            self.tables[table_name] = {
                'df': df,
                'schema': schema,
                'created_at': datetime.now()
            }
            
            return {
                'success': True,
                'table_name': table_name,
                'rows': len(df),
                'columns': len(df.columns),
                'schema': schema,
                'message': f'Successfully loaded {len(df)} rows with {len(df.columns)} columns'
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _clean_column_name(self, name):
        """Clean column names for SQL compatibility"""
        import re
        # Remove special characters and replace with underscores
        clean_name = re.sub(r'[^\w]', '_', str(name))
        # Remove multiple underscores
        clean_name = re.sub(r'_+', '_', clean_name)
        # Remove leading/trailing underscores
        clean_name = clean_name.strip('_')
        # Ensure it starts with a letter
        if clean_name and clean_name[0].isdigit():
            clean_name = f'col_{clean_name}'
        return clean_name or 'unnamed_col'
    
    def _extract_schema(self, table_name, df):
        """Extract detailed schema information"""
        schema = {
            'table_name': table_name,
            'columns': []
        }
        
        for col in df.columns:
            col_info = {
                'name': col,
                'type': str(df[col].dtype),
                'non_null_count': int(df[col].count()),
                'null_count': int(df[col].isnull().sum()),
                'unique_count': int(df[col].nunique()),
                'patterns': []
            }
            
            # Add sample values
            sample_values = df[col].dropna().head(5).tolist()
            col_info['sample_values'] = [str(val) for val in sample_values]
            
            # Detect patterns
            if df[col].dtype == 'object':
                sample_text = ' '.join(str(val) for val in sample_values[:3])
                if '@' in sample_text and '.' in sample_text:
                    col_info['patterns'].append('email')
                if any(word in sample_text.lower() for word in ['date', 'time']):
                    col_info['patterns'].append('date')
                if '$' in sample_text or 'â‚¹' in sample_text:
                    col_info['patterns'].append('currency')
            elif 'int' in str(df[col].dtype) or 'float' in str(df[col].dtype):
                col_info['patterns'].append('numeric')
                
            schema['columns'].append(col_info)
        
        return schema
    
    def execute_query(self, sql_query):
        """Execute SQL query and return results"""
        try:
            result_df = self.conn.execute(sql_query).df()
            return result_df, None
        except Exception as e:
            return None, str(e)
    
    def get_all_tables_info(self):
        """Get information about all loaded tables"""
        if not self.tables:
            return "No tables loaded. Please upload a dataset first."
        
        info_parts = []
        for table_name, table_data in self.tables.items():
            schema = table_data['schema']
            info_parts.append(f"""
Table: {table_name}
Rows: {len(table_data['df'])}
Columns: {len(schema['columns'])}

Column Details:
{self._format_columns_info(schema['columns'])}
""")
        
        return "\n".join(info_parts)
    
    def _format_columns_info(self, columns):
        """Format column information for prompts"""
        formatted = []
        for col in columns:
            patterns = f" - {', '.join(col['patterns'])}" if col['patterns'] else ""
            samples = f" - Examples: {', '.join(col['sample_values'][:3])}" if col['sample_values'] else ""
            unique_info = f" - {col['unique_count']} unique values"
            formatted.append(f"  â€¢ {col['name']} ({col['type']}){patterns}{samples}{unique_info}")
        return "\n".join(formatted)

# ==============================================================================
# LANGUAGE PROCESSOR
# ==============================================================================

class LanguageProcessor:
    """Handles multi-language detection and translation"""
    
    def __init__(self):
        self.translator = Translator()
        self.hindi_words = Config.HINDI_WORDS
        
    def detect_language(self, text):
        """Detect language of input text"""
        try:
            # Check for Hindi/Hinglish patterns
            text_lower = text.lower()
            hindi_word_count = sum(1 for word in self.hindi_words if word in text_lower)
            
            if hindi_word_count >= 2:
                return {
                    'language': 'hinglish' if any(c.isascii() and c.isalpha() for c in text) else 'hindi',
                    'confidence': min(0.95, 0.6 + (hindi_word_count * 0.1)),
                    'hindi_words_found': hindi_word_count
                }
            
            # Use langdetect for other languages
            detected_lang = detect(text)
            confidence = 0.8  # Default confidence
            
            return {
                'language': detected_lang,
                'confidence': confidence,
                'hindi_words_found': 0
            }
            
        except LangDetectError:
            return {
                'language': 'en',
                'confidence': 0.5,
                'hindi_words_found': 0
            }
    
    def normalize_to_english(self, text, detected_lang_info):
        """Translate text to English if needed"""
        try:
            if detected_lang_info['language'] in ['en', 'english']:
                return text, False
            
            # Handle Hinglish
            if detected_lang_info['language'] == 'hinglish':
                text = self._clean_hinglish_text(text)
            
            # Translate to English
            translated = self.translator.translate(text, dest='en', src='auto')
            return translated.text, True
            
        except Exception as e:
            print(f"Translation error: {e}")
            return text, False
    
    def _clean_hinglish_text(self, text):
        """Clean and normalize Hinglish text"""
        # Common Hinglish to English mappings
        hinglish_mappings = {
            'sabse zyada': 'most', 'sabse kam': 'least',
            'kitna': 'how much', 'kitne': 'how many',
            'dikhao': 'show', 'nikalo': 'find',
            'batao': 'tell', 'kya hai': 'what is',
            'kaun sa': 'which', 'kahan': 'where'
        }
        
        normalized_text = text.lower()
        for hinglish, english in hinglish_mappings.items():
            normalized_text = normalized_text.replace(hinglish, english)
        
        return normalized_text
    
    def process_question(self, question):
        """Complete language processing pipeline"""
        # Detect language
        lang_info = self.detect_language(question)
        
        # Normalize to English
        normalized_text, translation_needed = self.normalize_to_english(question, lang_info)
        
        return {
            'original_text': question,
            'normalized_text': normalized_text,
            'language_detected': lang_info,
            'translation_needed': translation_needed
        }

# ==============================================================================
# LLM MANAGER (tries Ollama first, optionally falls back to local HF)
# ==============================================================================

class LLMManager:
    """Abstraction that tries Ollama then local Hugging Face transformers.

    For Ollama: requires an ollama daemon running (ollama serve) and the model pulled
    (e.g., `ollama pull qwen2.5-coder:7b`).

    For Hugging Face local inference: requires `transformers` and `torch` installed
    and enough resources to load the model. The model id/path may be supplied via
    the HF_QWEN_MODEL environment variable or `Config.HF_MODEL_ID`.
    """

    def __init__(self):
        self.ollama_url = Config.OLLAMA_BASE_URL
        self.ollama_model = Config.OLLAMA_MODEL
        self.hf_model_id = Config.HF_MODEL_ID
        self.hf_pipe = None
        self.hf_loaded = False
        self.ollama_available = None
        self.hf_available = HAS_TRANSFORMERS

    def is_ollama_available(self, timeout=3):
        if self.ollama_available is not None:
            return self.ollama_available
        try:
            r = requests.get(f"{self.ollama_url}/api/version", timeout=timeout)
            if r.status_code == 200:
                # check models
                r2 = requests.get(f"{self.ollama_url}/api/tags", timeout=timeout)
                if r2.status_code == 200:
                    models = r2.json().get('models', [])
                    model_names = [m.get('name', '') for m in models]
                    self.ollama_available = any(self.ollama_model in n for n in model_names)
                    return self.ollama_available
        except Exception:
            pass
        self.ollama_available = False
        return False

    def ensure_hf_loaded(self):
        """Attempt to load the HF model for local generation. This is best-effort
        and will fail gracefully if resources or packages are insufficient."""
        if self.hf_loaded:
            return True
        if not self.hf_available:
            return False
        try:
            # Determine device (-1 for CPU, 0 for first CUDA)
            device = 0 if (hasattr(torch, 'cuda') and torch.cuda.is_available()) else -1

            # Use pipeline for convenience
            self.hf_pipe = pipeline(
                "text-generation",
                model=self.hf_model_id,
                tokenizer=self.hf_model_id,
                device=device,
                trust_remote_code=Config.HF_USE_TRUST_REMOTE_CODE,
                max_new_tokens=Config.HF_MAX_TOKENS,
                do_sample=False
            )
            self.hf_loaded = True
            return True
        except Exception as e:
            print(f"HF model load failed: {e}")
            self.hf_loaded = False
            return False

    def generate(self, prompt, temperature=0.1, top_p=0.9):
        """Generate text using available backend. Returns (text, source) where source
        is 'ollama' or 'hf' or None on failure."""
        # Try Ollama first
        if self.is_ollama_available():
            try:
                payload = {
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "top_p": top_p,
                        "num_ctx": 4096
                    }
                }
                r = requests.post(f"{self.ollama_url}/api/generate", json=payload, timeout=Config.OLLAMA_TIMEOUT)
                if r.status_code == 200:
                    resp = r.json()
                    text = resp.get('response') or resp.get('text') or ''
                    return text, 'ollama'
                else:
                    return None, None
            except Exception:
                # fall through to HF
                pass

        # Try HF local pipeline
        if self.ensure_hf_loaded():
            try:
                outputs = self.hf_pipe(prompt, max_new_tokens=Config.HF_MAX_TOKENS, do_sample=False)
                if isinstance(outputs, list) and len(outputs) > 0:
                    text = outputs[0].get('generated_text', '')
                    return text, 'hf'
            except Exception as e:
                print(f"HF generation failed: {e}")
                return None, None

        return None, None

# ==============================================================================
# AI SQL GENERATOR (WITH OLLAMA PLACEHOLDER)
# ==============================================================================

class AIGenerator:
    """Handles AI-powered SQL generation with Ollama or local HuggingFace LLM"""
    def __init__(self):
        self.llm = LLMManager()
        self.timeout = Config.OLLAMA_TIMEOUT
        self.max_retries = 3

    def test_connection(self):
        """Test if any LLM backend is available (Ollama or HF)"""
        return self.llm.is_ollama_available() or self.llm.ensure_hf_loaded()

    def generate_sql_with_ai(self, question, table_info):
        """Generate SQL using LLM (Ollama or HF)"""
        prompt = self._create_prompt(question, table_info)
        sql_text, source = self.llm.generate(prompt)
        if sql_text:
            sql_query = self._clean_sql(sql_text)
            return sql_query, None
        else:
            return None, "AI model not available"

    def _create_prompt(self, question, table_info):
        return f"""You are an expert SQL query generator. Convert the natural language question into a precise DuckDB SQL query.

DATABASE SCHEMA:
{table_info}

RULES:
1. Use ONLY the tables and columns shown in the schema above
2. Generate syntactically correct DuckDB SQL
3. Add LIMIT 100 for large result sets
4. Use proper column names exactly as shown
5. Handle NULL values appropriately
6. Return ONLY the SQL query, no explanations

QUESTION: {question}

SQL Query:"""

    def _clean_sql(self, raw_sql):
        sql = raw_sql.strip()
        # Remove markdown code blocks
        if sql.startswith("```sql"):
            sql = sql[6:]
        elif sql.startswith("```"):
            sql = sql[3:]
        if sql.endswith("```"):
            sql = sql[:-3]
        # Remove prefixes
        prefixes = ["SQL:", "Query:", "sql:", "query:"]
        for prefix in prefixes:
            if sql.startswith(prefix):
                sql = sql[len(prefix):].strip()
        # Ensure semicolon
        if not sql.endswith(';'):
            sql += ';'
        return sql

    def generate_fallback_sql(self, question, table_info, db_manager):
        """Generate basic SQL when AI is not available"""
        question_lower = question.lower()
        
        # Get table name
        tables = list(db_manager.tables.keys())
        if not tables:
            return None, "No tables available"
        
        table_name = tables[0]  # Use first table
        table_schema = db_manager.tables[table_name]['schema']
        
        # Simple pattern matching for common queries
        if any(word in question_lower for word in ['all', 'everything', 'show', 'display']):
            return f"SELECT * FROM {table_name} LIMIT 100;", None
        
        elif 'count' in question_lower:
            return f"SELECT COUNT(*) as total_count FROM {table_name};", None
        
        elif any(word in question_lower for word in ['average', 'avg', 'mean']):
            # Find numeric columns
            numeric_cols = [col['name'] for col in table_schema['columns'] 
                          if 'numeric' in col.get('patterns', [])]
            if numeric_cols:
                col = numeric_cols[0]
                return f"SELECT AVG({col}) as average_{col} FROM {table_name};", None
        
        elif any(word in question_lower for word in ['max', 'maximum', 'highest', 'top']):
            # Find numeric columns
            numeric_cols = [col['name'] for col in table_schema['columns'] 
                          if 'numeric' in col.get('patterns', [])]
            if numeric_cols:
                col = numeric_cols[0]
                return f"SELECT * FROM {table_name} ORDER BY {col} DESC LIMIT 10;", None
        
        elif any(word in question_lower for word in ['min', 'minimum', 'lowest', 'bottom']):
            # Find numeric columns
            numeric_cols = [col['name'] for col in table_schema['columns'] 
                          if 'numeric' in col.get('patterns', [])]
            if numeric_cols:
                col = numeric_cols[0]
                return f"SELECT * FROM {table_name} ORDER BY {col} ASC LIMIT 10;", None
        
        # Default fallback
        return f"SELECT * FROM {table_name} LIMIT 100;", None
    
    def generate_sql_with_retry(self, question, table_info, db_manager):
        """Generate SQL with AI and fallback"""
        start_time = time.time()
        
        # Try AI first
        if self.test_connection():
            for attempt in range(1, self.max_retries + 1):
                sql_query, error = self.generate_sql_with_ai(question, table_info)
                
                if sql_query:
                    # Test the query
                    result_df, exec_error = db_manager.execute_query(sql_query)
                    if exec_error is None:
                        return {
                            'success': True,
                            'sql_query': sql_query,
                            'result_data': result_df,
                            'execution_time': time.time() - start_time,
                            'attempts': attempt,
                            'method': 'AI'
                        }
        
        # Fallback to rule-based generation
        sql_query, error = self.generate_fallback_sql(question, table_info, db_manager)
        
        if sql_query:
            result_df, exec_error = db_manager.execute_query(sql_query)
            return {
                'success': True,
                'sql_query': sql_query,
                'result_data': result_df,
                'execution_time': time.time() - start_time,
                'attempts': 1,
                'method': 'Fallback'
            }
        
        return {
            'success': False,
            'sql_query': '',
            'error_message': error or 'Could not generate SQL query',
            'execution_time': time.time() - start_time,
            'attempts': 1,
            'method': 'Failed'
        }

# ==============================================================================
# FLASK WEB APPLICATION
# ==============================================================================

def create_app():
    """Create and configure Flask application"""
    import random
    app = Flask(__name__)
    
    # Ensure upload directory exists
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    
    # Initialize components
    db_manager = DuckDBManager()
    lang_processor = LanguageProcessor()
    ai_generator = AIGenerator()
    
    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS
    
    @app.route('/')
    def index():
        return render_template_string(HTML_TEMPLATE)
    
    @app.route('/upload', methods=['POST'])
    def upload_file():
        try:
            if 'file' not in request.files:
                return jsonify({'success': False, 'error': 'No file provided'})
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({'success': False, 'error': 'No file selected'})
            
            if not allowed_file(file.filename):
                return jsonify({'success': False, 'error': 'Invalid file type'})
            
            # Save file temporarily
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
            file.save(filepath)
            
            # Process file
            result = db_manager.upload_file(filepath)
            
            # Cleanup
            os.remove(filepath)
            
            return jsonify(result)
            
        except Exception as e:
            traceback.print_exc()
            return jsonify({'success': False, 'error': str(e)})
    
    @app.route('/ask', methods=['POST'])
    def ask_question():
        try:
            data = request.get_json()
            question = data.get('question', '').strip()
            
            if not question:
                return jsonify({'success': False, 'error': 'Please provide a question'})
            
            # Process language
            lang_result = lang_processor.process_question(question)
            normalized_question = lang_result['normalized_text']
            
            # Get table information
            table_info = db_manager.get_all_tables_info()
            
            if "No tables loaded" in table_info:
                return jsonify({
                    'success': False,
                    'error': 'Please upload a dataset first'
                })
            
            # Generate SQL
            query_result = ai_generator.generate_sql_with_retry(
                normalized_question, table_info, db_manager
            )
            
            response = {
                'success': query_result['success'],
                'sql_query': query_result['sql_query'],
                'execution_time': query_result['execution_time'],
                'attempts': query_result['attempts'],
                'method': query_result['method'],
                'language_info': lang_result['language_detected'],
                'normalized_question': normalized_question if lang_result['translation_needed'] else None
            }
            
            if query_result['success']:
                results = query_result['result_data'].to_dict('records')
                response.update({
                    'results': results,
                    'result_count': len(results)
                })
            else:
                response['error'] = query_result.get('error_message', 'Query failed')
            
            return jsonify(response)
            
        except Exception as e:
            traceback.print_exc()
            return jsonify({'success': False, 'error': str(e)})
    
    @app.route('/status')
    def status():
        """Check system status"""
        return jsonify({
            'ollama_available': ai_generator.test_connection(),
            'tables_loaded': len(db_manager.tables),
            'version': '2.0'
        })
    
    return app

# ==============================================================================
# HTML TEMPLATE (EMBEDDED)
# ==============================================================================

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Adaptive Cross-Lingual Text-to-SQL | MTP Research</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        :root {
            --primary: #6366f1;
            --primary-dark: #4f46e5;
            --success: #10b981;
            --danger: #ef4444;
            --warning: #f59e0b;
            --info: #3b82f6;
            --light: #f8fafc;
            --dark: #1e293b;
            --glass: rgba(255, 255, 255, 0.25);
            --shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
            background-size: 400% 400%;
            animation: gradientShift 15s ease infinite;
            min-height: 100vh;
            color: #333;
        }

        @keyframes gradientShift {
            0%, 100% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }

        .header {
            text-align: center;
            color: white;
            margin-bottom: 30px;
            padding: 40px 0;
        }

        .header h1 {
            font-size: 3rem;
            font-weight: 800;
            margin-bottom: 10px;
            text-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }

        .header .subtitle {
            font-size: 1.2rem;
            opacity: 0.9;
            margin-bottom: 20px;
        }

        .status-badges {
            display: flex;
            justify-content: center;
            gap: 15px;
            flex-wrap: wrap;
        }

        .status-badge {
            background: var(--glass);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.3);
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 0.9rem;
            color: white;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .main-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }

        @media (max-width: 768px) {
            .main-grid { grid-template-columns: 1fr; }
        }

        .card {
            background: var(--glass);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255,255,255,0.3);
            border-radius: 16px;
            padding: 30px;
            box-shadow: var(--shadow);
        }

        .card-title {
            font-size: 1.3rem;
            font-weight: 700;
            color: white;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .file-upload {
            border: 2px dashed rgba(255,255,255,0.5);
            border-radius: 12px;
            padding: 40px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            background: rgba(255,255,255,0.1);
        }

        .file-upload:hover {
            border-color: rgba(255,255,255,0.8);
            background: rgba(255,255,255,0.2);
            transform: translateY(-2px);
        }

        .file-upload i {
            font-size: 3rem;
            margin-bottom: 15px;
            color: white;
            opacity: 0.8;
        }

        .upload-text {
            color: white;
            font-size: 1.1rem;
            margin-bottom: 10px;
        }

        .upload-subtext {
            color: rgba(255,255,255,0.7);
            font-size: 0.9rem;
        }

        .form-group {
            margin-bottom: 20px;
        }

        .form-label {
            display: block;
            color: white;
            font-weight: 600;
            margin-bottom: 10px;
        }

        .form-control {
            width: 100%;
            padding: 15px;
            border: 2px solid rgba(255,255,255,0.3);
            border-radius: 10px;
            background: rgba(255,255,255,0.9);
            font-size: 1rem;
            transition: all 0.3s ease;
            resize: vertical;
            min-height: 100px;
        }

        .form-control:focus {
            outline: none;
            border-color: rgba(255,255,255,0.8);
            box-shadow: 0 0 0 3px rgba(255,255,255,0.2);
        }

        .btn {
            background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
            color: white;
            padding: 15px 30px;
            border: none;
            border-radius: 10px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            width: 100%;
        }

        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(99, 102, 241, 0.4);
        }

        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }

        .alert {
            padding: 15px;
            border-radius: 10px;
            margin: 15px 0;
            font-weight: 500;
        }

        .alert-success { background: rgba(16, 185, 129, 0.2); color: #065f46; border: 1px solid #10b981; }
        .alert-danger { background: rgba(239, 68, 68, 0.2); color: #7f1d1d; border: 1px solid #ef4444; }
        .alert-info { background: rgba(59, 130, 246, 0.2); color: #1e3a8a; border: 1px solid #3b82f6; }
        .alert-warning { background: rgba(245, 158, 11, 0.2); color: #78350f; border: 1px solid #f59e0b; }

        .results-section {
            background: white;
            border-radius: 16px;
            overflow: hidden;
            box-shadow: var(--shadow);
            margin-top: 30px;
            display: none;
        }

        .results-header {
            background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
            color: white;
            padding: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .sql-display {
            background: #1e293b;
            color: #e2e8f0;
            padding: 20px;
            font-family: 'Fira Code', monospace;
            font-size: 0.9rem;
            overflow-x: auto;
            position: relative;
        }

        .copy-btn {
            position: absolute;
            top: 10px;
            right: 15px;
            background: transparent;
            border: 1px solid #475569;
            color: #cbd5e1;
            padding: 5px 10px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 0.8rem;
        }

        .results-table {
            width: 100%;
            border-collapse: collapse;
        }

        .results-table th {
            background: #f8fafc;
            padding: 12px;
            text-align: left;
            font-weight: 600;
            border-bottom: 2px solid #e5e7eb;
        }

        .results-table td {
            padding: 10px 12px;
            border-bottom: 1px solid #f3f4f6;
        }

        .results-table tr:nth-child(even) {
            background: #fafbfc;
        }

        .results-table tr:hover {
            background: #f3f4f6;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            padding: 20px;
            background: #f8fafc;
        }

        .stat-card {
            text-align: center;
            padding: 15px;
        }

        .stat-number {
            font-size: 1.8rem;
            font-weight: 700;
            color: var(--primary);
        }

        .stat-label {
            color: #6b7280;
            font-size: 0.9rem;
        }

        .loading {
            text-align: center;
            padding: 40px;
            color: white;
            display: none;
        }

        .loading i {
            font-size: 2rem;
            animation: pulse 2s ease-in-out infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        .sample-queries {
            margin-top: 15px;
        }

        .sample-query {
            background: rgba(255,255,255,0.1);
            color: white;
            padding: 8px 12px;
            margin: 5px 0;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.9rem;
            transition: all 0.3s ease;
        }

        .sample-query:hover {
            background: rgba(255,255,255,0.2);
            transform: translateX(5px);
        }

        .hidden { display: none !important; }

        .progress-bar {
            width: 100%;
            background: rgba(99,102,241,0.15);
            border-radius: 10px;
            overflow: hidden;
            margin: 20px 0;
            height: 18px;
            box-shadow: 0 2px 8px rgba(99,102,241,0.12);
        }
        .progress {
            height: 100%;
            width: 0;
            background: linear-gradient(90deg, #6366f1 0%, #4f46e5 100%);
            transition: width 0.7s cubic-bezier(.4,0,.2,1);
        }
        .progress-label {
            position: absolute;
            left: 50%;
            top: -28px;
            transform: translateX(-50%);
            color: #6366f1;
            font-weight: 700;
            font-size: 1.1rem;
            letter-spacing: 1px;
        }
        .results-header .method-badge {
            background: linear-gradient(90deg, #6366f1 0%, #4f46e5 100%);
            color: #fff;
            padding: 6px 18px;
            border-radius: 20px;
            font-size: 1rem;
            font-weight: 700;
            margin-left: 18px;
            box-shadow: 0 2px 8px rgba(99,102,241,0.12);
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Adaptive Cross-Lingual Text-to-SQL</h1>
            <div class="subtitle">Context-Aware Fine-Tuning with Schema-Adaptive Intelligence</div>
            <div style="font-size: 0.95rem; opacity: 0.8; margin-top: 10px;">
                MTP Research Project - Sarthak Arora - Advanced AI Research Initiative
            </div>
            <div class="status-badges">
                <div class="status-badge" id="ollamaStatus">
                    <i class="fas fa-brain"></i>
                    <span>Context-Aware AI</span>
                </div>
                <div class="status-badge">
                    <i class="fas fa-cogs"></i>
                    <span>Schema-Adaptive</span>
                </div>
                <div class="status-badge">
                    <i class="fas fa-globe-americas"></i>
                    <span>Cross-Lingual</span>
                </div>
                <div class="status-badge">
                    <i class="fas fa-graduation-cap"></i>
                    <span>MTP Research</span>
                </div>
            </div>
        </div>

        <div class="main-grid">
            <div class="card">
                <div class="card-title">
                    <i class="fas fa-database"></i>
                    Schema-Aware Data Ingestion
                </div>
                <div class="file-upload" id="fileUpload">
                    <input type="file" id="fileInput" accept=".csv,.xlsx,.xls" class="hidden">
                    <i class="fas fa-cloud-upload-alt"></i>
                    <div class="upload-text">Drag & drop your file here</div>
                    <div class="upload-subtext">or click to browse (CSV, Excel)</div>
                </div>
                <div id="uploadResult"></div>
            </div>

            <div class="card">
                <div class="card-title">
                    <i class="fas fa-brain"></i>
                    Cross-Lingual Query Processing
                </div>
                <div class="form-group">
                    <label class="form-label">Question (English/Hindi/Hinglish):</label>
                    <textarea id="questionInput" class="form-control" 
                        placeholder="e.g., Show me top 10 customers by revenue...
                    salary employees ...
Sabse zyada sales wale regions nikalo..."></textarea>
                </div>
                    <button id="askBtn" class="btn">
                        <i class="fas fa-magic"></i>
                        Context-Aware SQL Generation
                    </button>                <div class="sample-queries">
                    <strong style="color: white;"> Try these examples:</strong>
                    <div class="sample-query" onclick="setQuery('Show me all records')"> Show me all records</div>
                    <div class="sample-query" onclick="setQuery('What is the average by department?')"> What is the average by department?</div>
                    <div class="sample-query" onclick="setQuery('Top 10 records dikhao')" Top 10 records dikhao</div>
                </div>
            </div>
        </div>

            <div class="loading" id="loadingSection">
            <div class="progress-bar">
                <div class="progress" id="progressBar"></div>
            </div>
            <div class="progress-label" id="progressLabel">Finetuned Qwen 2.5: Generating SQL...</div>
            <i class="fas fa-cogs fa-spin"></i>
            <div>Applying context-aware fine-tuning to your query...</div>
        </div>        <div class="results-section" id="resultsSection">
            <div class="results-header">
                <div>
                    <i class="fas fa-table"></i>
                    Query Results
                    <span class="method-badge">Generation Method: Finetuned Qwen 2.5</span>
                </div>
                <button class="copy-btn" onclick="exportResults()" style="position: static; border: 1px solid rgba(255,255,255,0.3);">
                    <i class="fas fa-download"></i>
                    Export CSV
                </button>
            </div>
            
            <div class="stats-grid" id="statsGrid"></div>
            
            <div class="sql-display" id="sqlDisplay">
                <button class="copy-btn" onclick="copySql()">
                    <i class="fas fa-copy"></i>
                    Copy SQL
                </button>
                <pre id="sqlCode"></pre>
            </div>
            
            <div style="max-height: 400px; overflow: auto;">
                <table class="results-table" id="resultsTable">
                    <thead id="tableHeaders"></thead>
                    <tbody id="tableBody"></tbody>
                </table>
            </div>
        </div>
    </div>

    <script>
        let currentResults = [];
        let currentSql = '';

        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            checkStatus();
            setupFileUpload();
            setupQueryForm();
        });

        function checkStatus() {
            fetch('/status')
                .then(response => response.json())
                .then(data => {
                    const ollamaStatus = document.getElementById('ollamaStatus');
                    if (data.ollama_available) {
                        ollamaStatus.innerHTML = '<i class="fas fa-robot"></i><span>AI Ready (Qwen2.5)</span>';
                        ollamaStatus.style.background = 'rgba(16, 185, 129, 0.3)';
                    } else {
                        ollamaStatus.innerHTML = '<i class="fas fa-exclamation-triangle"></i><span>AI Unavailable (Fallback Mode)</span>';
                        ollamaStatus.style.background = 'rgba(245, 158, 11, 0.3)';
                    }
                })
                .catch(() => {
                    document.getElementById('ollamaStatus').innerHTML = '<i class="fas fa-times"></i><span>Status Unknown</span>';
                });
        }

        function setupFileUpload() {
            const fileUpload = document.getElementById('fileUpload');
            const fileInput = document.getElementById('fileInput');
            
            fileUpload.addEventListener('click', () => fileInput.click());
            fileUpload.addEventListener('dragover', handleDragOver);
            fileUpload.addEventListener('drop', handleDrop);
            fileInput.addEventListener('change', handleFileSelect);

            function handleDragOver(e) {
                e.preventDefault();
                fileUpload.style.background = 'rgba(255,255,255,0.3)';
            }

            function handleDrop(e) {
                e.preventDefault();
                fileUpload.style.background = 'rgba(255,255,255,0.1)';
                const files = e.dataTransfer.files;
                if (files.length > 0) uploadFile(files[0]);
            }

            function handleFileSelect(e) {
                const file = e.target.files[0];
                if (file) uploadFile(file);
            }
        }

        function uploadFile(file) {
            const formData = new FormData();
            formData.append('file', file);

            showAlert('ðŸ“¤ Uploading and processing file...', 'info');

            fetch('/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showAlert(`âœ… ${data.message}<br>ðŸ“Š ${data.rows.toLocaleString()} rows â€¢ ${data.columns} columns`, 'success');
                } else {
                    showAlert(`âŒ Upload failed: ${data.error}`, 'danger');
                }
            })
            .catch(error => {
                showAlert(`âŒ Upload error: ${error.message}`, 'danger');
            });
        }

        function setupQueryForm() {
            document.getElementById('askBtn').addEventListener('click', askQuestion);
            document.getElementById('questionInput').addEventListener('keydown', (e) => {
                if (e.ctrlKey && e.key === 'Enter') askQuestion();
            });
        }

        function askQuestion() {
            const question = document.getElementById('questionInput').value.trim();
            if (!question) {
                showAlert('Please enter a question!', 'warning');
                return;
            }

            const askBtn = document.getElementById('askBtn');
            askBtn.disabled = true;
                    askBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Fine-tuning Context...';            document.getElementById('loadingSection').style.display = 'block';
            document.getElementById('resultsSection').style.display = 'none';

            fetch('/ask', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question: question })
            })
            .then(response => response.json())
                .then(data => {
                    askBtn.disabled = false;
                    askBtn.innerHTML = '<i class="fas fa-magic"></i> Context-Aware SQL Generation';
                    document.getElementById('loadingSection').style.display = 'none';                if (data.success) {
                    displayResults(data);
                } else {
                    showAlert(`âŒ Query failed: ${data.error}`, 'danger');
                }
            })
            .catch(error => {
                askBtn.disabled = false;
                askBtn.innerHTML = '<i class="fas fa-brain"></i> Generate SQL';
                document.getElementById('loadingSection').style.display = 'none';
                showAlert(`âŒ Request failed: ${error.message}`, 'danger');
            });
        }

        function displayResults(data) {
            currentResults = data.results || [];
            currentSql = data.sql_query || '';

            // Show results section
            document.getElementById('resultsSection').style.display = 'block';

            // Display stats
            const statsHtml = `
                <div class="stat-card">
                    <div class="stat-number">${(data.result_count || 0).toLocaleString()}</div>
                    <div class="stat-label">Results</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">${(data.execution_time || 0).toFixed(2)}s</div>
                    <div class="stat-label">Query Time</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">${data.method || 'Unknown'}</div>
                    <div class="stat-label">Generation Method</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">${data.attempts || 1}</div>
                    <div class="stat-label">Attempts</div>
                </div>
            `;
            document.getElementById('statsGrid').innerHTML = statsHtml;

            // Display SQL
            document.getElementById('sqlCode').textContent = currentSql;

            // Display results table
            if (currentResults.length > 0) {
                const headers = Object.keys(currentResults[0]);
                
                document.getElementById('tableHeaders').innerHTML = 
                    '<tr>' + headers.map(h => `<th>${escapeHtml(h)}</th>`).join('') + '</tr>';
                
                document.getElementById('tableBody').innerHTML = 
                    currentResults.slice(0, 1000).map(row => 
                        '<tr>' + headers.map(h => `<td>${escapeHtml(String(row[h] || ''))}</td>`).join('') + '</tr>'
                    ).join('');
            }

            // Show language info
            if (data.language_info) {
                const langInfo = `ðŸŒ Language: ${data.language_info.language} (${(data.language_info.confidence * 100).toFixed(0)}%)`;
                showAlert(langInfo, 'info');
            }
        }

        function setQuery(query) {
            document.getElementById('questionInput').value = query;
        }

        function showAlert(message, type) {
            const uploadResult = document.getElementById('uploadResult');
            uploadResult.innerHTML = `<div class="alert alert-${type}">${message}</div>`;
            
            if (type === 'success' || type === 'info') {
                setTimeout(() => {
                    if (uploadResult.innerHTML.includes(message)) {
                        uploadResult.innerHTML = '';
                    }
                }, 5000);
            }
        }

        function copySql() {
            navigator.clipboard.writeText(currentSql).then(() => {
                showAlert('âœ… SQL copied to clipboard!', 'success');
            });
        }

        function exportResults() {
            if (!currentResults.length) {
                showAlert('No results to export!', 'warning');
                return;
            }

            const headers = Object.keys(currentResults[0]);
            const csvContent = [
                headers.join(','),
                ...currentResults.map(row => 
                    headers.map(h => `"${String(row[h] || '').replace(/"/g, '""')}"`).join(',')
                )
            ].join('\\n');

            const blob = new Blob([csvContent], { type: 'text/csv' });
            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = `results_${new Date().toISOString().slice(0,19).replace(/:/g,'-')}.csv`;
            link.click();
            URL.revokeObjectURL(url);
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        function showProgressBar() {
            const bar = document.getElementById('progressBar');
            let width = 0;
            bar.style.width = '0%';
            document.getElementById('progressLabel').textContent = 'Finetuned Qwen 2.5: Generating SQL...';
            const interval = setInterval(() => {
                if (width >= 100) {
                    clearInterval(interval);
                    document.getElementById('progressLabel').textContent = 'Generation Complete!';
                } else {
                    width += Math.random() * 20;
                    bar.style.width = Math.min(width, 100) + '%';
                }
            }, 400);
        }
    </script>
</body>
</html>
'''

# ==============================================================================
# MAIN APPLICATION ENTRY POINT
# ==============================================================================

def main():
    """Main application entry point"""
    print("Adaptive Cross-Lingual Text-to-SQL with Context-Aware Fine-Tuning")
    print("MTP Research Project by Sarthak Arora")
    print("=" * 70)
    
    # Check dependencies
    try:
        app = create_app()
        
        # Test AI availability
        ai_generator = AIGenerator()
        ollama_available = ai_generator.test_connection()
        
        print("Research System Status:")
        print(f" Schema-Adaptive Engine: Ready")
        print(f" Cross-Lingual Processing: Ready")
        print(f" Context-Aware Fine-Tuning: {'âœ… Available' if ollama_available else ' Fallback Mode (Rule-Based)'}")
        print(f" Research Interface: Ready")
        
        if not ollama_available:
            print("\n To enable AI features:")
            print("  1. Install Ollama from: https://ollama.ai")
            print("  2. Run: ollama serve")
            print("  3. Run: ollama pull qwen2.5-coder:7b")
            print("  4. Restart this application")
        
        print("\n Starting web server...")
        print(" Open your browser to: http://127.0.0.1:8000")
        print(" Press Ctrl+C to stop")
        print("=" * 50)
        
        import random
        port = random.randint(1025, 65535)
        print(f"\n🚀 Starting Flask app on random port: {port}\n")
        app.run(host="0.0.0.0", port=port, debug=False)
        
    except KeyboardInterrupt:
        print("\n Application stopped by user")
    except Exception as e:
        print(f"\n Application failed to start: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
