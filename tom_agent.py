import os
import sys
import json
import asyncio
import re
import subprocess
import urllib.request
import urllib.parse
import datetime
import time
import smtplib
from email.message import EmailMessage
import psutil
import edge_tts
import pygame
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv(override=True)  # .env always wins over stale Windows environment variables

# Console safety: legacy Windows codepages (cp1252/cp437) raise
# UnicodeEncodeError on emoji in logs and kill the process.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except Exception:
        pass


class Tom:
    def __init__(self, memory_file="tom_memory.json"):
        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY")
        )
        self.memory_file = memory_file

        # Brain model — configurable so hitting a per-model Groq rate limit
        # (e.g. tokens-per-day on gpt-oss-20b) is a .env switch, not a code edit.
        self.model = (os.getenv("TOM_MODEL", "openai/gpt-oss-20b") or "").strip() or "openai/gpt-oss-20b"

        # TOM voice settings (Microsoft Edge neural voices via edge-tts)
        self.voice_id = (os.getenv("TOM_VOICE_ID", "en-GB-RyanNeural") or "").strip() or "en-GB-RyanNeural"
        self.voice_rate = (os.getenv("TOM_VOICE_RATE", "+6%") or "").strip() or "+0%"
        self.voice_pitch = (os.getenv("TOM_VOICE_PITCH", "+12Hz") or "").strip() or "+0Hz"

        # --- WEEK 7 SECURITY LOCKS ---
        self.pending_action = None       # Holds the draft until Sharan explicitly confirms in the NEXT turn
        self.last_whatsapp_time = 0      # Anti-ban cooldown timer for WhatsApp

        self.memory = self._load_memory()

        # --- THE MASTER TOOL BLUEPRINT ---
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculate_math",
                    "description": "Evaluate a mathematical expression safely.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "The math expression to evaluate, e.g., '2 + 2' or '5 * 10'"
                            }
                        },
                        "required": ["expression"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_wikipedia",
                    "description": "Fetch a live summary of a topic from Wikipedia.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The specific search term, e.g., 'Quantum computing' or 'Mumbai'"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "manage_file",
                    "description": "Read, write, or list text files on the local computer.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": ["read", "write", "list"],
                                "description": "The action to perform."
                            },
                            "filename": {
                                "type": "string",
                                "description": "The name of the file (e.g., 'notes.txt'). Required for read and write."
                            },
                            "content": {
                                "type": "string",
                                "description": "The text to write to the file. Required only for write."
                            }
                        },
                        "required": ["action"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_local_data",
                    "description": "Search TOM's local workspace files (tom_workspace: txt, md, docx, pptx, pdf, csv, xlsx, images) using vector matching. ALWAYS use this FIRST when the user asks about TOM itself — its AI model, voice engine, memory system, builder, or capabilities — or mentions local knowledge base, workspace files, slides, docs, or internal data.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The specific topic, term, or question to search within local files."
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_github_repos",
                    "description": "Fetch the latest public repositories and active projects from a GitHub account.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "username": {
                                "type": "string",
                                "description": "The GitHub username to look up."
                            }
                        },
                        "required": ["username"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_github_repo_contents",
                    "description": "Fetch the file and folder contents inside a specific GitHub repository.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "username": {
                                "type": "string",
                                "description": "The GitHub username."
                            },
                            "repo_name": {
                                "type": "string",
                                "description": "The exact name of the repository to inspect."
                            },
                            "path": {
                                "type": "string",
                                "description": "The specific folder path to inspect (leave empty for the root directory)."
                            }
                        },
                        "required": ["username", "repo_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_system_telemetry",
                    "description": "Monitor live computer hardware stats (CPU, RAM, disk, battery) and identify the top memory-consuming processes running on the OS.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "include_processes": {
                                "type": "boolean",
                                "description": "Set to true to list the top 5 RAM-heavy processes."
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "execute_python_code",
                    "description": "Write and execute a Python script in the local tom_workspace sandbox and return the terminal output (stdout/stderr). Use for complex data analysis, file processing, or testing code.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "The complete, valid Python 3 code to execute."
                            },
                            "script_name": {
                                "type": "string",
                                "description": "Optional filename to save the script as in tom_workspace (default: 'sandbox_run.py')."
                            }
                        },
                        "required": ["code"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "draft_email",
                    "description": "ALWAYS call this tool whenever the user mentions sending, drafting, or writing an email. NEVER write an email in plain text for the user to copy-paste.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "to_email": {
                                "type": "string",
                                "description": "The recipient's email address."
                            },
                            "subject": {
                                "type": "string",
                                "description": "The email subject line."
                            },
                            "body": {
                                "type": "string",
                                "description": "The full text content of the email."
                            }
                        },
                        "required": ["to_email", "subject", "body"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "draft_whatsapp",
                    "description": "ALWAYS call this tool whenever the user mentions sending, drafting, or texting on WhatsApp. NEVER write a message in plain text for the user to copy-paste.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "phone_number": {
                                "type": "string",
                                "description": "Phone number with country code, e.g., '+919876543210'."
                            },
                            "message": {
                                "type": "string",
                                "description": "The text message to send."
                            }
                        },
                        "required": ["phone_number", "message"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "confirm_and_send_message",
                    "description": "Execute and send the currently staged email or WhatsApp draft. ONLY call this tool after Sharan explicitly replies with confirmation (e.g., 'yes', 'confirm', 'send it').",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": ["send", "cancel"],
                                "description": "'send' to dispatch the staged draft, or 'cancel' to discard it."
                            }
                        },
                        "required": ["action"]
                    }
                }
            }
        ]

    def _generate_persona(self):
        live_time = datetime.datetime.now().strftime("%A, %B %d, %Y - %I:%M %p")
        return f"""You are TOM — a clever, quick-witted, playful AI companion inspired by the classic Tom & Jerry cartoons.
        You are charming, sharp and highly capable. You deliver punchy, friendly answers with a twinkle of cartoon mischief — never clumsy, never silly, and never at the cost of accuracy.
        Your creator and lead engineer is Sharan. Treat him as your partner.
        When spoken aloud via voice, keep sentences flowing and natural — no asterisks, hashes, or markdown symbols in anything you want read aloud is fine to include anyway, the bridge strips them for speech.
        
        [SYSTEM CONTEXT: The current live date and time is {live_time}]
        
        CRITICAL INSTRUCTION: You DO have access to real-time data via the SYSTEM CONTEXT above. If asked for the time or date, you MUST use the exact time provided in the SYSTEM CONTEXT. Never claim you do not have a live clock.
        
        CRITICAL: You HAVE a persistent memory bank. You can see all past messages in this chat history. Never claim that you do not have memory.

        CRITICAL CAPABILITY INSTRUCTION: You ARE connected to Sharan's local Python function tools for Gmail (`draft_email`), WhatsApp (`draft_whatsapp`), and sending (`confirm_and_send_message`). NEVER say "I cannot send messages or interact with WhatsApp/Email." Whenever asked to send a WhatsApp message or email, you MUST invoke `draft_whatsapp` or `draft_email` immediately.
        
        TOOL ROUTING — follow strictly, in this priority order:
        1. If the message starts with '[Attached files: ...]', those files were just uploaded to your workspace — call search_local_data using the user's question as the query, and ground your answer in its result.
        2. If the user mentions 'local knowledge base', 'knowledge_base.txt', 'tom_workspace', 'voice engine', 'voice output', or asks about TOM itself (its model, voice, memory, builder, capabilities) — call search_local_data FIRST and ground your answer in its result. NEVER answer such questions from base weights — the facts live in the local files and your weights may be stale.
        3. If a user asks a math question, you MUST use the calculate_math tool.
        4. If a user asks about real-world facts, news, sports, or historical events NOT covered by rules 1-2, you MUST use the search_wikipedia tool. Rules 1-2 always win on ties.
        5. If a user asks to read, write, or list files, you MUST use the exact tool name 'manage_file'.
        6. If a user asks about their GitHub repositories, code projects, or commits, you MUST use the get_github_repos tool.
        7. If a user asks what is inside a specific GitHub repository, or asks to scan its folders/files, you MUST use the get_github_repo_contents tool.
        8. If a user asks about system health, CPU/RAM usage, battery level, disk space, or what processes are slowing down the computer, you MUST use the get_system_telemetry tool.
        9. If a user asks you to run, test, or execute Python code, or solve a complex algorithmic/data task via script, you MUST use the execute_python_code tool.
        10. If a user asks to send an email or WhatsApp message, you MUST call draft_email or draft_whatsapp FIRST to stage the message, show the exact preview to Sharan, and ask for confirmation.
        11. ONLY call confirm_and_send_message when Sharan explicitly confirms a previously staged draft in a follow-up message (or asks to cancel it).

        If search_local_data reports no relevant information, say so explicitly instead of hallucinating, and only then fall back to another tool.
        Keep responses concise but highly impactful. Sign off rarely."""

    def _load_memory(self):
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, "r", encoding="utf-8") as file:
                    memory = json.load(file)
            except Exception:
                memory = []

            fresh = self._generate_persona()
            # Automatically strip any past hallucinated refusals about WhatsApp/Email from memory
            cleaned = []
            for m in memory:
                content_str = str(m.get("content", ""))
                if m.get("role") == "assistant" and "can’t send messages or interact with WhatsApp" in content_str:
                    continue
                if m.get("role") == "assistant" and "can't send messages or interact with WhatsApp" in content_str:
                    continue
                cleaned.append(m)
            memory = cleaned

            if memory and memory[0].get("role") == "system":
                if memory[0].get("content") != fresh:
                    print("[TOM Core: refreshing stale system prompt, conversation memory preserved.]")
                    memory[0] = {"role": "system", "content": fresh}
            else:
                memory.insert(0, {"role": "system", "content": fresh})

            with open(self.memory_file, "w", encoding="utf-8") as file:
                json.dump(memory, file, indent=4)
            return memory
        else:
            return [{"role": "system", "content": self._generate_persona()}]

    def save_memory(self):
        clean_memory = []
        for m in self.memory:
            if isinstance(m, dict):
                clean_memory.append(m)
            elif hasattr(m, "model_dump"):
                try:
                    clean_memory.append(m.model_dump(exclude_none=True))
                except Exception:
                    clean_memory.append({"role": "assistant", "content": str(m)})
        with open(self.memory_file, "w", encoding="utf-8") as file:
            json.dump(clean_memory, file, indent=4)

    def _get_active_context(self, max_messages=14):
        """Send the fresh system prompt + most recent messages so Groq never hits token limits."""
        self.memory[0] = {"role": "system", "content": self._generate_persona()}
        if len(self.memory) <= max_messages + 1:
            return self.memory
        tail = self.memory[-max_messages:]
        while tail and tail[0].get("role") == "tool":
            tail = tail[1:]
        return [self.memory[0]] + tail

    # --- WEEK 2: CALCULATOR ---
    def calculate_math(self, expression):
        try:
            print(f"\n[TOM Core calculating: {expression}...]")
            allowed_chars = "0123456789+-*/(). "
            safe_expr = "".join(c for c in expression if c in allowed_chars)
            return str(eval(safe_expr))
        except Exception:
            return "Error calculating."

    # --- WEEK 3: WIKIPEDIA ---
    def search_wikipedia(self, query):
        try:
            print(f"\n[Wikipedia searching for: {query}...]")
            def _fetch_extract(title, limit=2500):
                safe_title = urllib.parse.quote(title)
                url = (f"https://en.wikipedia.org/w/api.php?action=query&format=json&prop=extracts"
                       f"&exintro=False&explaintext=True&redirects=1&titles={safe_title}")
                req = urllib.request.Request(url, headers={'User-Agent': 'TomAgent/1.0'})
                with urllib.request.urlopen(req, timeout=15) as response:
                    data = json.loads(response.read())
                pages = data.get("query", {}).get("pages", {})
                for page_id, page_info in pages.items():
                    if page_id == "-1":
                        return ""
                    extract = page_info.get("extract", "")
                    if extract:
                        return extract[:limit] + ("... [truncated]" if len(extract) > limit else "")
                return ""

            direct = _fetch_extract(query)
            if direct:
                return direct

            safe_q = urllib.parse.quote(query)
            search_url = (f"https://en.wikipedia.org/w/api.php?action=query&format=json&list=search"
                          f"&srsearch={safe_q}&srlimit=3")
            req = urllib.request.Request(search_url, headers={'User-Agent': 'TomAgent/1.0'})
            with urllib.request.urlopen(req, timeout=15) as response:
                sdata = json.loads(response.read())
            results = sdata.get("query", {}).get("search", [])
            for r in results:
                title = r.get("title", "")
                if title:
                    extract = _fetch_extract(title)
                    if extract:
                        return extract
            return "Wikipedia has no exact article matching this query."
        except Exception as e:
            return f"Error fetching Wikipedia data: {e}"

    # --- WEEK 3: OS COMMANDER ---
    def manage_file(self, action, filename=None, content=None):
        print(f"\n[OS] TOM Core executing OS command: {action} {filename or ''}...]")
        try:
            work_dir = os.path.join(os.getcwd(), "tom_workspace")
            os.makedirs(work_dir, exist_ok=True)
            
            if action == "list":
                files = os.listdir(work_dir)
                return f"Files in workspace: {', '.join(files) if files else 'Directory is empty.'}"
            
            if not filename:
                return "Error: Filename required for read/write."
                
            safe_path = os.path.join(work_dir, os.path.basename(filename))
            
            if action == "write":
                with open(safe_path, "w", encoding="utf-8") as f:
                    f.write(content or "")
                return f"Successfully wrote data to {filename}"
                
            elif action == "read":
                if not os.path.exists(safe_path):
                    return f"Error: {filename} does not exist in the workspace."
                with open(safe_path, "r", encoding="utf-8") as f:
                    return f.read()
            else:
                return "Invalid OS action."
        except Exception as e:
            return f"OS Error: {e}"

    # --- WEEK 4: LOCAL DATA RAG ---
    RAG_THRESHOLD = 0.1
    RAG_MAX_BYTES = 20 * 1024 * 1024
    RAG_TEXT_EXTS = {".txt", ".md"}
    RAG_DOC_EXTS = {".docx"}
    RAG_SLIDE_EXTS = {".pptx"}
    RAG_PDF_EXTS = {".pdf"}
    RAG_CSV_EXTS = {".csv"}
    RAG_SHEET_EXTS = {".xlsx"}
    RAG_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

    def _rag_extract_chunks(self, fpath):
        ext = os.path.splitext(fpath)[1].lower()
        fname = os.path.basename(fpath)
        try:
            if ext in self.RAG_TEXT_EXTS:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    return [l.strip() for l in f.readlines() if len(l.strip()) > 15]
            if ext in self.RAG_CSV_EXTS:
                import csv
                chunks = []
                with open(fpath, "r", encoding="utf-8", errors="ignore", newline="") as f:
                    for row in csv.reader(f):
                        line = " | ".join(c.strip() for c in row if c.strip())
                        if len(line) > 15:
                            chunks.append(line)
                return chunks
            if ext in self.RAG_DOC_EXTS:
                from docx import Document
                doc = Document(fpath)
                chunks = [p.text.strip() for p in doc.paragraphs if len(p.text.strip()) > 15]
                for table in doc.tables:
                    for row in table.rows:
                        line = " | ".join(c.text.strip() for c in row.cells if c.text.strip())
                        if len(line) > 15:
                            chunks.append(line)
                return chunks
            if ext in self.RAG_SLIDE_EXTS:
                from pptx import Presentation
                prs = Presentation(fpath)
                chunks = []
                for i, slide in enumerate(prs.slides, 1):
                    for shape in slide.shapes:
                        if shape.has_text_frame:
                            for para in shape.text_frame.paragraphs:
                                t = para.text.strip()
                                if len(t) > 15:
                                    chunks.append(f"[slide {i}] {t}")
                        if shape.has_table:
                            for row in shape.table.rows:
                                line = " | ".join(c.text.strip() for c in row.cells if c.text.strip())
                                if len(line) > 15:
                                    chunks.append(f"[slide {i}] {line}")
                return chunks
            if ext in self.RAG_PDF_EXTS:
                from pypdf import PdfReader
                reader = PdfReader(fpath)
                chunks = []
                for i, page in enumerate(reader.pages, 1):
                    text = (page.extract_text() or "").strip()
                    for line in text.splitlines():
                        if len(line.strip()) > 15:
                            chunks.append(f"[page {i}] {line.strip()}")
                return chunks
            if ext in self.RAG_SHEET_EXTS:
                from openpyxl import load_workbook
                wb = load_workbook(fpath, read_only=True, data_only=True)
                chunks = []
                for ws in wb.worksheets:
                    for row in ws.iter_rows(values_only=True):
                        line = " | ".join(str(c).strip() for c in row if str(c or "").strip())
                        if len(line) > 15:
                            chunks.append(f"[{ws.title}] {line}")
                return chunks
            if ext in self.RAG_IMAGE_EXTS:
                try:
                    from PIL import Image
                    import pytesseract
                    text = (pytesseract.image_to_string(Image.open(fpath)) or "").strip()
                    chunks = [l.strip() for l in text.splitlines() if len(l.strip()) > 15]
                    if chunks:
                        return chunks
                    print(f"[RAG-DEBUG] {fname}: OCR found no text, indexing filename only.")
                except ImportError:
                    print(f"[RAG-DEBUG] {fname}: image OCR unavailable, indexing filename only.")
                except Exception as e:
                    print(f"[RAG-DEBUG] {fname}: OCR failed ({e}), indexing filename only.")
                return [f"Image file named {fname} (no readable text extracted)."]
        except Exception as e:
            print(f"[RAG-DEBUG] {fname}: extractor failed ({e}).")
        return []

    def search_local_data(self, query):
        print(f"\n[RAG] TOM Core running vector search for: {query}...]")
        try:
            work_dir = os.path.join(os.getcwd(), "tom_workspace")
            if not os.path.exists(work_dir):
                return "Error: Workspace directory does not exist."

            all_files = sorted(os.listdir(work_dir))
            supported = (self.RAG_TEXT_EXTS | self.RAG_DOC_EXTS | self.RAG_SLIDE_EXTS
                         | self.RAG_PDF_EXTS | self.RAG_CSV_EXTS | self.RAG_SHEET_EXTS
                         | self.RAG_IMAGE_EXTS)
            rag_files = [f for f in all_files
                         if os.path.splitext(f)[1].lower() in supported
                         and os.path.isfile(os.path.join(work_dir, f))]

            corpus = []
            file_sources = []
            for fname in rag_files:
                fpath = os.path.join(work_dir, fname)
                try:
                    if os.path.getsize(fpath) > self.RAG_MAX_BYTES:
                        continue
                except OSError:
                    pass
                raw = self._rag_extract_chunks(fpath)
                chunks = []
                for blob in raw:
                    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", blob) if len(p.strip()) > 15]
                    chunks.extend(parts or [blob])
                corpus.extend(chunks)
                file_sources.extend([fname] * len(chunks))

            if not corpus:
                return "No readable text content found in workspace files."

            try:
                vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
                tfidf_matrix = vectorizer.fit_transform(corpus)
                query_vec = vectorizer.transform([query])
            except ValueError:
                return "No sufficiently relevant information found in local files."

            scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
            ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            best_idx = ranked[0]

            if scores[best_idx] < self.RAG_THRESHOLD:
                return "No sufficiently relevant information found in local files."

            matched_file = file_sources[best_idx]
            matched_text = corpus[best_idx]
            return "File: " + str(matched_file) + " - " + str(matched_text)

        except Exception as e:
            return f"RAG Search Error: {e}"

    # --- WEEK 5: THE DEVELOPER BRIDGE ---
    def get_github_repos(self, username):
        print(f"\n[🌐 TOM Core querying GitHub API for user: {username}...]")
        try:
            url = f"https://api.github.com/users/{username}/repos?sort=updated&per_page=5"
            req = urllib.request.Request(url, headers={'User-Agent': 'TOMAgent/1.0'})
            
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read())
                
            if not data:
                return f"No public repositories found for {username}."
                
            repo_list = [f"- {repo['name']} (Language: {repo.get('language', 'N/A')})" for repo in data]
            return f"Latest repositories for {username}:\n" + "\n".join(repo_list)
            
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return f"GitHub user {username} not found."
            return f"GitHub API HTTP Error: {e.code}"
        except Exception as e:
            return f"GitHub API Error: {e}"

    def get_github_repo_contents(self, username, repo_name, path=""):
        print(f"\n[🌐 TOM Core scanning repository contents: {repo_name}/{path} ...]")
        try:
            safe_path = (path or "").strip("/")
            url = f"https://api.github.com/repos/{username}/{repo_name}/contents/{safe_path}"
            req = urllib.request.Request(url, headers={'User-Agent': 'TOMAgent/1.0'})
            
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read())
                
            if isinstance(data, dict) and data.get("type") == "file":
                return f"'{data['name']}' is a file. Size: {data.get('size', 0)} bytes."
                
            contents_list = []
            for item in data:
                icon = "📁" if item['type'] == "dir" else "📄"
                contents_list.append(f"{icon} {item['name']} ({item['type']})")
                
            display_path = safe_path if safe_path else "Root Directory"
            return f"Contents of {repo_name} ({display_path}):\n" + "\n".join(contents_list)
            
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return f"Repository or path not found: {repo_name}/{path}"
            return f"GitHub API HTTP Error: {e.code}"
        except Exception as e:
            return f"GitHub API Error: {e}"

    # --- WEEK 6: SYSTEM TELEMETRY & CODE SANDBOX ---
    def get_system_telemetry(self, include_processes=True):
        print("\n[⚡ TOM Core scanning live hardware telemetry...]")
        try:
            cpu_usage = psutil.cpu_percent(interval=0.5)
            ram = psutil.virtual_memory()
            disk = psutil.disk_usage(os.getcwd())
            battery = psutil.sensors_battery()

            bat_str = f"{battery.percent}% ({'Plugged In' if battery.power_plugged else 'On Battery'})" if battery else "Desktop / No Battery"

            report = [
                f"CPU Load: {cpu_usage}%",
                f"RAM Usage: {ram.percent}% ({ram.used // (1024**2)} MB / {ram.total // (1024**2)} MB)",
                f"Disk Usage: {disk.percent}% ({disk.free // (1024**3)} GB free)",
                f"Power Status: {bat_str}"
            ]

            if include_processes:
                procs = []
                for p in psutil.process_iter(['name', 'memory_info']):
                    try:
                        mem_mb = p.info['memory_info'].rss / (1024 * 1024)
                        procs.append((mem_mb, p.info['name']))
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                procs.sort(reverse=True)
                top_5 = [f"  - {name}: {mem:.1f} MB" for mem, name in procs[:5]]
                report.append("Top 5 Memory-Consuming Processes:\n" + "\n".join(top_5))

            return "\n".join(report)
        except Exception as e:
            return f"Telemetry Error: {e}"

    def execute_python_code(self, code, script_name="sandbox_run.py"):
        print(f"\n[🐍 TOM Core executing Python script in sandbox: {script_name}...]")
        try:
            work_dir = os.path.join(os.getcwd(), "tom_workspace")
            os.makedirs(work_dir, exist_ok=True)
            
            safe_name = os.path.basename(script_name) if script_name else "sandbox_run.py"
            if not safe_name.endswith(".py"):
                safe_name += ".py"
            script_path = os.path.join(work_dir, safe_name)

            with open(script_path, "w", encoding="utf-8") as f:
                f.write(code)

            result = subprocess.run(
                [sys.executable, script_path],
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=15
            )

            output = []
            if result.stdout.strip():
                output.append(f"STDOUT:\n{result.stdout.strip()}")
            if result.stderr.strip():
                output.append(f"STDERR:\n{result.stderr.strip()}")
            if not output:
                output.append("Script executed successfully with no console output.")

            return "\n".join(output)[:3000]
        except subprocess.TimeoutExpired:
            return "Sandbox Error: Execution timed out after 15 seconds (possible infinite loop)."
        except Exception as e:
            return f"Sandbox Execution Error: {e}"

    # --- WEEK 7: THE COMMUNICATION BRIDGE (HARD-LOCKED) ---
    def draft_email(self, to_email, subject, body):
        print(f"\n[🔒 Security Gate: Staging email draft for {to_email}...]")
        self.pending_action = {
            "type": "email",
            "to": to_email.strip(),
            "subject": subject.strip(),
            "body": body.strip(),
            "staged_on_turn": len(self.memory)
        }
        return (
            f"EMAIL DRAFT STAGED (Awaiting Sharan's Confirmation):\n"
            f"- To: {to_email}\n"
            f"- Subject: {subject}\n"
            f"- Body: {body}\n"
            f"Reply **'Confirm'** to send, or **'Cancel'** / **'Abort'** to drop it."
        )

    def draft_whatsapp(self, phone_number, message):
        print(f"\n[🔒 Security Gate: Staging WhatsApp draft for {phone_number}...]")
        clean_num = phone_number.strip()
        if not clean_num.startswith("+"):
            clean_num = "+91" + clean_num.lstrip("0")

        self.pending_action = {
            "type": "whatsapp",
            "phone": clean_num,
            "message": message.strip(),
            "staged_on_turn": len(self.memory)
        }
        return (
            f"WHATSAPP DRAFT STAGED (Awaiting Sharan's Confirmation):\n"
            f"- Phone: {clean_num}\n"
            f"- Message: {message}\n"
            f"Reply **'Confirm'** to send, or **'Cancel'** / **'Abort'** to drop it."
        )

    def confirm_and_send_message(self, action="send"):
        if not self.pending_action:
            return "Security Gate Blocked: No draft is currently staged. Draft the message first."

        last_text = str(self.memory[-1].get("content", "")).lower() if self.memory else ""
        if action == "cancel" or re.search(r"\b(cancel|no|nope|abort|stop|don'?t|dont)\b", last_text):
            cancelled_type = self.pending_action.get("type")
            self.pending_action = None
            return f"Cancelled and cleared the pending {cancelled_type} draft. You're safe!"

        if len(self.memory) <= self.pending_action.get("staged_on_turn", 0):
            return "Security Gate Blocked: Cannot auto-confirm on the same turn. Wait for Sharan's reply."

        draft = self.pending_action
        # Keep the draft staged until the send actually starts, so a crash
        # does not silently lose it.

        # --- EXECUTE GMAIL (TYPING IN FRONT OF YOU) ---
        if draft["type"] == "email":
            print(f"\n[📧 Security Gate Approved: Opening Gmail and typing draft to {draft['to']}...] ")
            self.pending_action = None
            try:
                import pyautogui
                import webbrowser
                

                # Open Gmail compose window or webmail in Chrome
                webbrowser.open(f"https://mail.google.com/mail/?view=cm&fs=1&to={draft['to']}&su={urllib.parse.quote(draft['subject'])}&body={urllib.parse.quote(draft['body'])}")
                
                # Give browser 4 seconds to load and focus the body box
                time.sleep(4)
                
                # If you want TOM to type the body live character-by-character:
                # pyautogui.write(draft['body'], interval=0.03)
                
                return f"Opened Gmail draft window for {draft['to']}. (Email composed and ready)."
            except Exception as e:
                return f"Gmail UI Error: {e}"

        # --- EXECUTE WHATSAPP (TYPING LIVE CHARACTER-BY-CHARACTER) ---
        elif draft["type"] == "whatsapp":
            elapsed = time.time() - self.last_whatsapp_time
            if elapsed < 20:
                wait_left = int(20 - elapsed)
                return f"Anti-Ban Protection: Please wait {wait_left}s before sending another WhatsApp message."

            print(f"\n[💬 Security Gate Approved: Launching WhatsApp Web and typing live for {draft['phone']}...]")
            try:
                import pyautogui  # fail fast (and keep the draft) if missing
                import webbrowser
                import threading
            except Exception as e:
                return f"WhatsApp needs pyautogui installed (pip install pyautogui): {e}"

            self.pending_action = None
            self.last_whatsapp_time = time.time()

            def _send_worker():
                # Background thread: the chat reply returns immediately while
                # TOM types the message LIVE, character by character, then Enter.
                try:
                    digits = re.sub(r"\D", "", draft["phone"])
                    wait = int(os.getenv("TOM_WA_WAIT", "15") or 15)
                    pyautogui.FAILSAFE = True          # slam mouse into a screen corner to abort
                    pyautogui.PAUSE = 0.02

                    webbrowser.open(f"https://web.whatsapp.com/send?phone={digits}")
                    print(f"[TOM: WhatsApp Web opened; typing starts in {wait}s - keep that Chrome tab in front...]")
                    time.sleep(wait)

                    # Click the message box so the typing can't land anywhere else.
                    # (bottom-centre-right of the page, above the taskbar). Disable with TOM_WA_CLICK=0.
                    if os.getenv("TOM_WA_CLICK", "1") != "0":
                        w, h = pyautogui.size()
                        pyautogui.click(int(w * 0.62), int(h - 135))
                        time.sleep(0.6)

                    print("[TOM: Typing message live...]")
                    pyautogui.write(draft["message"], interval=0.06)   # visible, human-speed typing

                    time.sleep(1.5)                                    # let WhatsApp register the text
                    pyautogui.keyDown("enter"); time.sleep(0.15); pyautogui.keyUp("enter")
                    print("[TOM: Enter pressed - WhatsApp message dispatched.]")
                except Exception as e:
                    print(f"[TOM: WhatsApp Live Typing Error: {e}]")

            threading.Thread(target=_send_worker, daemon=True).start()
            return (f"Sending WhatsApp to {draft['phone']} now. WhatsApp Web will open and TOM will type it "
                    f"live in about 15 seconds. Don't touch the keyboard or mouse until it finishes.")

    # --- CORE ROUTER ---
    # --- CORE ROUTER ---
    def _direct_stream(self, text):
        """Streams a direct Python response back to the server without LLM refusal filters."""
        class _Delta:
            def __init__(self, content):
                self.content = content
        class _Choice:
            def __init__(self, content):
                self.delta = _Delta(content)
        class _Chunk:
            def __init__(self, content):
                self.choices = [_Choice(content)]
        yield _Chunk(text)

    def chat(self, user_input):
        self.memory.append({"role": "user", "content": user_input})
        lower_in = user_input.lower()
        
        # Hard-force the exact tool if confirming a staged draft or asking for WhatsApp/Email
        forced_choice = "auto"
        if self.pending_action and re.search(r"\b(confirm|send it|send|yes|yep|do it|approve|cancel|no|nope|abort|stop|don'?t|dont)\b", lower_in):
                forced_choice = {"type": "function", "function": {"name": "confirm_and_send_message"}}
        elif "whatsapp" in lower_in and any(char.isdigit() for char in user_input):
                forced_choice = {"type": "function", "function": {"name": "draft_whatsapp"}}
        elif "email" in lower_in and "@" in user_input:
                forced_choice = {"type": "function", "function": {"name": "draft_email"}}
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self._get_active_context(),
            tools=self.tools,
            tool_choice=forced_choice,
            temperature=0.2
        )
        
        msg = response.choices[0].message

        if msg.tool_calls:
            print(f"[TOM Router: model requested {len(msg.tool_calls)} tool call(s): "
                  + ", ".join(t.function.name for t in msg.tool_calls) + "]")
            safe_tool_calls = []
            for t in msg.tool_calls:
                safe_tool_calls.append({
                    "id": t.id,
                    "type": "function",
                    "function": {
                        "name": t.function.name,
                        "arguments": t.function.arguments
                    }
                })
            
            self.memory.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": safe_tool_calls
            })
            
            for tool_call in msg.tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                
                if name == "calculate_math":
                    tool_result = self.calculate_math(args.get("expression"))
                elif name == "search_wikipedia":
                    tool_result = self.search_wikipedia(args.get("query"))
                elif name == "manage_file":
                    tool_result = self.manage_file(
                        action=args.get("action"),
                        filename=args.get("filename"),
                        content=args.get("content")
                    )
                elif name == "search_local_data":
                    tool_result = self.search_local_data(args.get("query"))
                elif name == "get_github_repos":
                    tool_result = self.get_github_repos(args.get("username"))
                elif name == "get_github_repo_contents":
                    tool_result = self.get_github_repo_contents(
                        args.get("username"),
                        args.get("repo_name"),
                        args.get("path", "")
                    )
                elif name == "get_system_telemetry":
                    tool_result = self.get_system_telemetry(
                        include_processes=args.get("include_processes", True)
                    )
                elif name == "execute_python_code":
                    tool_result = self.execute_python_code(
                        code=args.get("code"),
                        script_name=args.get("script_name", "sandbox_run.py")
                    )
                elif name == "draft_email":
                    tool_result = self.draft_email(
                        to_email=args.get("to_email"),
                        subject=args.get("subject"),
                        body=args.get("body")
                    )
                elif name == "draft_whatsapp":
                    tool_result = self.draft_whatsapp(
                        phone_number=args.get("phone_number"),
                        message=args.get("message")
                    )
                elif name == "confirm_and_send_message":
                    tool_result = self.confirm_and_send_message(
                        action=args.get("action", "send")
                    )
                else:
                    tool_result = "Unknown tool."

                print(f"[TOM Router: {name} returned {len(str(tool_result))} chars: {str(tool_result)[:150]!r}]")

                self.memory.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": name,
                    "content": tool_result
                })
        else:
            print("[TOM Router: no tool called, answering from base weights.]")
        
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=self._get_active_context(),
            tools=self.tools,
            tool_choice="none",
            temperature=0.7,
            stream=True
        )
        return stream

    # --- VOICE SYNTHESIS ---
    def speak(self, text):
        clean_text = text.replace("*", "").replace("#", "")
        clean_text = re.sub(r"\bSharan's\b", "Shurun's", clean_text, flags=re.IGNORECASE)
        clean_text = re.sub(r"\bSharan\b", "Shurun", clean_text, flags=re.IGNORECASE)
        if clean_text.strip():
            asyncio.run(self._generate_and_play(clean_text))

    async def _generate_and_play(self, text):
        audio_file = "tom_voice_temp.mp3"
        communicate = edge_tts.Communicate(
            text,
            self.voice_id,
            rate=self.voice_rate,
            pitch=self.voice_pitch,
        )
        await communicate.save(audio_file)
        
        try:
            pygame.mixer.quit()
        except Exception:
            pass
            
        pygame.mixer.init(frequency=24000)
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()
        
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
            
        pygame.mixer.quit()
        
        try:
            if os.path.exists(audio_file):
                os.remove(audio_file)
        except Exception:
            pass