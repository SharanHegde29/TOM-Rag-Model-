"""
TOM Bridge Server — Python bridge for the TOM frontend (Tom & Jerry-inspired AI companion).

It wraps the existing `Tom` class from tom_agent.py (TOM core) and exposes it
as a tiny HTTP JSON API for the React frontend.

Run:
    tom-env\\Scripts\\python.exe tom_server.py
    # serves on http://localhost:8000

Endpoints:
    GET  /api/health    -> { ok, service }
    GET  /api/status    -> { online, model, voice, voice_rate, voice_pitch, memories }
    GET  /api/history   -> { messages: [{role, content}] } (user+assistant only)
    POST /api/chat      -> { message } => { reply, memories, tool|null }
                          tool is {name, label, detail, duration_ms} when the
                          backend detects TOM is using a tool.
    POST /chat          -> alias of /api/chat (bare-path convenience)
    POST /api/upload    -> multipart 'file' => { ok, filename, bytes, chunks }
                          saves into tom_workspace for RAG search.
    POST /api/clear     -> clears chat (keeps system persona), saves file
    POST /api/speak     -> { text } => audio/mpeg (edge-tts, Tom-style voice)
    POST /api/speak-cues-> { text } => { audio (base64 mp3), mime, cues, voice }
                          `cues` are word-level timestamps from edge-tts
                          ({t, d, text} in milliseconds) that drive TOM's
                          real-time lip-sync on the frontend.

Zero new pip dependencies — stdlib only + existing tom_agent deps.
"""
import io
import json
import os
import re
import sys
import time
import asyncio
import base64
import hashlib
import struct
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse
from tom_agent import Tom

# NOTE: an earlier Flask stub (app.run(port=5000)) used to live here. It served
# only an incomplete /chat route and blocked forever at module level, so the real
# bridge below (127.0.0.1:8000) never started — the frontend then fell back to
# demo mode. The bridge below is the single server; /chat is an alias of /api/chat.

# Same console-safety guard as tom_agent (tom_server also logs emoji).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except Exception:
        pass

HOST = "127.0.0.1"
PORT = 8000

# Upload allowlist — must stay in sync with Tom.RAG_*_EXTS in tom_agent.py
UPLOAD_EXTS = {".txt", ".md", ".docx", ".pptx", ".pdf", ".csv", ".xlsx",
               ".png", ".jpg", ".jpeg", ".bmp", ".webp"}
UPLOAD_MAX_BYTES = 20 * 1024 * 1024


def _parse_multipart(body, content_type):
    """Minimal multipart/form-data parser (stdlib only, cgi is gone in 3.13).
    Returns [(filename, data_bytes)] for file parts. Never raises."""
    try:
        m = re.search(r"boundary=([^;]+)", content_type or "")
        if not m:
            return []
        boundary = m.group(1).strip().strip('"').encode("latin-1")
        parts = []
        for chunk in body.split(b"--" + boundary):
            if b"\r\n\r\n" not in chunk:
                continue
            head, data = chunk.split(b"\r\n\r\n", 1)
            if data.endswith(b"\r\n"):
                data = data[:-2]
            if not data or data == b"--":
                continue
            fm = re.search(br'filename="([^"]*)"', head)
            if not fm:
                continue
            filename = os.path.basename(fm.group(1).decode("utf-8", "ignore").strip())
            if filename:
                parts.append((filename, data))
        return parts
    except Exception:
        return []

# Single shared instance so memory persists across requests
tom = Tom()
_chat_lock = threading.RLock()  # one chat turn at a time: shared memory must not interleave

FRONTEND_ORIGINS = {
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
}


def _persona_text():
    for m in tom.memory:
        if m.get("role") == "system":
            return m.get("content", "")
    return ""


def _public_history():
    # Hide system prompt from the UI
    return [m for m in tom.memory if m.get("role") in ("user", "assistant")]


def _memory_count():
    return max(0, len(tom.memory) - 1)


# ---------------------------------------------------------------------------
# Shared speech synthesis (edge-tts). One helper for both /api/speak and
# /api/speak-cues so voice, rate and pitch always match. Word-level
# timestamps (WordBoundary events) are what drive TOM's lip-sync.
# ---------------------------------------------------------------------------
def _clean_tts_text(text):
    return str(text or "").replace("*", "").replace("#", "").strip()


def _synth_speech(text, with_cues=False):
    """Return (audio_bytes, cues). cues is [] unless with_cues=True.
    cues = [{"t": ms, "d": ms, "text": word}, ...] straight from edge-tts.
    Raises on failure — callers translate that into a 500."""
    import edge_tts

    async def _gen():
        # Fix: do the pronunciation swap into a NEW variable. The old code
        # reassigned `text` inside _gen, shadowing the parameter and making the
        # earlier `text[:2000]` read raise UnboundLocalError (every TTS call 500'd).
        spoken = re.sub(r"\bSharan\b", "Shurun", str(text or ""), flags=re.IGNORECASE)
        communicate = edge_tts.Communicate(
            spoken[:2000],
            getattr(tom, "voice_id", "en-GB-RyanNeural"),
            rate=getattr(tom, "voice_rate", "+0%"),
            pitch=getattr(tom, "voice_pitch", "+0Hz"),
            boundary="WordBoundary" if with_cues else "SentenceBoundary",
        )
        buf = io.BytesIO()
        cues = []
        async for chunk in communicate.stream():
            ctype = chunk.get("type")
            if ctype == "audio" and chunk.get("data"):
                buf.write(chunk["data"])
            elif ctype == "WordBoundary" and with_cues:
                cues.append({
                    "t": round(chunk.get("offset", 0) / 10000.0, 1),   # 100ns -> ms
                    "d": round(chunk.get("duration", 0) / 10000.0, 1),
                    "text": chunk.get("text", ""),
                })
        return buf.getvalue(), cues

    return asyncio.run(_gen())


# --- Tool-use signalling (additive only: core TOM behaviour untouched) ---
# Keyword heuristics that guess which "tool" TOM is reaching for,
# so the React chat feed can show a "Tool Execution" pill while waiting
# and a collapsed log once the final text arrives.
TOOL_RULES = [
    (r"\b(run|execute|exec|script|code|debug|compile|terminal|command)\b",
     "python_script", "executing a script", "Ran a helper script in the sandbox"),
    (r"\b(calculat|compute|math|equation|sum of|average|percent)\b|\d\s*[\+\-\*/]\s*\d",
     "calculator", "crunching numbers", "Evaluated a numeric expression"),
    (r"\b(search|find|look up|lookup|google|recall|remember|memory|memories)\b",
     "memory_search", "searching memory", "Searched conversation memory"),
    (r"\b(write|create|save|file|draft|email|letter|essay|poem|story)\b",
     "composer", "composing a document", "Composed structured text"),
]


def _detect_tool(message):
    """Return {name, label, detail} or None — never raises."""
    try:
        text = (message or "").lower()
        for pattern, name, label, detail in TOOL_RULES:
            if re.search(pattern, text):
                return {"name": name, "label": label, "detail": detail}
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Minimal stdlib WebSocket support (no new pip deps).
# Implements RFC 6455 server side just enough for TOM:
#   GET /ws  +  Upgrade: websocket  ->  101 + raw frames
#   client -> server : masked text frames (raw user string)
#   server -> client : unmasked JSON text frames with
#     {type: tool_start|tool_end|audio_start|audio_end|chat_reply|error}
# ---------------------------------------------------------------------------
_WS_MAGIC = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def _ws_accept_key(key):
    sha = hashlib.sha1((key.strip() + _WS_MAGIC).encode("utf-8")).digest()
    return base64.b64encode(sha).decode("utf-8")


def _ws_send_text(conn, text):
    """Send one unmasked text frame. Raises on disconnect."""
    if isinstance(text, str):
        payload = text.encode("utf-8")
    else:
        payload = bytes(text)
    header = bytearray()
    header.append(0x81)  # FIN + text opcode
    n = len(payload)
    if n < 126:
        header.append(n)
    elif n < 65536:
        header.append(126)
        header.extend(struct.pack("!H", n))
    else:
        header.append(127)
        header.extend(struct.pack("!Q", n))
    conn.sendall(bytes(header) + payload)


def _ws_send_json(conn, obj):
    _ws_send_text(conn, json.dumps(obj, ensure_ascii=False))


def _ws_recv_exact(conn, n):
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("ws closed")
        buf += chunk
    return buf


def _ws_recv_text(conn):
    """Return next client text message as str, or None on close.
    Handles ping/pong/close automatically. Supports masked frames."""
    while True:
        hdr = _ws_recv_exact(conn, 2)
        b1, b2 = hdr[0], hdr[1]
        opcode = b1 & 0x0F
        masked = (b2 & 0x80) != 0
        length = b2 & 0x7F
        if length == 126:
            length = struct.unpack("!H", _ws_recv_exact(conn, 2))[0]
        elif length == 127:
            length = struct.unpack("!Q", _ws_recv_exact(conn, 8))[0]
        mask = _ws_recv_exact(conn, 4) if masked else None
        payload = _ws_recv_exact(conn, length) if length else b""
        if masked and payload:
            payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        if opcode == 0x8:  # close
            try:
                # echo close
                conn.sendall(b"\x88\x00")
            except Exception:
                pass
            return None
        if opcode == 0x9:  # ping -> pong
            try:
                pong = bytearray([0x8A])
                if length < 126:
                    pong.append(length)
                elif length < 65536:
                    pong.append(126)
                    pong.extend(struct.pack("!H", length))
                else:
                    pong.append(127)
                    pong.extend(struct.pack("!Q", length))
                conn.sendall(bytes(pong) + payload)
            except Exception:
                return None
            continue
        if opcode == 0xA:  # pong
            continue
        if opcode in (0x1, 0x0):  # text / continuation (treat as text)
            try:
                return payload.decode("utf-8")
            except Exception:
                return payload.decode("utf-8", errors="ignore")
        # ignore binary / others
        continue


def _run_tom_chat(user_text):
    """Run the shared TOM LLM once. Returns reply str. Never raises."""
    with _chat_lock:
        return _run_tom_chat_locked(user_text)


def _run_tom_chat_locked(user_text):
    snap = len(tom.memory)
    try:
        stream = tom.chat(user_text)
        full = ""
        for chunk in stream:
            try:
                token = chunk.choices[0].delta.content
            except Exception:
                token = None
            if token:
                full += token
        if not full.strip():
            full = "Signal lost for a moment, Sharan. Try that again?"
        try:
            tom.memory.append({"role": "assistant", "content": full})
        except Exception:
            pass
        try:
            tom.save_memory()
        except Exception:
            pass
        return full
    except Exception as e:
        try:
            del tom.memory[snap:]  # drop user msg + any dangling tool_calls
        except Exception:
            pass
        return f"TOM core error: {e}"


def _schedule_audio_end(conn, seconds):
    """Send audio_end later so the visualizer doesn't flicker off before TTS starts."""
    def _fire():
        try:
            _ws_send_json(conn, {"type": "audio_end"})
        except Exception:
            pass
    t = threading.Timer(max(2.0, min(30.0, seconds)), _fire)
    t.daemon = True
    t.start()


class Handler(BaseHTTPRequestHandler):
    server_version = "TOM-Bridge/1.0"
    protocol_version = "HTTP/1.1"

    def _send_json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._send_cors()
        self.end_headers()
        self.wfile.write(body)

    def _send_cors(self):
        origin = self.headers.get("Origin", "")
        allow = origin if origin in FRONTEND_ORIGINS else "*"
        self.send_header("Access-Control-Allow-Origin", allow)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Content-Length", "0")
        self._send_cors()
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/ws", "/ws/"):
            return self._handle_ws()
        if path == "/api/health":
            return self._send_json({"ok": True, "service": "TOM bridge"})
        if path == "/api/status":
            return self._send_json({
                "online": True,
                "model": getattr(tom, "model", "openai/gpt-oss-20b"),
                "voice": getattr(tom, "voice_id", "en-GB-RyanNeural"),
                "voice_rate": getattr(tom, "voice_rate", "+0%"),
                "voice_pitch": getattr(tom, "voice_pitch", "+0Hz"),
                "memories": _memory_count(),
            })
        if path == "/api/history":
            return self._send_json({"messages": _public_history()})
        if path == "/":
            return self._send_json({"service": "TOM bridge. Use /api/* + /ws . Frontend runs on :5173."})
        return self._send_json({"error": "not found"}, 404)

    # -- WebSocket endpoint -------------------------------------------------
    def _handle_ws(self):
        upgrade = (self.headers.get("Upgrade") or "").lower()
        if upgrade != "websocket":
            return self._send_json({"error": "use ws://localhost:8000/ws with WebSocket upgrade"}, 400)
        key = self.headers.get("Sec-WebSocket-Key") or ""
        if not key:
            return self._send_json({"error": "missing Sec-WebSocket-Key"}, 400)
        accept = _ws_accept_key(key)
        # 101 Switching Protocols — raw socket from here on
        self.send_response(101, "Switching Protocols")
        self.send_header("Upgrade", "websocket")
        self.send_header("Connection", "Upgrade")
        self.send_header("Sec-WebSocket-Accept", accept)
        self.end_headers()
        conn = self.connection
        # Take over the socket; prevent BaseHTTPRequestHandler from reusing it
        self.close_connection = True
        try:
            conn.settimeout(300)
        except Exception:
            pass
        print("bridge: ws client connected (/ws)")
        try:
            while True:
                try:
                    msg = _ws_recv_text(conn)
                except Exception:
                    break
                if msg is None:  # client closed
                    break
                user_text = (msg or "").strip()
                if not user_text:
                    continue
                # 1) tool signalling
                tool = _detect_tool(user_text)
                t_start = time.time()
                if tool is not None:
                    try:
                        _ws_send_json(conn, {"type": "tool_start", "tool": tool["name"]})
                    except Exception:
                        break
                # 2) LLM reply (blocking this WS thread only)
                reply = _run_tom_chat(user_text)
                if tool is not None:
                    try:
                        _ws_send_json(conn, {
                            "type": "tool_end",
                            "tool": tool["name"],
                            "duration_ms": int((time.time() - t_start) * 1000),
                        })
                    except Exception:
                        break
                # 3) audio + final text. audio_start first so the
                #    visualizer is already live when chat_reply renders.
                #    audio_end is delayed so TTS (/api/speak) can take over.
                try:
                    _ws_send_json(conn, {"type": "audio_start"})
                    _ws_send_json(conn, {"type": "chat_reply", "text": reply})
                    _schedule_audio_end(conn, 3 + len(reply) / 20.0)
                except Exception:
                    break
        finally:
            print("bridge: ws client disconnected")
        return

    def _read_json_body(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8") or "{}")
        except Exception:
            return {}

    def do_POST(self):
        path = urlparse(self.path).path

        if path in ("/api/chat", "/chat"):
            data = self._read_json_body()
            message = str(data.get("message", "")).strip()
            if not message:
                return self._send_json({"error": "message is required"}, 400)
            _chat_lock.acquire()
            snap = len(tom.memory)
            try:
                tool = _detect_tool(message)
                t_start = time.time()
                stream = tom.chat(message)
                full_reply = ""
                for chunk in stream:
                    try:
                        token = chunk.choices[0].delta.content
                    except Exception:
                        token = None
                    if token:
                        full_reply += token
                if not full_reply.strip():
                    full_reply = "Signal lost for a moment, Sharan. Try that again?"
                tom.memory.append({"role": "assistant", "content": full_reply})
                try:
                    tom.save_memory()
                except Exception:
                    pass
                if tool is not None:
                    tool = {**tool, "duration_ms": int((time.time() - t_start) * 1000)}
                return self._send_json({"reply": full_reply, "memories": _memory_count(), "tool": tool})
            except Exception as e:
                # Roll back the optimistic user message so a failed call
                # doesn't pollute memory.
                try:
                    del tom.memory[snap:]  # drop user msg + any dangling tool_calls
                except Exception:
                    pass
                print(f"bridge: chat error: {e!r}")
                return self._send_json({"error": f"TOM core error: {e}"}, 500)
            finally:
                _chat_lock.release()

        if path == "/api/upload":
            # File attach: save into tom_workspace so RAG (search_local_data)
            # can find it on the very next chat turn. stdlib multipart only.
            ctype = self.headers.get("Content-Type", "")
            if "multipart/form-data" not in ctype:
                return self._send_json({"error": "use multipart/form-data with a 'file' field"}, 400)
            try:
                length = int(self.headers.get("Content-Length", 0) or 0)
            except ValueError:
                length = 0
            if length <= 0 or length > UPLOAD_MAX_BYTES + 1024 * 1024:
                return self._send_json({"error": "empty file or file too large (20MB max)"}, 400)
            try:
                body = self.rfile.read(length)
                parts = _parse_multipart(body, ctype)
                if not parts:
                    return self._send_json({"error": "no file found in upload"}, 400)
                filename, data = parts[0]
                ext = os.path.splitext(filename)[1].lower()
                if ext not in UPLOAD_EXTS:
                    return self._send_json({
                        "error": f"'{ext or '?'}' is not ingestible. Use: txt, md, docx, pptx, pdf, csv, xlsx, png, jpg."
                    }, 400)
                if len(data) > UPLOAD_MAX_BYTES:
                    return self._send_json({"error": "file too large (20MB max)"}, 400)
                work_dir = os.path.join(os.getcwd(), "tom_workspace")
                os.makedirs(work_dir, exist_ok=True)
                safe = os.path.basename(filename)
                dest = os.path.join(work_dir, safe)
                with open(dest, "wb") as f:
                    f.write(data)
                try:
                    chunks = len(tom._rag_extract_chunks(dest))
                except Exception:
                    chunks = 0
                print(f"bridge: uploaded {safe} ({len(data)} bytes, {chunks} chunks)")
                return self._send_json({"ok": True, "filename": safe, "bytes": len(data), "chunks": chunks})
            except Exception as e:
                return self._send_json({"error": f"upload failed: {e}"}, 500)

        if path == "/api/clear":
            persona = _persona_text() or "You are TOM."
            tom.memory = [{"role": "system", "content": persona}]
            try:
                tom.save_memory()
            except Exception:
                pass
            return self._send_json({"ok": True, "memories": 0})

        if path == "/api/speak":
            # Plain audio path (kept for compatibility): audio/mpeg, no cues.
            data = self._read_json_body()
            text = _clean_tts_text(data.get("text", ""))
            if not text:
                return self._send_json({"error": "text is required"}, 400)
            try:
                audio, _cues = _synth_speech(text, with_cues=False)
                if not audio:
                    return self._send_json({"error": "tts produced no audio"}, 500)
                self.send_response(200)
                self.send_header("Content-Type", "audio/mpeg")
                self.send_header("Content-Length", str(len(audio)))
                self._send_cors()
                self.end_headers()
                self.wfile.write(audio)
                return
            except Exception as e:
                return self._send_json({"error": f"tts error: {e}"}, 500)

        if path == "/api/speak-cues":
            # Lip-sync path: JSON carrying the exact same audio PLUS
            # word-level timestamps so the frontend can animate TOM's
            # mouth against the real playback timeline.
            data = self._read_json_body()
            text = _clean_tts_text(data.get("text", ""))
            if not text:
                return self._send_json({"error": "text is required"}, 400)
            try:
                audio, cues = _synth_speech(text, with_cues=True)
                if not audio:
                    return self._send_json({"error": "tts produced no audio"}, 500)
                return self._send_json({
                    "audio": base64.b64encode(audio).decode("ascii"),
                    "mime": "audio/mpeg",
                    "cues": cues,
                    "voice": getattr(tom, "voice_id", "en-GB-RyanNeural"),
                    "chars": len(text),
                })
            except Exception as e:
                return self._send_json({"error": f"tts error: {e}"}, 500)

        return self._send_json({"error": "not found"}, 404)

    def log_message(self, fmt, *args):  # quieter logs
        print("bridge:", fmt % args)

import os
import subprocess
import webbrowser
from threading import Timer

FRONTEND_URL = "http://localhost:5173"

def start_frontend_server():
    """Auto-detects the React/Vite folder and launches 'npm run dev' in the background."""
    base_dir = os.getcwd()
    # Check common frontend folder names (or root directory)
    candidates = ["frontend", "client", "ui", "web", "tom-frontend", "tom-frontend", "tom-ui", "tom-ui", "tom-ui", "."]
    
    frontend_dir = None
    for folder in candidates:
        check_path = os.path.join(base_dir, folder)
        if os.path.exists(os.path.join(check_path, "package.json")):
            frontend_dir = check_path
            break
            
    # Fallback: scan any immediate subfolder for a package.json
    if not frontend_dir:
        for item in os.listdir(base_dir):
            sub_path = os.path.join(base_dir, item)
            if os.path.isdir(sub_path) and os.path.exists(os.path.join(sub_path, "package.json")):
                frontend_dir = sub_path
                break

    if not frontend_dir:
        print("  [⚠️ Frontend Auto-Start: Could not find package.json. Start npm run dev manually.]")
        return None

    print(f"  [⚡ Auto-starting Vite Frontend in: {os.path.basename(os.path.abspath(frontend_dir))}]")
    try:
        # shell=True is required on Windows so subprocess can locate 'npm.cmd'
        return subprocess.Popen(
            "npm run dev",
            cwd=frontend_dir,
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    except Exception as e:
        print(f"  [⚠️ Failed to start Vite server: {e}]")
        return None


def open_in_chrome():
    print(f"  [🚀 Auto-launching TOM UI in Chrome: {FRONTEND_URL}]")
    chrome_paths = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe")
    ]
    for path in chrome_paths:
        if os.path.exists(path):
            subprocess.Popen([path, FRONTEND_URL])
            return
    webbrowser.open_new(FRONTEND_URL)


def main():
    # 1. Spin up Vite (npm run dev) in the background
    auto = os.environ.get("TOM_NO_AUTOSTART") != "1"
    vite_proc = start_frontend_server() if auto else None

    # 2. Boot the Python Backend Bridge
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    server.allow_reuse_address = True
    server.daemon_threads = True
    print("=" * 48)
    print("  TOM BRIDGE — ONLINE")
    print(f"  http://{HOST}:{PORT}  (frontend: {FRONTEND_URL})")
    print(f"  ws://{HOST}:{PORT}/ws  (live chat socket)")
    print(f"  [Memories Restored: {_memory_count()}]")
    print("=" * 48)
    
    # 3. Give Vite 2.5 seconds to compile before popping open Chrome
    if auto:
        Timer(2.5, open_in_chrome).start()
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nBridge shutting down. Memories already saved per-chat.")
    finally:
        # Cleanly terminate the background Node/Vite process on exit
        if vite_proc:
            subprocess.run(
                f"taskkill /F /T /PID {vite_proc.pid}",
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )


if __name__ == "__main__":
    main()