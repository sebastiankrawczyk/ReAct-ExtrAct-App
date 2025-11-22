import os
import shutil
import io
import json
import time
import threading
import subprocess
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Set
#s
import streamlit as st
import sys
import traceback
import csv
import re
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass
# Bridge Streamlit Community Cloud secrets → environment variables
def _apply_streamlit_secrets_to_env() -> None:
    try:
        secrets = getattr(st, 'secrets', None)
        if not secrets:
            return
        keys = [
            'OPENROUTER_API_KEY','OPENAI_API_KEY','LLAMA_CLOUD_API_KEY',
            'ALLOW_WRITE_KEYS_FILE','EVALUATION','RAGAS','G_EVAL','CLEAR_STORAGE','COHERE_RERANK',
            'INPUT_PATH','OUTPUT_PATH','STORAGE_PATH',
        ]
        for k in keys:
            try:
                v = secrets.get(k)
            except Exception:
                v = None
            if v is not None and (not os.environ.get(k)):
                os.environ[k] = str(v)
    except Exception:
        pass

_apply_streamlit_secrets_to_env()
def _ensure_dependencies() -> None:
    try:
        import importlib
        missing = []
        for pkg, mod in [
            ("llama-index", "llama_index"),
            ("llama-index-core", "llama_index.core"),
            ("llama-parse", "llama_parse"),
            ("streamlit", "streamlit"),
            ("pandas", "pandas"),
            ("llama-index-llms-ollama", "llama_index.llms.ollama"),
            ("llama-index-embeddings-ollama", "llama_index.embeddings.ollama"),
        ]:
            try:
                importlib.import_module(mod)
            except Exception:
                missing.append(pkg)
        if missing:
            import subprocess, sys
            cmd = [sys.executable, "-m", "pip", "install", "-q"] + missing
            subprocess.run(cmd, check=False)
    except Exception:
        pass

_ensure_dependencies()

# Ensure project root is on sys.path before importing local modules (Streamlit Cloud)
try:
    _prj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if _prj_root not in sys.path:
        sys.path.insert(0, _prj_root)
except Exception:
    pass

from utils.VectorQueryEngineCreator import VectorQueryEngineCreator
from llama_index.core import Settings
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
INPUT_DIR = os.path.join(REPO_ROOT, 'input')  # default user input
INPUT_DIR_DEMO = os.path.join(REPO_ROOT, 'input_demo')  # separate demo input
OUTPUT_DIR = os.path.join(REPO_ROOT, 'output')
CONFIG_DIR = os.path.join(REPO_ROOT, 'config')
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


# ----------------------------- Helpers -----------------------------

def _ensure_dirs() -> None:
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(INPUT_DIR_DEMO, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def _strip_think(text: str) -> str:
    try:
        if not isinstance(text, str) or not text:
            return text
        import re as _re
        return _re.sub(r"(?is)<\s*think\s*>[\s\S]*?<\s*/\s*think\s*>", "", text).strip()
    except Exception:
        return text


def _clean_field_text(text: str) -> str:
    try:
        if not isinstance(text, str):
            return ''
        text = _strip_think(text)
        text = text.replace('\n', ' ').replace('\t', ' ')
        import re as _re
        text = _re.sub(r"\s+", " ", text).strip()
        return text
    except Exception:
        return (text or '').strip()


def _clear_inspector_state() -> None:
    try:
        st.session_state.pop('inspector_open', None)
        st.session_state.pop('inspector_paper', None)
        st.session_state.pop('inspector_topic', None)
    except Exception:
        pass


# Default in-memory API keys. Prefer environment variables; no hardcoded secrets.
# Debug flag: block inspector rendering on New Extraction page only
DEBUG_BLOCK_NEW_INSPECTOR = False
DEFAULT_API_KEYS: Dict[str, str] = {
    'OPENROUTER_API_KEY': os.getenv('OPENROUTER_API_KEY', ''),
    'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY', ''),
    'LLAMA_CLOUD_API_KEY': os.getenv('LLAMA_CLOUD_API_KEY', ''),
}


def _apply_keys_to_env(keys: Dict[str, str]) -> None:
    try:
        for k, v in (keys or {}).items():
            if isinstance(k, str) and isinstance(v, str) and v:
                os.environ[k] = v
    except Exception:
        pass


def _ensure_default_keys() -> None:
    try:
        if 'api_keys' not in st.session_state:
            st.session_state['api_keys'] = dict(DEFAULT_API_KEYS)
        _apply_keys_to_env(st.session_state['api_keys'])
    except Exception:
        pass


def _read_json(path: str) -> Any:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def _list_run_dirs() -> List[str]:
    try:
        runs = [d for d in os.listdir(OUTPUT_DIR) if os.path.isdir(os.path.join(OUTPUT_DIR, d))]
        runs.sort(reverse=True)
        return runs
    except Exception:
        return []


def _discover_results(run_dir_name: str) -> Dict[str, Dict[str, Any]]:
    """
    returns mapping: file_stem -> results_json (contents of <file>/<file>_result.json)
    Prefers <file>_baseline_like.json when present for a uniform schema.
    """
    out: Dict[str, Dict[str, Any]] = {}
    base = os.path.join(OUTPUT_DIR, run_dir_name)
    if not os.path.isdir(base):
        return out
    for d in os.listdir(base):
        p = os.path.join(base, d)
        if not os.path.isdir(p):
            continue
        # Prefer baseline_like.json
        bl = os.path.join(p, f"{d}_baseline_like.json")
        if os.path.isfile(bl):
            data = _read_json(bl) or {}
            out[d] = {"results": data}
            continue
        # Fallback to _result.json
        rj = os.path.join(p, f"{d}_result.json")
        if os.path.isfile(rj):
            data = _read_json(rj) or {}
            # If this is a meetings-style payload, convert extracted_data -> results list expected by UI
            try:
                payload = data.get('extracted_data') if isinstance(data, dict) else None
                if isinstance(payload, dict):
                    results_list = []
                    for topic, ent in payload.items():
                        ent = ent or {}
                        best_ctx = ent.get('best_context') or []
                        if isinstance(best_ctx, dict):
                            best_ctx = [best_ctx]
                        if not best_ctx:
                            ev = ent.get('evidence') or []
                            best_ctx = [{
                                'context': (e.get('text') or ''),
                                'score': e.get('score'),
                                'page': e.get('page'),
                                'section': e.get('section'),
                            } for e in ev[:5]]
                        results_list.append({
                            'query': {'topic': topic, 'possible_options': 'None'},
                            'question': topic,
                            'answer': ent.get('answer', ''),
                            'answer_concise': ent.get('concise_answer', ''),
                            'code': '',
                            'best_context': best_ctx[:5],
                        })
                    out[d] = {'results': results_list}
                else:
                    out[d] = data
            except Exception:
                out[d] = data
    return out


# --- Validation persistence (per run, per paper, per topic) ---
def _validation_path(run_dir_name: str) -> str:
    return os.path.join(OUTPUT_DIR, run_dir_name, 'validation.json')

def _read_validation(run_dir_name: Optional[str]) -> Dict[str, Dict[str, bool]]:
    if not run_dir_name:
        return {}
    try:
        p = _validation_path(run_dir_name)
        if os.path.isfile(p):
            with open(p, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data  # {paper: {topic: bool}}
    except Exception:
        pass
    return {}

def _write_validation(run_dir_name: Optional[str], data: Dict[str, Dict[str, bool]]) -> None:
    if not run_dir_name:
        return
    try:
        out_dir = os.path.join(OUTPUT_DIR, run_dir_name)
        os.makedirs(out_dir, exist_ok=True)
        p = _validation_path(run_dir_name)
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _ensure_llm_ready() -> None:
    try:
        if getattr(Settings, 'llm', None) is not None and getattr(Settings, 'embed_model', None) is not None:
            return
    except Exception:
        pass
    try:
        from config.config import EXECUTION_MODEL, EMBEDDING_MODEL, API, EMBEDDING_API, OLLAMA_BASE_URL, OLLAMA_EXECUTION_MODEL, OLLAMA_EMBEDDING_MODEL
        keys = _read_api_keys()
        if API == 'openrouter':
            Settings.llm = OpenAILike(
                model=EXECUTION_MODEL,
                api_base="https://openrouter.ai/api/v1",
                api_key=keys.get('OPENROUTER_API_KEY',''),
                is_chat_model=True,
            )
        elif API == 'ollama':
            exec_model = (os.getenv('OLLAMA_EXECUTION_MODEL') or OLLAMA_EXECUTION_MODEL or EXECUTION_MODEL)
            base_url = os.getenv('OLLAMA_BASE_URL') or OLLAMA_BASE_URL
            Settings.llm = Ollama(model=exec_model, base_url=base_url)
        else:
            raise ValueError("Unsupported API. Choose 'openrouter' or 'ollama'.")
        if EMBEDDING_API == 'openai':
            Settings.embed_model = OpenAIEmbedding(
                model=EMBEDDING_MODEL,
                api_base="https://api.openai.com/v1",
                api_key=keys.get('OPENAI_API_KEY',''),
            )
        elif EMBEDDING_API == 'ollama':
            emb_model = (os.getenv('OLLAMA_EMBEDDING_MODEL') or OLLAMA_EMBEDDING_MODEL or EMBEDDING_MODEL)
            base_url = os.getenv('OLLAMA_BASE_URL') or OLLAMA_BASE_URL
            Settings.embed_model = OllamaEmbedding(model_name=emb_model, base_url=base_url)
        else:
            raise ValueError("Unsupported EMBEDDING_API. Choose 'openai' or 'ollama'.")
    except Exception:
        pass


def _run_followup_query(paper: str, question: str, mode_choice: str = 'Quick') -> Dict[str, Any]:
    out = {"answer": "", "contexts": []}
    if not paper or not question:
        return out
    _ensure_llm_ready()
    try:
        from config.config import STORAGE_PATH, COHERE_RERANK, EMBEDDING_MODEL
        keys = _read_api_keys()
        # Prefer active input dir if present during the session
        active_inp = getattr(st.session_state.get('run_state', object()), 'active_input_dir', None) or INPUT_DIR
        vqc = VectorQueryEngineCreator(
            llama_parse_api_key=keys.get('LLAMA_CLOUD_API_KEY',''),
            cohere_api_key=keys.get('COHERE_API_KEY',''),
            input_path=active_inp,
            storage_path=STORAGE_PATH,
            cohere_rerank=COHERE_RERANK,
            embedding_model_name=EMBEDDING_MODEL,
            enable_section_reasoner=False,
            response_mode='compact',
        )
        # Optional: force rebuild by clearing persisted index for this paper
        try:
            if bool(st.session_state.get('rebuild_indexes')) or os.getenv('CLEAR_STORAGE') == '1':
                persist_dir = os.path.join(STORAGE_PATH, f"{paper}_vector_index")
                if os.path.isdir(persist_dir):
                    shutil.rmtree(persist_dir, ignore_errors=True)
        except Exception:
            pass
        qe = vqc.get_query_engine(paper)
        if mode_choice == 'Iterative (2 steps)':
            from llama_index.core import Settings as _S
            def _expand(_topic: str, prev: str) -> str:
                try:
                    prompt = (
                        "Return a short clause (<=10 words) to refine the query. "
                        "If none, reply STOP.\n"
                        f"Topic: {_topic}\nCurrent answer: {prev}\n"
                    )
                    raw = f"{_S.llm.complete(prompt)!s}".strip()
                    if not raw or raw.lower().startswith('stop'):
                        return ''
                    return raw.splitlines()[0].strip()
                except Exception:
                    return ''
            base_q = question.strip()
            resp1 = qe.query(base_q)
            ans1 = f"{resp1!s}".strip()
            clause = _expand(base_q, ans1)
            final_q = (base_q + (" " + clause if clause else '')).strip()
            resp2 = qe.query(final_q)
            out["answer"] = f"{resp2!s}".strip() or ans1
            nodes = getattr(resp2, 'source_nodes', None) or getattr(resp1, 'source_nodes', [])
        else:
            resp = qe.query(question)
            out["answer"] = f"{resp!s}".strip()
            nodes = getattr(resp, 'source_nodes', [])
        bctx = []
        for n in nodes[:5]:
            try:
                content = (n.node.get_content() or '').strip()
            except Exception:
                try:
                    content = str(getattr(n, 'get_content', lambda: '')()).strip()
                except Exception:
                    content = ''
            # attach page/section metadata when available so UI can display page even if marker is missing
            try:
                meta = getattr(n, 'node', None)
                meta = getattr(meta, 'metadata', None) or {}
            except Exception:
                meta = {}
            page_val = meta.get('page_label') if isinstance(meta, dict) else None
            if page_val is None and isinstance(meta, dict):
                page_val = meta.get('page')
            section_val = meta.get('section') if isinstance(meta, dict) else None
            bctx.append({
                "context": content,
                "score": getattr(n, 'score', None),
                "page": page_val,
                "section": section_val,
            })
        out["contexts"] = bctx
    except Exception:
        pass
    return out


def _write_queries_py(queries: List[Dict[str, Any]]) -> None:
    path = os.path.join(CONFIG_DIR, 'queries.py')
    # Write as a Python assignment using JSON to ensure proper escaping
    try:
        safe_list: List[Dict[str, str]] = []
        for q in (queries or []):
            safe_list.append({
                "topic": str((q or {}).get('topic', '') or ''),
                "possible_options": str((q or {}).get('possible_options', 'None') or ''),
            })
        payload = json.dumps(safe_list, ensure_ascii=False, indent=2)
        with open(path, 'w', encoding='utf-8') as f:
            f.write('QUERIES = ' + payload + '\n')
    except Exception:
        # Fallback: write an empty list to avoid breaking imports
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write('QUERIES = []\n')
        except Exception:
            pass


# UI draft queries (kept local to the app folder; does not affect config)
UI_DRAFT_PATH = os.path.join(os.path.dirname(__file__), 'queries_ui.json')

def _read_ui_draft() -> List[Dict[str, Any]]:
    try:
        if os.path.isfile(UI_DRAFT_PATH):
            with open(UI_DRAFT_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return list(data) if isinstance(data, list) else []
    except Exception:
        pass
    return []

def _write_ui_draft(queries: List[Dict[str, Any]]) -> None:
    try:
        with open(UI_DRAFT_PATH, 'w', encoding='utf-8') as f:
            json.dump(list(queries or []), f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _read_current_queries() -> List[Dict[str, Any]]:
    path = os.path.join(CONFIG_DIR, 'queries.py')
    try:
        # naive parse by exec in empty globals
        namespace: Dict[str, Any] = {}
        with open(path, 'r', encoding='utf-8') as f:
            code = f.read()
        exec(compile(code, path, 'exec'), namespace, namespace)
        return list(namespace.get('QUERIES') or [])
    except Exception:
        return []


def _pick_main_script(mode: str) -> str:
    if mode == 'baseline':
        return os.path.join(REPO_ROOT, 'backend', 'main_baseline.py')
    if mode == 'iter_retgen':
        return os.path.join(REPO_ROOT, 'backend', 'main_iter_retgen.py')
    # react_group removed
    # New canonical modes
    if mode == 'meetings':
        return os.path.join(REPO_ROOT, 'backend', 'main_react_extract.py')
    if mode == 'react_extract':
        return os.path.join(REPO_ROOT, 'backend', 'main_react_extract.py')
    # Backward-compatible aliases
    if mode == 'react_meetings':
        return os.path.join(REPO_ROOT, 'backend', 'main_react_extract.py')
    if mode == 'iter_react_extract':
        return os.path.join(REPO_ROOT, 'backend', 'main_react_extract.py')
    return os.path.join(REPO_ROOT, 'backend', 'main_baseline.py')

# --- API keys management ---

def _read_api_keys() -> Dict[str, str]:
    # Prefer environment, then in-session, then defaults. Optionally fill from local file.
    data: Dict[str, str] = {}
    wanted = ['OPENROUTER_API_KEY','OPENAI_API_KEY','LLAMA_CLOUD_API_KEY','OLLAMA_BASE_URL']
    for k in wanted:
        v = os.getenv(k) or (st.session_state.get('api_keys', {}).get(k) if 'api_keys' in st.session_state else None) or DEFAULT_API_KEYS.get(k, '')
        data[k] = v if isinstance(v, str) else ''
    # Optional local file fallback if still missing
    try:
        keys_path = os.path.join(CONFIG_DIR, 'config_keys.py')
        if os.path.isfile(keys_path):
            namespace: Dict[str, Any] = {}
            with open(keys_path, 'r', encoding='utf-8') as f:
                code = f.read()
            exec(compile(code, keys_path, 'exec'), namespace, namespace)
            for k in wanted:
                if not data.get(k) and isinstance(namespace.get(k), str):
                    data[k] = namespace.get(k)  # type: ignore
    except Exception:
        pass
    return data


def _write_api_keys(keys: Dict[str, str]) -> None:
    # Update in-memory and env only; avoid persisting secrets by default
    try:
        st.session_state['api_keys'] = {
            'OPENROUTER_API_KEY': keys.get('OPENROUTER_API_KEY',''),
            'OPENAI_API_KEY': keys.get('OPENAI_API_KEY',''),
            'LLAMA_CLOUD_API_KEY': keys.get('LLAMA_CLOUD_API_KEY',''),
        }
    except Exception:
        pass
    _apply_keys_to_env(_read_api_keys())
    if os.getenv('ALLOW_WRITE_KEYS_FILE') == '1':
        try:
            path = os.path.join(CONFIG_DIR, 'config_keys.py')
            os.makedirs(CONFIG_DIR, exist_ok=True)
            lines = [
                f"OPENROUTER_API_KEY = '{_read_api_keys().get('OPENROUTER_API_KEY','')}'\n",
                f"OPENAI_API_KEY = '{_read_api_keys().get('OPENAI_API_KEY','')}'\n",
                f"LLAMA_CLOUD_API_KEY = '{_read_api_keys().get('LLAMA_CLOUD_API_KEY','')}'\n",
            ]
            with open(path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
        except Exception:
            pass


def _keys_ready(keys: Dict[str, str]) -> bool:
    # Require Llama Cloud parse always
    if not keys.get('LLAMA_CLOUD_API_KEY'):
        return False
    try:
        from config.config import API, EMBEDDING_API
        api = (API or 'openrouter').strip().lower()
        eapi = (EMBEDDING_API or 'openai').strip().lower()
    except Exception:
        api, eapi = 'openrouter', 'openai'
    if api == 'openrouter' and not keys.get('OPENROUTER_API_KEY'):
        return False
    if eapi == 'openai' and not keys.get('OPENAI_API_KEY'):
        return False
    return True


# ----------------------------- Progress & Run -----------------------------

@dataclass
class RunState:
    process: Optional[subprocess.Popen] = None
    stdout: str = ''
    stderr: str = ''
    running: bool = False
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    run_dir_name: Optional[str] = None
    progress_pct: int = 0
    progress_stage: str = ''
    exit_code: Optional[int] = None
    script_path: str = ''
    child_pid: Optional[int] = None
    # optional friendly name for this run
    run_name: Optional[str] = None
    # progress tracking
    files_total: int = 0
    queries_total: int = 0
    per_file_max_q: Dict[str, int] = None  # lazy init as {}
    files_seen: Set[str] = None  # lazy init as set()
    # step-based progress
    steps_total: int = 0
    steps_done: int = 0
    step_seen: Set[str] = None  # e.g., {'load:paperA', 'parse:paperB', 'build:paperB', 'persist:paperB', 'eval'}
    query_seen: Dict[str, Set[int]] = None  # file -> set(query_index)
    parse_pending_files: Set[str] = None  # files that have seen parse but not yet persisted
    # staged totals
    idx_steps_total: int = 0
    query_steps_total: int = 0
    eval_steps_total: int = 0
    # active input directory for this run (overrides default INPUT_DIR)
    active_input_dir: Optional[str] = None


def _detect_new_run_dir(before: List[str]) -> Optional[str]:
    after = _list_run_dirs()
    for d in after:
        if d not in before:
            return d
    return None


def _read_run_name(run_dir_name: str) -> Optional[str]:
    try:
        p = os.path.join(OUTPUT_DIR, run_dir_name, 'run_name.txt')
        if os.path.isfile(p):
            with open(p, 'r', encoding='utf-8') as f:
                name = f.read().strip()
                return name or None
    except Exception:
        pass
    return None


def _write_run_name(run_dir_name: str, name: str) -> None:
    try:
        if not name:
            return
        out_dir = os.path.join(OUTPUT_DIR, run_dir_name)
        os.makedirs(out_dir, exist_ok=True)
        p = os.path.join(out_dir, 'run_name.txt')
        with open(p, 'w', encoding='utf-8') as f:
            f.write(name.strip())
    except Exception:
        pass


def _extract_run_timestamp(run_dir_name: str) -> Optional[str]:
    try:
        import re as _re
        m = _re.match(r"^(\d{4}\.\d{2}\.\d{2}_\d{2}\.\d{2}\.\d{2})", run_dir_name)
        if m:
            return m.group(1)
    except Exception:
        pass
    return None


def _format_run_display(run_dir_name: str) -> str:
    nm = _read_run_name(run_dir_name)
    ts = _extract_run_timestamp(run_dir_name)
    if nm and ts:
        return f"{nm} ({ts})"
    if nm:
        return nm
    return run_dir_name


def _slugify_name(name: str) -> str:
    try:
        s = (name or '').strip().lower()
        s = re.sub(r"[^a-z0-9\-\_\.\s]", "", s)
        s = re.sub(r"\s+", "_", s)
        s = re.sub(r"_+", "_", s)
        return s.strip("_-")[:60] or "project"
    except Exception:
        return "project"


def _rename_run_dir_if_needed(current_dir: Optional[str], friendly_name: Optional[str]) -> Optional[str]:
    """If a project name is provided, rename the run directory to include it as suffix.
    Format: <timestamp_mode>__<slug>. Returns final directory name (or current_dir on failure)."""
    try:
        if not current_dir or not friendly_name:
            return current_dir
        slug = _slugify_name(friendly_name)
        if not slug:
            return current_dir
        base_cur = os.path.join(OUTPUT_DIR, current_dir)
        if not os.path.isdir(base_cur):
            return current_dir
        # Already suffixed
        if current_dir.endswith(f"__{slug}"):
            return current_dir
        target_dir_name = f"{current_dir}__{slug}"
        target_path = os.path.join(OUTPUT_DIR, target_dir_name)
        # Handle collisions
        if os.path.exists(target_path):
            idx = 2
            while True:
                cand = f"{target_dir_name}-{idx}"
                cand_path = os.path.join(OUTPUT_DIR, cand)
                if not os.path.exists(cand_path):
                    target_dir_name = cand
                    target_path = cand_path
                    break
                idx += 1
        os.rename(base_cur, target_path)
        return target_dir_name
    except Exception:
        return current_dir


def _estimate_progress_from_log(buf: str, state: RunState) -> Tuple[int, str]:
    """
    Parse recent lines to estimate progress across files and queries.
    - Tracks max seen Query[i/N] per file
    - Uses QUERIES count as N and INPUT_DIR pdf count as total files (or files seen)
    Returns (percent, stage_text)
    """
    try:
        if state.per_file_max_q is None:
            state.per_file_max_q = {}
        if state.files_seen is None:
            state.files_seen = set()
        if state.step_seen is None:
            state.step_seen = set()
        if state.query_seen is None:
            state.query_seen = {}
        if state.parse_pending_files is None:
            state.parse_pending_files = set()
        lines = buf.splitlines()[-200:]
        raw_marker = ''
        # detect totals
        if state.queries_total <= 0:
            try:
                state.queries_total = max(1, len(_read_current_queries()))
            except Exception:
                state.queries_total = 5
        if state.files_total <= 0:
            try:
                state.files_total = len([f for f in os.listdir(INPUT_DIR) if f.lower().endswith('.pdf')])
            except Exception:
                state.files_total = 1
        import re
        for ln in lines:
            s = ln.strip()
            # remember the last meaningful marker line
            if s.startswith('[parse]') or s.startswith('[md]') or s.startswith('[index]') or s.startswith('[engine]') or 'RAGAS evaluation' in s or s.startswith('Evaluating:'):
                raw_marker = s
            # file started
            if s.startswith('Processing file:'):
                m = re.search(r'^Processing file:\s*(.+)$', s)
                if m:
                    file_key = m.group(1).strip()
                    state.files_seen.add(file_key)
            # index/engine stage
            if s.startswith('[index]') or s.startswith('[engine]'):
                # try to extract file key also
                m = re.search(r'^\[(index|engine)\]\s*(.+?):', s)
                if m:
                    state.files_seen.add(m.group(2).strip())
            # step: parse/build/persist/load detection
            # parse started
            mps = re.search(r'^\[parse\]\s*(.+?):', s)
            if mps:
                fkey = mps.group(1).strip()
                # normalize possible .pdf suffix
                fkey_norm = re.sub(r"\.pdf$", "", fkey, flags=re.IGNORECASE)
                state.parse_pending_files.add(fkey_norm)
                state.step_seen.add(f"parse:{fkey_norm}")
            # building VectorStoreIndex
            if '[index] building VectorStoreIndex' in s:
                # attribute to any pending file if available
                fkeyp = next(iter(state.parse_pending_files), None)
                if fkeyp:
                    state.step_seen.add(f"build:{fkeyp}")
            # persisted index written
            mper = re.search(r'^\[index\]\s*(.+?):\s*persisted index', s)
            if mper:
                fkey = mper.group(1).strip()
                state.step_seen.add(f"persist:{fkey}")
                if fkey in state.parse_pending_files:
                    try:
                        state.parse_pending_files.remove(fkey)
                    except Exception:
                        pass
            # loading existing index
            mload = re.search(r'^\[index\]\s*(.+?):\s*loading existing index', s)
            if mload:
                fkey = mload.group(1).strip()
                state.step_seen.add(f"load:{fkey}")
            # query lines (baseline/react format)
            m1 = re.search(r'^-?\s*\[\[(.+?)\]\]\s*Query\s*\[(\d+)/(\d+)\]', s)
            if m1:
                file_key = m1.group(1).strip()
                i = int(m1.group(2))
                N = int(m1.group(3))
                state.per_file_max_q[file_key] = max(i, state.per_file_max_q.get(file_key, 0))
                if N > state.queries_total:
                    state.queries_total = N
                raw_marker = 'QUERY'
                # mark query step done once per (file, i)
                qset = state.query_seen.setdefault(file_key, set())
                qset.add(i)
                continue
            # generic format: "Query [i/N]" (fallback to any file)
            m2 = re.search(r'Query\s*\[(\d+)/(\d+)\]', s)
            if m2:
                i = int(m2.group(1))
                N = int(m2.group(2))
                ph = '(current)'
                state.per_file_max_q[ph] = max(i, state.per_file_max_q.get(ph, 0))
                if N > state.queries_total:
                    state.queries_total = N
                raw_marker = 'QUERY'
        # compute step-based percent (and fallback percent)
        # dynamically plan totals based on current knowledge; never decrease total
        try:
            active_inp = state.active_input_dir or INPUT_DIR
            files = [f for f in os.listdir(active_inp) if f.lower().endswith('.pdf')]
        except Exception:
            files = list(state.files_seen) or []
        planned_files = list(state.files_seen) or [os.path.splitext(f)[0] for f in files]
        if not planned_files:
            planned_files = ["(current)"]
        per_file_steps = 0
        for fkey in planned_files:
            if f"load:{fkey}" in state.step_seen:
                per_file_steps += 1
            elif fkey in (state.parse_pending_files or set()) or f"parse:{fkey}" in state.step_seen or f"build:{fkey}" in state.step_seen or f"persist:{fkey}" in state.step_seen:
                per_file_steps += 3
            else:
                per_file_steps += 1
        planned_idx = per_file_steps
        planned_queries = max(1, state.queries_total) * max(1, len(planned_files))
        planned_eval = 0
        planned_total = max(1, planned_idx + planned_queries + planned_eval)
        # never shrink the planned total to avoid backward jumps
        state.idx_steps_total = max(state.idx_steps_total or 0, planned_idx)
        state.query_steps_total = max(state.query_steps_total or 0, planned_queries)
        state.eval_steps_total = 0
        state.steps_total = max(state.steps_total or 0, planned_total)
        # compute done steps from markers
        build_steps_done = len([1 for sname in state.step_seen if sname.startswith('load:') or sname.startswith('parse:') or sname.startswith('build:') or sname.startswith('persist:')])
        query_steps_done = sum(len(v) for v in state.query_seen.values())
        eval_done = 0
        state.steps_done = build_steps_done + query_steps_done + eval_done
        pct_calc = int(round(state.steps_done / max(1, state.steps_total) * 100))
        # Keep a visible minimum during a run so the bar doesn't appear blank
        visible_min = 3
        # Cap step-driven percent to avoid premature 99%
        max_running_cap = 95
        step_pct = (min(max_running_cap, max(visible_min, pct_calc)) if state.running else min(100, max(visible_min, pct_calc)))
        # fallback percent from query-only heuristic (downweighted)
        total_files = max(state.files_total, len(state.files_seen) or 1)
        done_units = sum(max_q for max_q in state.per_file_max_q.values())
        total_units = max(1, state.queries_total * total_files)
        fallback_raw = int(max(1, min(95, round(done_units / total_units * 100)))) if done_units > 0 else (1 if state.running else 0)
        # Blend gently; never exceed step_pct + 10 while running
        blended = min(step_pct + 10, max(step_pct, int(0.7 * step_pct + 0.3 * fallback_raw))) if state.running else max(step_pct, fallback_raw)
        pct = blended
        # friendly stage text mapping
        stage = ''
        # Stage label from high-level steps
        if state.steps_done == 0 and state.running:
            # early signals
            if any('Processing file:' in l for l in lines[-200:]):
                stage = 'Preparing documents'
            elif any(l.startswith('[parse]') for l in lines[-200:]):
                stage = 'Parsing documents'
            else:
                stage = 'Initializing...'
        else:
            # Prefer index stages
            if any(s.startswith('parse:') for s in state.step_seen) and any(s.startswith('build:') for s in state.step_seen) and (len([x for x in state.step_seen if x.startswith('persist:')]) < len([x for x in state.step_seen if x.startswith('parse:')])):
                stage = 'Building vector index'
            elif any(s.startswith('parse:') for s in state.step_seen):
                stage = 'Parsing documents'
            elif any(s.startswith('load:') for s in state.step_seen) and build_steps_done < state.idx_steps_total:
                stage = 'Loading vector index'
            elif any(l.strip().startswith('[plan]') for l in lines[-200:]) and query_steps_done == 0:
                stage = 'Planning'
            elif query_steps_done < state.query_steps_total:
                stage = 'Querying your documents'
            elif eval_done < state.eval_steps_total:
                stage = ''
            else:
                stage = 'Finalizing'
        # fallback informative counts
        if not stage:
            stage = f"Files {len(state.files_seen)}/{total_files} | Queries {done_units}/{total_units}"
        return pct, stage
    except Exception:
        return 0, ''


def _run_script(mode: str, state: RunState, eval_off: bool = False, clear_storage: bool = False) -> None:
    try:
        state.running = True
        state.started_at = time.time()
        # init progress totals
        try:
            state.queries_total = max(1, len(_read_current_queries()))
        except Exception:
            state.queries_total = 5
        try:
            active_inp = state.active_input_dir or INPUT_DIR
            state.files_total = len([f for f in os.listdir(active_inp) if f.lower().endswith('.pdf')])
        except Exception:
            state.files_total = 1
        state.per_file_max_q = {}
        state.files_seen = set()
        state.steps_total = 0
        state.steps_done = 0
        state.step_seen = set()
        state.query_seen = {}
        state.parse_pending_files = set()
        before_runs = _list_run_dirs()
        script = _pick_main_script(mode)
        state.script_path = script
        # start with empty buffer (suppress debug preamble)
        buf: List[str] = []
        state.stdout = ''

        env_vars = dict(os.environ)
        env_vars['PYTHONUNBUFFERED'] = '1'
        # Propagate parsing options (GROBID) and storage layout
        try:
            # Ensure USE_GROBID and GROBID_URL are forwarded to the child
            if os.getenv('USE_GROBID') is not None:
                env_vars['USE_GROBID'] = '1' if str(os.getenv('USE_GROBID')).strip().lower() in ('1','true','yes','y','on') else '0'
            if os.getenv('GROBID_URL'):
                env_vars['GROBID_URL'] = os.getenv('GROBID_URL')
            # Keep STORAGE_PATH consistent with parser/provider
            def _as_bool(v: str) -> bool:
                return str(v).strip().lower() in ('1','true','yes','y','on')
            api_now = (env_vars.get('API') or os.getenv('API') or 'openrouter').strip().lower()
            parser_now = 'grobid' if _as_bool(env_vars.get('USE_GROBID') or os.getenv('USE_GROBID') or '0') else 'llamaparse'
            env_vars['STORAGE_PATH'] = os.path.join(REPO_ROOT, 'storage', parser_now, api_now)
        except Exception:
            pass
        try:
            pp = env_vars.get('PYTHONPATH', '')
            sep = os.pathsep
            if not pp:
                env_vars['PYTHONPATH'] = REPO_ROOT
            elif REPO_ROOT not in pp.split(sep):
                env_vars['PYTHONPATH'] = REPO_ROOT + sep + pp
        except Exception:
            env_vars['PYTHONPATH'] = REPO_ROOT
        # Ensure storage path matches selected provider to avoid mixing indexes
        try:
            api_now = (env_vars.get('API') or os.getenv('API') or 'openrouter').strip().lower()
            if not env_vars.get('STORAGE_PATH'):
                env_vars['STORAGE_PATH'] = os.path.join(REPO_ROOT, 'storage', api_now)
        except Exception:
            pass
        # Evaluation disabled in this demo
        env_vars['EVALUATION'] = '0'
        env_vars['RAGAS'] = '0'
        env_vars['G_EVAL'] = '0'
        # Rebuild vector indexes by clearing storage at start
        env_vars['CLEAR_STORAGE'] = '1' if clear_storage else '0'
        proc = subprocess.Popen(
            [sys.executable, '-u', script],
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            text=True,
            universal_newlines=True,
            env=env_vars,
        )
        state.process = proc
        state.child_pid = proc.pid
        # Set early stage so the UI shows activity before first log line
        if not state.progress_stage:
            state.progress_stage = 'Initializing libraries...'
        last_line_ts = time.time()
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                buf.append(line)
                state.stdout = ''.join(buf)[-12000:]
                pct, stage = _estimate_progress_from_log(state.stdout, state)
                state.progress_pct = pct
                if stage:
                    state.progress_stage = stage
                last_line_ts = time.time()
                if not state.running:
                    break
                # If no output for 12s, show init heartbeat
                if time.time() - last_line_ts > 12 and not state.progress_stage:
                    state.progress_stage = 'Initializing libraries (first run can take 1-2 min)...'
        finally:
            proc.wait()
            state.exit_code = int(proc.returncode)
            buf.append(f"\n[process exited with code {state.exit_code}]\n")
            state.stdout = ''.join(buf)[-12000:]
            state.running = False
            # Ensure 100% on completion
            try:
                state.steps_done = max(state.steps_total, state.steps_done)
                state.progress_pct = 100
                state.progress_stage = 'Completed'
            except Exception:
                pass
            state.finished_at = time.time()
            # detect new run dir
            new_dir = _detect_new_run_dir(before_runs)
            # optionally rename to include project name suffix
            final_dir = _rename_run_dir_if_needed(new_dir, getattr(state, 'run_name', None)) if new_dir else None
            state.run_dir_name = final_dir or new_dir
            try:
                if state.run_dir_name:
                    # Persist friendly run name if provided
                    if getattr(state, 'run_name', None):
                        _write_run_name(state.run_dir_name, state.run_name)
                    # Track session runs for curated history view
                    sess = st.session_state.get('session_runs') or []
                    # replace old with new if renamed
                    try:
                        if new_dir in sess:
                            sess = [state.run_dir_name if r == new_dir else r for r in sess]
                    except Exception:
                        pass
                    if state.run_dir_name not in sess:
                        sess.insert(0, state.run_dir_name)
                    st.session_state['session_runs'] = sess[:50]
            except Exception:
                pass
    except Exception:
        state.running = False
        state.finished_at = time.time()
        tb = traceback.format_exc()
        state.stdout = (state.stdout or '') + "\n[thread exception]\n" + tb


# ----------------------------- UI -----------------------------

st.set_page_config(page_title="ReAct‑ExtrAct Runner", layout="wide", initial_sidebar_state="expanded")
_ensure_dirs()
_ensure_default_keys()

if 'run_state' not in st.session_state:
    st.session_state['run_state'] = RunState()
if 'session_runs' not in st.session_state:
    st.session_state['session_runs'] = []  # list of run dir names created this session

def _curate_runs() -> Tuple[List[str], Dict[str, str]]:
    """Return (display_names, display_to_real_dir) per user's requirement:
    - Keep only runs from this session plus one demo baseline from history
    - The demo baseline is displayed as 'demo_baseline'
    """
    all_runs = _list_run_dirs()
    # map to display name using optional run_name.txt
    def _display_for(r: str) -> str:
        return _format_run_display(r)
    session_runs = [r for r in (st.session_state.get('session_runs') or []) if r in all_runs]
    # pick one baseline run from history (latest by sort order)
    demo_dir = next((r for r in all_runs if r.endswith('_baseline')), None)
    display: List[str] = []
    mapping: Dict[str, str] = {}
    if session_runs:
        # Only surface a historical baseline as demo if it's NOT from this session
        if demo_dir and (demo_dir not in session_runs):
            display.append('demo_baseline')
            mapping['demo_baseline'] = demo_dir
        for r in session_runs:
            disp = _display_for(r)
            display.append(disp)
            mapping[disp] = r
    else:
        # No session runs (e.g., after reload). Show recent runs to avoid hiding latest results.
        if demo_dir:
            display.append('demo_baseline')
            mapping['demo_baseline'] = demo_dir
        for r in all_runs:
            if r == demo_dir:
                continue
            disp = _display_for(r)
            display.append(disp)
            mapping[disp] = r
    return display, mapping

# Add CSS for nav buttons
st.markdown(
    """
    <style>
    /* Header divider */
    .header-bar {border-bottom: 1px solid #e6e6e6; padding: 0; margin-bottom: 2px;}

    /* Make nav buttons orange themed */
    /* Default orange theme for primary buttons */
    div.stButton > button[kind="primary"] {
        background-color: #ff7a00 !important;
        border-color: #ff7a00 !important;
        color: white !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #ff8a1a !important;
        border-color: #ff8a1a !important;
    }
    /* Default orange outline for secondary buttons */
    div.stButton > button[kind="secondary"] {
        background-color: #ffffff !important;
        color: #ff7a00 !important;
        border: 2px solid #ff7a00 !important;
    }
    div.stButton > button[kind="secondary"]:hover {
        color: #ff8a1a !important;
        border-color: #ff8a1a !important;
    }
    /* Marker-based exceptions for wizard buttons */
    #add-field-marker ~ div.stButton button {
        background-color: #1976d2 !important;
        border-color: #1976d2 !important;
        color: #ffffff !important;
    }
    #add-field-marker ~ div.stButton button:hover { background-color: #1e88e5 !important; border-color: #1e88e5 !important; }
    #save-fields-marker ~ div.stButton button {
        background-color: #2e7d32 !important;
        border-color: #2e7d32 !important;
        color: #ffffff !important;
    }
    #save-fields-marker ~ div.stButton button:hover { background-color: #388e3c !important; border-color: #388e3c !important; }
    #delete-all-marker ~ div.stButton button {
        background-color: #d32f2f !important;
        border-color: #d32f2f !important;
        color: #ffffff !important;
    }
    #delete-all-marker ~ div.stButton button:hover { background-color: #e53935 !important; border-color: #e53935 !important; }
    /* Scoped color variants for specific buttons (override primary/secondary) */
    .btn-blue div.stButton > button[kind] {
        background-color: #1976d2 !important;
        border-color: #1976d2 !important;
        color: #ffffff !important;
    }
    .btn-blue div.stButton > button[kind]:hover {
        background-color: #1e88e5 !important;
        border-color: #1e88e5 !important;
    }
    .btn-green div.stButton > button[kind] {
        background-color: #2e7d32 !important;
        border-color: #2e7d32 !important;
        color: #ffffff !important;
    }
    .btn-green div.stButton > button[kind]:hover {
        background-color: #388e3c !important;
        border-color: #388e3c !important;
    }
    .btn-red div.stButton > button[kind] {
        background-color: #d32f2f !important;
        border-color: #d32f2f !important;
        color: #ffffff !important;
    }
    .btn-red div.stButton > button[kind]:hover {
        background-color: #e53935 !important;
        border-color: #e53935 !important;
    }
    /* Remove any extra text next to the sidebar arrow */
    [data-testid="collapsedControl"]::after,
    [data-testid="stSidebarCollapseControl"]::after { content: ""; }
    /* Disable fallback hint (we now attach next to arrow) */
    .sidebar-hint { display: none; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Top row: logo left, nav right
top_cols = st.columns([1, 3])
with top_cols[0]:
    # Load logo from frontend directory
    top_logo_path = os.path.join(REPO_ROOT, 'frontend', 'Logo2.png')
    if os.path.isfile(top_logo_path):
        st.image(top_logo_path, width=220)

with top_cols[1]:
    if 'nav_page' not in st.session_state:
        st.session_state['nav_page'] = 'New Extraction'

    # Add spacer to bottom-align nav buttons with logo (tweak height as needed)
    st.markdown("<div style='height: 140px'></div>", unsafe_allow_html=True)

    nav_map = {
        "🔍✨ New Extraction": "New Extraction",
        "📊 Results Dashboard": "Results Dashboard",
        "⚙️ Settings": "Settings",
    }
    nav_keys = list(nav_map.keys())
    nav_cols = st.columns(len(nav_keys))
    for i, key in enumerate(nav_keys):
        is_active = (st.session_state['nav_page'] == nav_map[key])
        btn_type = "primary" if is_active else "secondary"
        if nav_cols[i].button(key, key=f"navbtn_{i}", type=btn_type, use_container_width=True):
            st.session_state['nav_page'] = nav_map[key]
            st.rerun()
    

st.markdown('<div class="header-bar"></div>', unsafe_allow_html=True)

# Page from nav_page
page = st.session_state['nav_page']

# Clear inspector state when switching pages to avoid stale dialogs
_prev_page = st.session_state.get('last_nav_page')
if _prev_page != page:
    _clear_inspector_state()
    st.session_state['last_nav_page'] = page

# Make run state available before building controls
state: RunState = st.session_state['run_state']
# If idle with no logs, clear any leftover stage/percent
if not state.running and (not state.stdout):
    state.progress_stage = ''
    state.progress_pct = 0

# Sidebar: description only (no navigation)
with st.sidebar:
    st.caption("Click on arrow above for more workspace.")
    st.markdown("#### ReAct-ExtrAct: A Tool for Source-Grounded Automated Data Extraction in Systematic Reviews")
    st.markdown(
        """
        **Enhancing Reproducibility in Systematic Reviews 🔬**
        - Ensures consistency with a defined extraction schema.
        - Guarantees traceability with source-grounded data points.
        - Facilitates analysis with structured, exportable datasets.
        """
    )
    st.markdown("---")
    st.markdown("**🧭 Demo Flow**")
    st.markdown("- 🗂️ Upload PDFs (sample papers)")
    st.markdown("- 🧭 Define data items (fields)")
    st.markdown("- ⚙️ Pick a mode: Naive RAG · Iterative RAG · ReAct‑ExtrAct")
    st.markdown("- ▶️ Run extraction and monitor progress")
    st.markdown("- 📥 Export a CSV for analysis")
    st.markdown("---")
    st.caption("You can hide this panel to maximize workspace.")

# Remove duplicate navigation: header bar radio is the only nav

# Handle demo run trigger
if page == "New Extraction":
    # In-wizard controls
    # Demonstration video removed for local app experience

    st.markdown("### Step 1: Setup Project")
    st.info("First, give your project a name and upload the source documents (PDFs) that will form your corpus.")
    # Project name
    proj_cols = st.columns([2, 2, 2])
    with proj_cols[0]:
        project_name = st.text_input("Project name *", value=st.session_state.get('project_name',''), key="wiz_project_name", placeholder="e.g., my_research_project")
        st.session_state['project_name'] = project_name
    # Demo mode toggle
    demo_mode = st.toggle("Use example dataset", value=bool(st.session_state.get('use_demo', False)), key="wiz_demo")
    st.session_state['use_demo'] = demo_mode
    # Upload PDFs (disabled in demo mode)
    if not demo_mode:
        up_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True, key="wiz_upload")
        if up_files:
            # Clear input/ and copy uploaded files there (single input directory)
            for f in os.listdir(INPUT_DIR):
                try:
                    os.remove(os.path.join(INPUT_DIR, f))
                except Exception:
                    pass
            for up in up_files:
                with open(os.path.join(INPUT_DIR, up.name), 'wb') as f:
                    f.write(up.read())
            st.success(f"Selected {len(up_files)} file(s) for this run (saved in input/).")
    else:
        # Show available demo PDFs
        try:
            demo_pdfs = [f for f in os.listdir(INPUT_DIR_DEMO) if f.lower().endswith('.pdf')]
        except Exception:
            demo_pdfs = []
        if demo_pdfs:
            st.success(f"Example dataset enabled. Using {len(demo_pdfs)} PDF(s).")
            st.caption(", ".join(sorted(demo_pdfs)))
        else:
            st.warning("Example dataset enabled but no PDFs found. Add 5 sample PDFs to the example folder.")

    st.markdown("---")
    st.markdown("### Step 2: Define Extraction Fields")
    st.info(
        """
        **1. Define Your Extraction Fields**

        Start by specifying the data items you want to extract. For each field, list the specific information you're looking for. The more specific, the better the results.

        
        **2. Provide Codes for Automated Categorization**

        To use our LLM-assisted inductive coding feature, you must provide a list of Codes for each extraction field you defined above. The system will use these codes to categorize the content it extracts, returning concise and structured answers based on your list. If codes are not provided, a concise answer will be a simple summary of the final answer. **Warning:** The concise-answer feature, in particular if codes are not provided, might contain errors.


        **3. How to Get the Best Results**

        This tool uses Retrieval-Augmented Generation (RAG), which performs best when you provide specific, context-rich questions rather than a simple list of keywords.

        **💡 Tips for Precise Data Extraction:**

        - Frame it as a research question. Treat each extraction field as a clear, unambiguous question you would ask a research assistant.
        - Include protocol concepts. Incorporate terms from your review protocol (e.g., Population, Intervention, Outcomes) to anchor the system's retrieval process and improve accuracy.
        """
    )
    # Initialize from config; if empty, preload 5 default queri
    if 'wiz_queries' not in st.session_state:
        try:
            _existing_qs = _read_current_queries() or []
        except Exception:
            _existing_qs = []
        if not _existing_qs:
            _existing_qs = [
                {"topic": "What were the machine learning algorithms used in this study?", "possible_options": "AdaBoost, Best-first Search, Char-LSTM, GR-Learnt LSTM, LabelSpreading (RBF), Linear SVC, Logistic Regression, Naive Bayes, Random Forest, Rule-based Classifier, Semi-supervised ML, SVM, SVM-OAA"},
                {"topic": "What features were used to train machine learning models in this study?", "possible_options": "None"},
                {"topic": "Which social network or platform was used for data collection?", "possible_options": "Articles, Blogs, DWFP, Facebook, Forms, Kaggle, Magazine, News, OSAC, Tumblr, Twitter, WhatsApp, YouTube"},
                {"topic": "What was the size of the analyzed dataset?", "possible_options": "None"},
                {"topic": "What were the performance metrics and their reported values for each machine learning model in the study?", "possible_options": "Accuracy, F1 Score, MAP, MRR, NDCG, Precision, Recall, ROC-AUC"},
            ]
        st.session_state['wiz_queries'] = list(_existing_qs)
    # Fields Formulator removed per request
    q_rows: List[Dict[str, Any]] = []
    _to_delete: List[int] = []
    # If empty, show a gentle hint
    if not st.session_state['wiz_queries']:
        st.info("No fields found. Default 5 will appear if config is empty.")
    # Removed tip per request
    for i, q in enumerate(st.session_state['wiz_queries']):
        cols = st.columns([3, 3])
        with cols[0]:
            topic = st.text_input(
                f"Topic {i+1}",
                value=q.get('topic',''),
                key=f"wiz_topic_{i}",
                label_visibility="collapsed",
                placeholder=f"Topic {i+1}",
            )
        with cols[1]:
            opts = st.text_input(
                f"Codes {i+1}",
                value=q.get('possible_options','None'),
                key=f"wiz_opts_{i}",
                label_visibility="collapsed",
                placeholder=f"Codes {i+1}",
            )
        q_rows.append({"topic": topic, "possible_options": opts})
    # Per-field delete removed per request
    # Buttons in requested order: Add, Save, Delete All (stacked vertically)
    st.markdown("<div id='add-field-marker'></div>", unsafe_allow_html=True)
    if st.button("➕ Add New Field"):
        st.session_state['wiz_queries'].append({"topic": "", "possible_options": "None"})
        st.rerun()
    st.markdown("<div id='save-fields-marker'></div>", unsafe_allow_html=True)
    if st.button("❗ Save Extraction Fields", type="primary"):
        _write_queries_py(q_rows)
        st.session_state['wiz_queries'] = list(q_rows)
        st.success("Saved extraction fields to config/queries.py")
    st.markdown("<div id='delete-all-marker'></div>", unsafe_allow_html=True)
    if st.button("🗑️ Clear All Fields"):
        st.session_state['wiz_queries'] = []
        _write_queries_py([])
        st.success("All extraction fields cleared")
        st.rerun()

    st.markdown("---")
    st.markdown("### Step 3: Configure & Run")
    st.info(
        """
        Pick an extraction mode and start extraction.

        Requirements:
        - For hosted runs, API access is provided for you in this demo.
        - For local runs, you need both Ollama (for the LLM) and GROBID (for PDF parsing).
        """
    )
    st.markdown("**Mode Guide**")
    st.markdown("- Naive RAG: single‑pass retrieval and answer. Fastest; good for straightforward fields. Lowest cost; ideal for quick demos and sanity checks.")
    st.markdown("- Iterative RAG: two‑step refinement before answering. Slower; improves precision on nuanced fields. Balances speed and quality; reduces missed context and hallucinations.")
    
    st.markdown("- ReAct‑ExtrAct: adds confidence cues and structured extraction. Slowest; richest outputs. Produces citation‑backed, traceable entries; best when quality matters most.")
    
    st.markdown("---")
    # Allow running even if config/queries.py is empty (old behavior)
    try:
        _cfg_qs = [q for q in (_read_current_queries() or []) if str((q or {}).get('topic','')).strip()]
    except Exception:
        _cfg_qs = []
    # Friendly labels mapped to internal modes
    _mode_map = {
        "Naive RAG": "baseline",
        "Iterative RAG": "iter_retgen",
        "ReAct-ExtrAct": "react_extract",
    }
    _rev_map = {v: k for k, v in _mode_map.items()}
    # Hosted-demo notice removed for local app experience
    _current_internal = st.session_state.get('mode', 'baseline')
    _current_label = _rev_map.get(_current_internal, "Naive RAG")
    mode_label = st.selectbox("Run mode", options=list(_mode_map.keys()), index=list(_mode_map.keys()).index(_current_label), key="wiz_mode")
    st.session_state['mode'] = _mode_map.get(mode_label, 'baseline')
    # Optional planner heuristics override shown only for ReAct‑ExtrAct
    if st.session_state.get('mode') == 'react_extract':
        _ph_def = st.session_state.get('react_planner_heuristics', '')
        _ph_example = (
            "You are a meticulous research assistant using ReAct. Plan how to extract the target fields.\n"
            "Heuristics: (1) Methods/Experiments are ground truth for what was done.\n"
            "(2) Results contain performance metrics close to the model/dataset.\n"
            "(3) Related Work/References should not be primary evidence.\n"
            "(4) If ML paper, expect metrics, dataset details, model/algorithm, features, platform.\n"
            "(5) Prefer sections: Methods/Experiments/Results/Dataset/Conclusion; avoid: Related Work/References/Acknowledgments/Appendix.\n"
        )
        _ph_text = st.text_area(
            "Planner heuristic (ReAct‑ExtrAct)",
            value=_ph_def,
            height=140,
            placeholder=_ph_example,
            key="wiz_react_planner_heuristics",
            help="If provided, this replaces the default planner heuristics for ReAct‑ExtrAct.",
        )
        st.session_state['react_planner_heuristics'] = _ph_text
    # Provider selection
    try:
        from config.config import API as _API_DEF, EMBEDDING_API as _EAPI_DEF, OLLAMA_BASE_URL as _OLLAMA_URL_DEF, OLLAMA_EXECUTION_MODEL as _OLLAMA_EXEC_DEF, OLLAMA_EMBEDDING_MODEL as _OLLAMA_EMB_DEF
    except Exception:
        _API_DEF, _EAPI_DEF, _OLLAMA_URL_DEF, _OLLAMA_EXEC_DEF, _OLLAMA_EMB_DEF = 'openrouter', 'openai', 'http://localhost:11434', 'llama3.1:8b-instruct', 'nomic-embed-text'
    # Parsing Options (GROBID) — move above providers
    try:
        from config.config import USE_GROBID as _DEF_USE_GROBID, GROBID_URL as _DEF_GROBID_URL
    except Exception:
        _DEF_USE_GROBID, _DEF_GROBID_URL = False, "http://localhost:8070"
    pg_cols = st.columns([1,2,1])
    with pg_cols[0]:
        use_grobid = st.toggle("Use GROBID Parser", value=bool(os.getenv('USE_GROBID') == '1') or _DEF_USE_GROBID, key="wiz_use_grobid")
        os.environ['USE_GROBID'] = '1' if use_grobid else '0'
    with pg_cols[1]:
        if use_grobid:
            grobid_url = st.text_input("GROBID URL", value=(os.getenv('GROBID_URL') or _DEF_GROBID_URL), key="wiz_grobid_url")
            if grobid_url:
                os.environ['GROBID_URL'] = grobid_url
    with pg_cols[2]:
        if use_grobid:
            if st.button("Check GROBID status"):
                try:
                    import requests  # type: ignore
                    url = (os.getenv('GROBID_URL') or _DEF_GROBID_URL).rstrip('/') + '/api/isalive'
                    r = requests.get(url, timeout=8)
                    if r.status_code == 200:
                        st.success("GROBID is alive")
                    else:
                        st.warning(f"GROBID not healthy (HTTP {r.status_code})")
                except Exception as e:
                    st.error(f"GROBID check failed: {e}")

    # Providers and models below GROBID
    prov_cols = st.columns([1,1,2])
    with prov_cols[0]:
        api_choice = st.selectbox("LLM Provider", options=["openrouter","ollama"], index=["openrouter","ollama"].index(_API_DEF) if _API_DEF in ["openrouter","ollama"] else 0, key="wiz_api")
        os.environ['API'] = api_choice
    with prov_cols[1]:
        eapi_choice = st.selectbox("Embedding Provider", options=["openai","ollama"], index=["openai","ollama"].index(_EAPI_DEF) if _EAPI_DEF in ["openai","ollama"] else 0, key="wiz_eapi")
        os.environ['EMBEDDING_API'] = eapi_choice
    with prov_cols[2]:
        st.empty()

    # Ollama settings block below providers
    if api_choice == 'ollama' or eapi_choice == 'ollama':
        oll_cols = st.columns([2,2,2,1])
        with oll_cols[0]:
            obase = st.text_input("Ollama URL", value=os.getenv('OLLAMA_BASE_URL') or _OLLAMA_URL_DEF, key="wiz_ollama_base")
            if obase:
                os.environ['OLLAMA_BASE_URL'] = obase
        with oll_cols[1]:
            oexec = st.text_input("Ollama Exec Model", value=os.getenv('OLLAMA_EXECUTION_MODEL') or _OLLAMA_EXEC_DEF, key="wiz_ollama_exec")
            if oexec:
                os.environ['OLLAMA_EXECUTION_MODEL'] = oexec
        with oll_cols[2]:
            oemb = st.text_input("Ollama Embedding Model", value=os.getenv('OLLAMA_EMBEDDING_MODEL') or _OLLAMA_EMB_DEF, key="wiz_ollama_emb")
            if oemb:
                os.environ['OLLAMA_EMBEDDING_MODEL'] = oemb
        with oll_cols[3]:
            if st.button("Check Ollama status"):
                try:
                    import requests  # type: ignore
                    base = (os.getenv('OLLAMA_BASE_URL') or _OLLAMA_URL_DEF).rstrip('/')
                    url = base + '/api/version'
                    r = requests.get(url, timeout=8)
                    if r.status_code == 200:
                        try:
                            ver = r.json().get('version')
                            st.success(f"Ollama is alive (version: {ver or 'unknown'})")
                        except Exception:
                            st.success("Ollama is alive")
                    else:
                        st.warning(f"Ollama not healthy (HTTP {r.status_code})")
                except Exception as e:
                    st.error(f"Ollama check failed: {e}")

    # Executor (execution model) selection
    try:
        from config.config import EXECUTION_MODEL as _EXEC_DEF
    except Exception:
        _EXEC_DEF = "qwen/qwen-2.5-7b-instruct"
    exec_opts = [
        "qwen/qwen-2.5-7b-instruct",
        "qwen/qwen3-14b",
        "qwen/qwen3-32b",
    ]
    exec_opts = sorted(set(exec_opts))
    if api_choice == 'openrouter':
        sel_exec = st.selectbox("OpenRouter Exec Model", options=exec_opts, index=exec_opts.index(_EXEC_DEF) if _EXEC_DEF in exec_opts else 0, key="wiz_exec_model")
        # Apply to environment so downstream imports pick it up
        if sel_exec:
            os.environ['EXECUTION_MODEL'] = sel_exec
    # Evaluation disabled in this demo (no toggle)
    st.session_state['eval_off'] = True
    # Rebuild vector indexes option removed for demo; default is off
    st.session_state['rebuild_indexes'] = False
    run_btn = st.button("▶️ Start Extraction of Data", type="primary")


if page == "New Extraction" and run_btn and not state.running:
    # Read keys from config/env
    keys_now = _read_api_keys()
    if not _keys_ready(keys_now):
        state.progress_stage = 'Missing API keys. In this demo, keys are provided; please try again shortly.'
    else:
        active_dir = INPUT_DIR_DEMO if bool(st.session_state.get('use_demo')) else INPUT_DIR
        try:
            has_pdfs = any(name.lower().endswith('.pdf') for name in os.listdir(active_dir))
        except Exception:
            has_pdfs = False
        if not has_pdfs:
            state.progress_stage = ('No PDFs found in example dataset.' if bool(st.session_state.get('use_demo')) else 'No PDFs found in input/. Upload files first.')
        else:
            # Start background thread and initialize timer/progress immediately
            state.stdout = ''
            state.stderr = ''
            # Pre-compute planned steps (indexing + queries + eval)
            try:
                pdf_stems = [os.path.splitext(f)[0] for f in os.listdir(active_dir) if f.lower().endswith('.pdf')]
            except Exception:
                pdf_stems = []
            storage_base = os.path.join(REPO_ROOT, 'storage', 'openrouter')
            def _has_index_files(stem: str) -> bool:
                p = os.path.join(storage_base, f"{stem}_vector_index")
                return os.path.exists(os.path.join(p, 'docstore.json')) and os.path.exists(os.path.join(p, 'index_store.json'))
            needs_build = []
            idx_steps = 0
            clear_storage_flag = bool(st.session_state.get('rebuild_indexes'))
            if clear_storage_flag:
                idx_steps = 3 * len(pdf_stems)
                needs_build = list(pdf_stems)
            else:
                for stem in pdf_stems:
                    if _has_index_files(stem):
                        idx_steps += 1
                    else:
                        idx_steps += 3
                        needs_build.append(stem)
            try:
                q_total = max(1, len(_read_current_queries()))
            except Exception:
                q_total = 5
            query_steps = len(pdf_stems) * q_total
            eval_off_flag = bool(st.session_state.get('eval_off'))
            eval_steps = 0 if eval_off_flag else 1
            state.idx_steps_total = idx_steps
            state.query_steps_total = query_steps
            state.eval_steps_total = eval_steps
            state.steps_total = max(1, idx_steps + query_steps + eval_steps)
            state.steps_done = 0
            if idx_steps > 0:
                state.progress_stage = ('Parsing documents' if needs_build else 'Loading vector index')
            elif query_steps > 0:
                state.progress_stage = 'Querying your documents'
            else:
                state.progress_stage = 'Initializing...'
            state.progress_pct = 1
            state.started_at = time.time()
            # Pass optional planner heuristic override to backend via environment for ReAct‑ExtrAct
            try:
                if st.session_state.get('mode') == 'react_extract':
                    _ph_text = str(st.session_state.get('react_planner_heuristics') or '').strip()
                    if _ph_text:
                        os.environ['PLANNER_HEURISTICS_OVERRIDE'] = _ph_text
                    else:
                        try:
                            del os.environ['PLANNER_HEURISTICS_OVERRIDE']
                        except Exception:
                            pass
            except Exception:
                pass
            # Ensure environment override points to selected input directory
            state.active_input_dir = active_dir
            os.environ['INPUT_PATH'] = active_dir
            # capture friendly run name: "<project> (<mode>, <N> PDFs)"
            try:
                proj = (st.session_state.get('project_name') or '').strip()
                pdf_count = len(pdf_stems)
                internal_mode = st.session_state.get('mode', 'baseline')
                saved_name = f"{proj} ({internal_mode}, {pdf_count} PDFs)" if proj else f"{internal_mode} ({pdf_count} PDFs)"
                state.run_name = saved_name
            except Exception:
                state.run_name = (st.session_state.get('project_name') or '').strip() or None
            _mode_internal = st.session_state.get('mode', 'baseline')
            t = threading.Thread(target=_run_script, args=(_mode_internal, state, bool(st.session_state.get('eval_off')), bool(st.session_state.get('rebuild_indexes'))), daemon=True)
            t.start()
            st.rerun()


if page == "New Extraction":
    left = st.container()
    with left:
        st.subheader("Live Status")
        progress_ph = st.empty()
        show_progress = bool(state.running or state.progress_pct > 0 or state.progress_stage)
        if show_progress:
            # Ensure a minimally visible bar while running
            _disp_pct = max(5, int(state.progress_pct)) if state.running else int(state.progress_pct)
            # Derive a robust stage label from recent logs to avoid being stuck on "Launching..."
            recent = (state.stdout or '').splitlines()[-200:]
            recent_join = "\n".join(recent)
            display_stage = state.progress_stage or ''
            if state.running:
                try:
                    import re as _re
                    # Highest priority: evaluation, then active query loop
                    if any((l.strip().startswith('Evaluating:') or 'RAGAS evaluation' in l) for l in recent[-100:]):
                        display_stage = 'Evaluating answers'
                    elif any(_re.search(r"Query\s*\[\d+/\d+\]", l) for l in recent[-200:]):
                        display_stage = 'Querying your documents'
                    # New backend hints: executing/planning/engine
                    elif any(l.strip().startswith('[execute]') for l in recent[-200:]):
                        display_stage = 'Executing queries — please wait…'
                    elif any(l.strip().startswith('[plan]') for l in recent[-200:]):
                        display_stage = 'Planning'
                    elif any(l.strip().startswith('[engine]') for l in recent[-200:]):
                        display_stage = 'Query engine ready'
                    # Indexing/IO stages
                    elif any('loading existing index' in l for l in recent[-200:]):
                        display_stage = 'Loading vector index'
                    elif any('building VectorStoreIndex' in l or 'persisted index' in l for l in recent[-200:]):
                        display_stage = 'Building vector index'
                    elif any(l.startswith('[parse]') or 'Started parsing the file' in l for l in recent[-200:]):
                        display_stage = 'Parsing documents'
                    elif any('Processing file:' in l for l in recent[-200:]):
                        display_stage = 'Preparing documents'
                    elif not display_stage:
                        display_stage = 'Initializing...'
                except Exception:
                    pass
            else:
                # Respect completion/failure when run is finished
                if state.exit_code is not None and state.exit_code != 0:
                    display_stage = 'Failed'
                elif (state.progress_pct >= 100) or (not display_stage):
                    display_stage = 'Completed'
            # Render concise status and place spinner directly below
            status_txt = display_stage or 'Working...'
            progress_ph.markdown(f"**{status_txt}**")
            if state.running:
                st.image("https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExdDJjZ3JqZTV1ZzNkM2k4MHNobDBmYmdkZmQ4eWJ6b2sxajBodW4xMiZlcD12MV9naWZzX3NlYXJjaCZjdD1n/3oEjI6SIIHBdRxXI40/giphy.gif", width=120)
            else:
                # One-time completion/failure notification
                try:
                    notif_key = f"completed_notified__{state.run_dir_name or 'current'}"
                except Exception:
                    notif_key = "completed_notified__current"
                already = bool(st.session_state.get(notif_key))
                if not already:
                    if display_stage == 'Completed' and (state.exit_code is None or state.exit_code == 0):
                        st.success("Completed ✅")
                        st.session_state[notif_key] = True
                    elif display_stage == 'Failed' or (state.exit_code and state.exit_code != 0):
                        st.error("Failed ❌")
                        st.session_state[notif_key] = True
        else:
            progress_ph.empty()

        with st.expander("Logs of the system", expanded=False):
            st.code(state.stdout or "(no logs yet)")

        if state.running and state.process and st.button("Stop Extraction"):
            try:
                state.process.terminate()
            except Exception:
                pass
            state.progress_stage = 'Terminated by user'

        # Remove separate Debug; include essential diagnostics collapsed inside Logs if needed

        # Auto-refresh every second while run is active (placed after rendering)
        try:
            if state.running:
                time.sleep(1)
                st.rerun()
        except Exception:
            pass

    # Step 4 – Results table
        st.markdown("---")
    st.markdown("### Step 4: Examine Results")
    st.info(
        """
        Select a completed extraction to view answers and supporting evidence. Click an answer (🔍) to open Inspector mode with full answer and detailed evidence for that field.
        """
    )
    # Show a simple selector bar (newest to oldest). Do not auto-select a run.
    runs_list = _list_run_dirs()
    sentinel = "— Select a run —"
    runs_display = [_format_run_display(r) for r in runs_list]
    display_to_real = {d: r for d, r in zip(runs_display, runs_list)}
    sel_opt = st.selectbox("Select extraction to view", options=[sentinel] + runs_display, index=0, key="new_run_selected_run")
    selected_run = None if sel_opt == sentinel else display_to_real.get(sel_opt)
    # Keep selected run in session so Inspector writes validation to the same run
    if selected_run:
        st.session_state['selected_run'] = selected_run
    else:
        try:
            st.session_state.pop('selected_run', None)
        except Exception:
            pass
    if selected_run:
        data = _discover_results(selected_run)
        # Quick diagnostics
        num_papers = len(data)
        num_answers = sum(len((p or {}).get('results') or []) for p in data.values())
        st.caption(f"Papers: {num_papers} | Answers: {num_answers}")
        # Derive topic columns from actual results
        all_topics: Set[str] = set()
        for payload in (data.values() or []):
            for r in (payload.get('results') or []):
                t = ((r.get('query') or {}).get('topic') or r.get('question'))
                if t:
                    all_topics.add(str(t))
        topic_columns = sorted(all_topics)
        rows: List[Dict[str, Any]] = []
        for paper, payload in data.items():
            results = payload.get('results') or []
            topic_to_row = {((r.get('query') or {}).get('topic') or r.get('question')): r for r in results}
            row = {"paper": paper}
            for t in topic_columns:
                r = topic_to_row.get(t, {})
                conc = _strip_think((r.get('answer_concise') or '').strip())
                full = _strip_think((r.get('answer') or '').strip())
                row[t] = (conc if conc else full)
            rows.append(row)
        if rows and topic_columns:
            import pandas as pd
            df = pd.DataFrame(rows)
            st.caption("Click any cell to open the Inspector.")

            # Selection controls
            sel_papers = st.multiselect("Select papers", options=sorted(df['paper'].unique()), default=list(df['paper'].unique()), key="new_run_sel_papers")
            sel_topics = st.multiselect("Select fields", options=topic_columns, default=topic_columns, key="new_run_sel_topics")

            # Filter df and topics
            fdf = df[df['paper'].isin(sel_papers)] if sel_papers else df
            visible_topics = [t for t in topic_columns if t in (sel_topics or topic_columns)]

            # Pagination controls for topics (fields)
            topics_per_page = st.number_input(
                "Fields per page",
                min_value=1,
                max_value=50,
                value=10,
                step=1,
                key="new_run_topics_per_page",
            )
            num_pages = max(1, (len(visible_topics) + topics_per_page - 1) // topics_per_page)
            page_index = st.number_input(
                "Page",
                min_value=1,
                max_value=num_pages,
                value=1,
                step=1,
                key="new_run_topics_page",
            )
            start_idx = (page_index - 1) * topics_per_page
            end_idx = start_idx + topics_per_page
            page_topics = visible_topics[start_idx:end_idx]
            st.caption(f"Showing fields {start_idx + 1}–{min(end_idx, len(visible_topics))} of {len(visible_topics)}")

            # Header row: Paper | paginated topics
            header_cols = st.columns([1.2] + [2.2] * len(page_topics))
            with header_cols[0]:
                st.markdown("**Paper**")
            for j, t in enumerate(page_topics):
                with header_cols[j+1]:
                    st.markdown(f"**{t}**")

            # Data rows
            for ridx, row in fdf.iterrows():
                paper_nm = row['paper']
                row_cols = st.columns([1.2] + [2.2] * len(page_topics))
                with row_cols[0]:
                    short_paper = (paper_nm[:26] + '…') if len(paper_nm) > 28 else paper_nm
                    st.write(short_paper)
                for j, t in enumerate(page_topics):
                    ans_preview = str(_strip_think(row.get(t, '') or ''))
                    label = (ans_preview[:85] + '…') if len(ans_preview) > 88 else (ans_preview or "(empty)")
                    btn_key = f"new_run_tbl_btn__{paper_nm}__{t}__{ridx}__{j}"
                    if row_cols[j+1].button(f"🔍 {label}", key=btn_key):
                        st.session_state['inspector_paper'] = paper_nm
                        st.session_state['inspector_topic'] = t
                        st.session_state['inspector_open'] = True

            # Inspector helpers and renderer (scoped for new run)
            paper_to_payload = data
            _has_dialog = hasattr(st, 'dialog')

            def _find_result_for_topic_new(payload: Dict[str, Any], topic: str) -> Dict[str, Any]:
                try:
                    topic_norm = (topic or '').strip()
                    for r in (payload.get('results') or []):
                        t = ((r.get('query') or {}).get('topic') or r.get('question') or '')
                        if (t or '').strip() == topic_norm:
                            return r
                except Exception:
                    pass
                return {}

            def _extract_ctx_metadata_new(bc_or_ctx, _paper: Optional[str] = None) -> Dict[str, Any]:
                page = None
                section = ''
                if isinstance(bc_or_ctx, dict):
                    pv = bc_or_ctx.get('page')
                    if isinstance(pv, (int, float)):
                        page = int(pv)
                    elif isinstance(pv, str) and pv.isdigit():
                        page = int(pv)
                    sval = bc_or_ctx.get('section')
                    if isinstance(sval, str):
                        section = sval
                return {"page": page, "section": section}

            def _render_inspector_body_new(paper: str, topic: str):
                if not paper or not topic:
                    st.info("Select a paper and field to inspect.")
                    return
                payload = paper_to_payload.get(paper) or {}
                result = _find_result_for_topic_new(payload, topic)
                st.markdown(f"### 🔎 Inspector — {paper} · {topic}")
                # Answers (concise, full, code) first
                ca = _strip_think(result.get('answer_concise') or '')
                fa = _strip_think(result.get('answer') or '')
                if ca.strip():
                    st.markdown("**Concise Answer**")
                    st.write(ca)
                st.markdown("**Full Answer**")
                st.write(fa)
                if (result.get('code') or '').strip():
                    st.markdown("**Code**")
                    st.write(result.get('code') or '')
                # Validation toggle (per run / paper / topic)
                selected_run = st.session_state.get('selected_run') or st.session_state.get('current_run_dir') or st.session_state.get('run_dir_name')
                val_data = _read_validation(selected_run)
                was_valid = bool(((val_data.get(paper) or {}).get(topic)))
                new_valid = st.toggle("Validated", value=was_valid, key=f"val_new_{paper}_{topic}")
                if new_valid != was_valid:
                    val_for_paper = val_data.get(paper) or {}
                    val_for_paper[topic] = bool(new_valid)
                    val_data[paper] = val_for_paper
                    _write_validation(selected_run, val_data)
                # Then evidence
                st.markdown("**Evidence**")
                bclist = (result.get('best_context') or [])
                if not bclist:
                    st.caption("No evidence available.")
                for i, bc in enumerate(bclist[:5], start=1):
                    ctx = (bc or {}).get('context') or ''
                    meta = _extract_ctx_metadata_new(bc or {})
                    with st.expander(f"Snippet {i} — section: {meta.get('section') or 'n/a'} | page: {meta.get('page') if meta.get('page') is not None else 'n/a'}", expanded=(i==1)):
                        st.write(ctx)
                # Evaluation removed from inspector per request

                # Follow-up Q&A
                st.markdown("**❓ Ask a follow-up question**")
                fu_cols = st.columns([4,2,1])
                with fu_cols[0]:
                    fu_q = st.text_input("Question", key="new_run_insp_fu_q", placeholder="Ask about this paper…")
                with fu_cols[1]:
                    fu_mode = st.selectbox("Mode", options=["Quick", "Iterative (2 steps)"] , key="new_run_insp_fu_mode")
                with fu_cols[2]:
                    ask = st.button("Ask", key="new_run_insp_fu_ask")
                    fu_status = st.empty()
                if ask and (fu_q or '').strip():
                    fu_status.info("Querying index…")
                    resp = _run_followup_query(paper, fu_q.strip(), fu_mode)
                    fu_status.empty()
                    st.markdown("**Follow-up Answer**")
                    st.write(resp.get('answer') or '')
                    st.markdown("**Supporting contexts**")
                    for j, bc in enumerate((resp.get('contexts') or [])[:5], start=1):
                        meta = _extract_ctx_metadata_new(bc or {}, paper)
                        with st.expander(f"Context {j} — section: {meta.get('section') or 'n/a'} | page: {meta.get('page') if meta.get('page') is not None else 'n/a'}", expanded=(j==1)):
                            st.write((bc or {}).get('context') or '')

            if st.session_state.get('inspector_open'):
                if DEBUG_BLOCK_NEW_INSPECTOR:
                    st.info("Inspector is temporarily disabled on this page while debugging.")
                else:
                    ipaper = st.session_state.get('inspector_paper')
                    itopic = st.session_state.get('inspector_topic')
                    if _has_dialog:
                        @st.dialog("Results Inspector", width="large")
                        def _dlg_new():
                            _render_inspector_body_new(ipaper, itopic)
                        _dlg_new()
                    else:
                        st.markdown("---")
                        _render_inspector_body_new(ipaper, itopic)
                # Close inspector after rendering to avoid re-opening on refresh
                st.session_state['inspector_open'] = False
        else:
            st.warning("No answers found for this run. Ensure the run completed and wrote <paper>_result.json files.")

        st.markdown("---")
        st.markdown("### Step 5: Export CSV")
        st.caption("Choose which papers, fields, and columns to include in the CSV.")
        # Paper and field selection for export
        exp_papers = st.multiselect(
            "Select papers to include",
            options=sorted(list(data.keys())),
            default=sorted(list(data.keys())),
            key="new_run_exp_papers",
        )
        exp_cols = st.multiselect("Select fields (topics) to include", options=topic_columns, default=topic_columns, key="new_run_exp_cols")
        col_types = st.multiselect(
            "Columns to include per field",
            options=[
                "Concise answer",
                "Full answer",
                "Code",
                "Top context",
                "Top context score",
                "Validated",
            ],
            default=["Full answer"],
            key="new_run_col_types",
            help="Pick which columns you want for each field in the CSV.",
        )
        out_rows: List[Dict[str, Any]] = []
        for paper, payload in data.items():
            if exp_papers and paper not in exp_papers:
                continue
            reslist = payload.get('results') or []
            by_topic = {((r.get('query') or {}).get('topic') or r.get('question')): r for r in reslist}
            row: Dict[str, Any] = {"paper": paper}
            for t in exp_cols:
                r = by_topic.get(t)
                full = r.get('answer') or ''
                if "Concise answer" in col_types:
                    row[f"CONCISE ANSWER: {t}"] = (r.get('answer_concise') if r else '')
                if "Full answer" in col_types:
                    row[f"FULL ANSWER: {t}"] = full
                if "Code" in col_types:
                    row[f"CODE: {t}"] = (r.get('code') if r else '')
                if "Top context" in col_types:
                    bc = (r.get('best_context') or [{}])[0] if r else {}
                    row[f"TOP CONTEXT: {t}"] = bc.get('context') or ''
                if "Top context score" in col_types:
                    bc = (r.get('best_context') or [{}])[0] if r else {}
                    row[f"TOP CONTEXT SCORE: {t}"] = bc.get('score') if bc else ''
                if "Validated" in col_types:
                    # Use validation store if available
                    val_data = _read_validation(selected_run)
                    row[f"VALIDATED: {t}"] = bool(((val_data.get(paper) or {}).get(t)))
            out_rows.append(row)
        import pandas as pd
        out_df = pd.DataFrame(out_rows)
        if len(out_df.index) == 0:
            st.warning("No rows to export. Check that results exist for this run.")
        else:
            from config.config import CSV_ENCODING, CSV_DELIMITER
            csv_text = out_df.to_csv(index=False, sep=CSV_DELIMITER)
            csv_bytes = csv_text.encode(CSV_ENCODING)
            st.download_button("Download CSV", data=csv_bytes, file_name=f"{selected_run}_results.csv", mime="text/csv", key="new_run_download_csv")
            # No additional save options
    else:
        st.info("No finished run detected yet.")



elif page == "Results Dashboard":
    st.subheader("Extraction Results History")
    st.info("Review previous runs, filter by paper and field, and open the Inspector for answer details and evidence.")
    runs_display, display_to_real = _curate_runs()
    sentinel = "— Select a run —"
    options_list = [sentinel] + runs_display
    selected_display = st.selectbox("Select run", options=options_list, index=0, key="hist_selected_run")
    selected_run = display_to_real.get(selected_display) if selected_display != sentinel else None
    if selected_run is not None:
        data = _discover_results(selected_run)
        # Quick diagnostics
        num_papers = len(data)
        num_answers = sum(len((p or {}).get('results') or []) for p in data.values())
        st.caption(f"Papers: {num_papers} | Answers: {num_answers}")
        # Derive topic columns from actual results (robust to config changes)
        all_topics: Set[str] = set()
        for payload in (data.values() or []):
            for r in (payload.get('results') or []):
                t = ((r.get('query') or {}).get('topic') or r.get('question'))
                if t:
                    all_topics.add(str(t))
        topic_columns = sorted(all_topics)
        rows: List[Dict[str, Any]] = []
        for paper, payload in data.items():
            results = payload.get('results') or []
            topic_to_row = {((r.get('query') or {}).get('topic') or r.get('question')): r for r in results}
            row = {"paper": paper}
            for t in topic_columns:
                r = topic_to_row.get(t, {})
                conc = _strip_think((r.get('answer_concise') or '').strip())
                full = _strip_think((r.get('answer') or '').strip())
                row[t] = (conc if conc else full)
            rows.append(row)
        if rows and topic_columns:
            import pandas as pd
            df = pd.DataFrame(rows)
            st.caption("Click any cell to open the Inspector.")

            # Selection controls
            sel_papers = st.multiselect("Select papers", options=sorted(df['paper'].unique()), default=list(df['paper'].unique()), key="hist_sel_papers")
            sel_topics = st.multiselect("Select fields", options=topic_columns, default=topic_columns, key="hist_sel_topics")

            # Filter df and topics
            fdf = df[df['paper'].isin(sel_papers)] if sel_papers else df
            visible_topics = [t for t in topic_columns if t in (sel_topics or topic_columns)]

            # Pagination controls for topics (fields)
            topics_per_page = st.number_input(
                "Fields per page",
                min_value=1,
                max_value=50,
                value=10,
                step=1,
                key="hist_topics_per_page",
            )
            num_pages = max(1, (len(visible_topics) + topics_per_page - 1) // topics_per_page)
            page_index = st.number_input(
                "Page",
                min_value=1,
                max_value=num_pages,
                value=1,
                step=1,
                key="hist_topics_page",
            )
            start_idx = (page_index - 1) * topics_per_page
            end_idx = start_idx + topics_per_page
            page_topics = visible_topics[start_idx:end_idx]
            st.caption(f"Showing fields {start_idx + 1}–{min(end_idx, len(visible_topics))} of {len(visible_topics)}")

            # Header row: Paper | paginated topics
            header_cols = st.columns([1.2] + [2.2] * len(page_topics))
            with header_cols[0]:
                st.markdown("**Paper**")
            for j, t in enumerate(page_topics):
                with header_cols[j+1]:
                    st.markdown(f"**{t}**")

            # Data rows
            for ridx, row in fdf.iterrows():
                paper_nm = row['paper']
                row_cols = st.columns([1.2] + [2.2] * len(page_topics))
                with row_cols[0]:
                    short_paper = (paper_nm[:26] + '…') if len(paper_nm) > 28 else paper_nm
                    st.write(short_paper)
                for j, t in enumerate(page_topics):
                    ans_preview = str(_strip_think(row.get(t, '') or ''))
                    label = (ans_preview[:85] + '…') if len(ans_preview) > 88 else (ans_preview or "(empty)")
                    btn_key = f"hist_tbl_btn__{paper_nm}__{t}__{ridx}__{j}"
                    if row_cols[j+1].button(f"🔍 {label}", key=btn_key):
                        st.session_state['inspector_paper'] = paper_nm
                        st.session_state['inspector_topic'] = t
                        st.session_state['inspector_open'] = True

            # Inspector helpers and renderer
            paper_to_payload = data
            _has_dialog = hasattr(st, 'dialog')

            def _find_result_for_topic(payload: Dict[str, Any], topic: str) -> Dict[str, Any]:
                try:
                    topic_norm = (topic or '').strip()
                    for r in (payload.get('results') or []):
                        t = ((r.get('query') or {}).get('topic') or r.get('question') or '')
                        if (t or '').strip() == topic_norm:
                            return r
                except Exception:
                    pass
                return {}

            def _extract_ctx_metadata(bc_or_ctx, _paper: Optional[str] = None) -> Dict[str, Any]:
                page = None
                section = ''
                if isinstance(bc_or_ctx, dict):
                    pv = bc_or_ctx.get('page')
                    if isinstance(pv, (int, float)):
                        page = int(pv)
                    elif isinstance(pv, str) and pv.isdigit():
                        page = int(pv)
                    sval = bc_or_ctx.get('section')
                    if isinstance(sval, str):
                        section = sval
                return {"page": page, "section": section}

            def _render_inspector_body_hist(paper: str, topic: str):
                if not paper or not topic:
                    st.info("Select a paper and field to inspect.")
                    return
                payload = paper_to_payload.get(paper) or {}
                result = _find_result_for_topic(payload, topic)
                st.markdown(f"### 🔎 Inspector — {paper} · {topic}")
                # Answers (concise, full, code) first
                ca = _strip_think(result.get('answer_concise') or '')
                fa = _strip_think(result.get('answer') or '')
                if ca.strip():
                    st.markdown("**Concise Answer**")
                    st.write(ca)
                st.markdown("**Full Answer**")
                st.write(fa)
                if (result.get('code') or '').strip():
                    st.markdown("**Code**")
                    st.write(result.get('code') or '')
                # Then evidence
                st.markdown("**Evidence**")
                bclist = (result.get('best_context') or [])[:5]
                if not bclist:
                    st.caption("No evidence available.")
                for i, bc in enumerate(bclist, start=1):
                    ctx = (bc or {}).get('context') or ''
                    meta = _extract_ctx_metadata(bc or {})
                    with st.expander(f"Snippet {i} — section: {meta.get('section') or 'n/a'} | page: {meta.get('page') if meta.get('page') is not None else 'n/a'}", expanded=(i==1)):
                        st.write(ctx)
                # Evaluation removed from inspector per request

                # Follow-up Q&A
                st.markdown("**❓ Ask a follow-up question**")
                fu_cols = st.columns([4,2,1])
                with fu_cols[0]:
                    fu_q = st.text_input("Question", key="hist_insp_fu_q", placeholder="Ask about this paper…")
                with fu_cols[1]:
                    fu_mode = st.selectbox("Mode", options=["Quick", "Iterative (2 steps)"] , key="hist_insp_fu_mode")
                with fu_cols[2]:
                    ask = st.button("Ask", key="hist_insp_fu_ask")
                    fu_status = st.empty()
                if ask and (fu_q or '').strip():
                    fu_status.info("Querying index…")
                    resp = _run_followup_query(paper, fu_q.strip(), fu_mode)
                    fu_status.empty()
                    st.markdown("**Follow-up Answer**")
                    st.write(resp.get('answer') or '')
                    st.markdown("**Supporting contexts**")
                    for j, bc in enumerate((resp.get('contexts') or [])[:5], start=1):
                        meta = _extract_ctx_metadata(bc or {}, paper)
                        with st.expander(f"Context {j} — section: {meta.get('section') or 'n/a'} | page: {meta.get('page') if meta.get('page') is not None else 'n/a'}", expanded=(j==1)):
                            st.write((bc or {}).get('context') or '')

            if st.session_state.get('inspector_open'):
                ipaper = st.session_state.get('inspector_paper')
                itopic = st.session_state.get('inspector_topic')
                if _has_dialog:
                    @st.dialog("Results Inspector", width="large")
                    def _dlg_hist():
                        _render_inspector_body_hist(ipaper, itopic)
                    _dlg_hist()
                else:
                    st.markdown("---")
                    _render_inspector_body_hist(ipaper, itopic)
                # Close inspector after rendering to avoid re-opening on refresh
                st.session_state['inspector_open'] = False
        else:
            st.warning("No answers found for this run. Ensure the run completed and wrote <paper>_result.json files.")

        st.markdown("---")
        st.markdown("### Export CSV")
        st.caption("Choose which papers, fields, and columns to include in the CSV.")
        # Paper and field selection for export
        hist_exp_papers = st.multiselect(
            "Select papers to include",
            options=sorted(list(data.keys())),
            default=sorted(list(data.keys())),
            key="hist_exp_papers",
        )
        exp_cols = st.multiselect("Select fields (topics) to include", options=topic_columns, default=topic_columns, key="hist_exp_cols")
        col_types = st.multiselect(
            "Columns to include per field",
            options=[
                "Concise answer",
                "Full answer",
                "Code",
                "Top context",
                "Top context score",
                "Validated",
            ],
            default=["Full answer"],
            key="hist_col_types",
            help="Pick which columns you want for each field in the CSV.",
        )
        out_rows: List[Dict[str, Any]] = []
        for paper, payload in data.items():
            if hist_exp_papers and paper not in hist_exp_papers:
                continue
            reslist = payload.get('results') or []
            by_topic = {((r.get('query') or {}).get('topic') or r.get('question')): r for r in reslist}
            row: Dict[str, Any] = {"paper": paper}
            for t in exp_cols:
                r = by_topic.get(t)
                full = r.get('answer') or ''
                if "Concise answer" in col_types:
                    row[f"CONCISE ANSWER: {t}"] = (r.get('answer_concise') if r else '')
                if "Full answer" in col_types:
                    row[f"FULL ANSWER: {t}"] = full
                if "Code" in col_types:
                    row[f"CODE: {t}"] = (r.get('code') if r else '')
                if "Top context" in col_types:
                    bc = (r.get('best_context') or [{}])[0] if r else {}
                    row[f"TOP CONTEXT: {t}"] = bc.get('context') or ''
                if "Top context score" in col_types:
                    bc = (r.get('best_context') or [{}])[0] if r else {}
                    row[f"TOP CONTEXT SCORE: {t}"] = bc.get('score') if bc else ''
                if "Validated" in col_types:
                    val_data = _read_validation(selected_run)
                    row[f"VALIDATED: {t}"] = bool(((val_data.get(paper) or {}).get(t)))
            out_rows.append(row)
        import pandas as pd
        out_df = pd.DataFrame(out_rows)
        if len(out_df.index) == 0:
            st.warning("No rows to export. Select a different run or check that results exist.")
        else:
            from config.config import CSV_ENCODING, CSV_DELIMITER
            csv_text = out_df.to_csv(index=False, sep=CSV_DELIMITER)
            csv_bytes = csv_text.encode(CSV_ENCODING)
            st.download_button("Download CSV", data=csv_bytes, file_name=f"{selected_run}_results.csv", mime="text/csv", key="hist_download_csv")

elif page == "Settings":
    st.subheader("Settings")
    st.caption("Manage API keys and optional persistence. Keys are stored in-memory for this session and applied to environment variables.")

    keys_now = _read_api_keys()

    with st.form("keys_form"):
        st.markdown("**API Keys**")
        col_a, col_b = st.columns(2)
        with col_a:
            or_key = st.text_input("OPENROUTER_API_KEY", value=keys_now.get("OPENROUTER_API_KEY", ""), type="password")
            oa_key = st.text_input("OPENAI_API_KEY", value=keys_now.get("OPENAI_API_KEY", ""), type="password")
            lc_key = st.text_input("LLAMA_CLOUD_API_KEY", value=keys_now.get("LLAMA_CLOUD_API_KEY", ""), type="password")
        with col_b:
            oll_url = st.text_input("OLLAMA_BASE_URL (optional)", value=os.getenv("OLLAMA_BASE_URL") or keys_now.get("OLLAMA_BASE_URL", ""))
            grobid_url = st.text_input("GROBID_URL (optional)", value=os.getenv("GROBID_URL") or "http://localhost:8070")

        persist = st.checkbox("Persist to config/config_keys.py (writes on Save)", value=(os.getenv("ALLOW_WRITE_KEYS_FILE") == "1"))
        submitted = st.form_submit_button("Save")
        if submitted:
            if persist:
                os.environ["ALLOW_WRITE_KEYS_FILE"] = "1"
            _write_api_keys({
                'OPENROUTER_API_KEY': or_key,
                'OPENAI_API_KEY': oa_key,
                'LLAMA_CLOUD_API_KEY': lc_key,
            })
            if oll_url:
                os.environ['OLLAMA_BASE_URL'] = oll_url
            if grobid_url:
                os.environ['GROBID_URL'] = grobid_url
            st.success("Keys saved. They will be used immediately for this session.")

    st.divider()
    st.markdown("**Providers**")
    try:
        from config.config import API as _API_DEF, EMBEDDING_API as _EAPI_DEF
    except Exception:
        _API_DEF, _EAPI_DEF = 'openrouter', 'openai'
    prov_cols = st.columns(2)
    with prov_cols[0]:
        api_choice = st.selectbox("LLM Provider", options=["openrouter","ollama"], index=["openrouter","ollama"].index(_API_DEF) if _API_DEF in ["openrouter","ollama"] else 0, key="settings_api")
        os.environ['API'] = api_choice
    with prov_cols[1]:
        eapi_choice = st.selectbox("Embedding Provider", options=["openai","ollama"], index=["openai","ollama"].index(_EAPI_DEF) if _EAPI_DEF in ["openai","ollama"] else 0, key="settings_eapi")
        os.environ['EMBEDDING_API'] = eapi_choice


