import os
import json
import time
from typing import Dict, Any, List

from config.config import INPUT_PATH, OUTPUT_PATH, STORAGE_PATH, EMBEDDING_MODEL
from config.config_keys import OPENAI_API_KEY, LLAMA_CLOUD_API_KEY, COHERE_API_KEY
from utils.VectorQueryEngineCreator import VectorQueryEngineCreator


def _persist_dir_for(file_stem: str) -> str:
    return os.path.join(STORAGE_PATH, f"{file_stem}_vector_index")


def _read_text_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def _collect_sections_markdown(file_stem: str) -> List[str]:
    # section_tree.md is produced in vector index persist dir
    p = os.path.join(_persist_dir_for(file_stem), "section_tree.md")
    raw = _read_text_file(p)
    if not raw:
        return []
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    # naive: take lines that look like headings
    headings = []
    for ln in lines:
        if ln.startswith('#') or ln.startswith('- '):
            # strip bullets and hashes
            h = ln.lstrip('#').lstrip('-').strip()
            if h:
                headings.append(h)
    return headings


def _collect_nodes_sections(qe) -> List[str]:
    # sample a broad query to pull nodes and read their sections
    sections: List[str] = []
    try:
        resp = qe.query("overview methods results dataset evaluation related work references")
        nodes = getattr(resp, 'source_nodes', [])
    except Exception:
        nodes = []
    for n in nodes[:50]:
        try:
            meta = getattr(n, 'node', None)
            meta = getattr(meta, 'metadata', None) or {}
        except Exception:
            meta = {}
        sec = meta.get('section') if isinstance(meta, dict) else None
        if isinstance(sec, str) and sec.strip():
            sections.append(sec.strip())
    return sections


def main():
    ts_dir = os.path.join(OUTPUT_PATH, f"{time.strftime('%Y.%m.%d_%H.%M.%S')}_section_audit")
    os.makedirs(ts_dir, exist_ok=True)

    files = [os.path.splitext(f)[0] for f in os.listdir(INPUT_PATH) if f.lower().endswith('.pdf')]
    summary: Dict[str, Any] = {
        "files": len(files),
        "total_headings": 0,
        "total_node_sections": 0,
        "by_file": {}
    }

    vqc = VectorQueryEngineCreator(
        llama_parse_api_key=LLAMA_CLOUD_API_KEY,
        cohere_api_key=COHERE_API_KEY,
        input_path=INPUT_PATH,
        storage_path=STORAGE_PATH,
        cohere_rerank=False,
        embedding_model_name=EMBEDDING_MODEL,
        response_mode="compact",
    )

    for stem in files:
        try:
            qe = vqc.get_query_engine(stem)
        except Exception:
            qe = None
        headings = _collect_sections_markdown(stem)
        node_secs = _collect_nodes_sections(qe) if qe else []
        summary["total_headings"] += len(headings)
        summary["total_node_sections"] += len(node_secs)
        summary["by_file"][stem] = {
            "headings_detected": len(headings),
            "sample_node_sections": len(node_secs),
            "headings_examples": headings[:10],
            "node_section_examples": node_secs[:10],
        }

    out_path = os.path.join(ts_dir, "section_audit.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Wrote section audit: {out_path}")


if __name__ == "__main__":
    main()


