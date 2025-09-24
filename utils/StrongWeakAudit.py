import os
import json
import time
from typing import Dict, Any, List, Optional

from config.config import INPUT_PATH, OUTPUT_PATH, STORAGE_PATH, EMBEDDING_MODEL
from config.config_keys import OPENAI_API_KEY, LLAMA_CLOUD_API_KEY, COHERE_API_KEY
from config.queries import QUERIES

from utils.VectorQueryEngineCreator import VectorQueryEngineCreator
from utils.SectionClassifier import SectionClassifier


def _persist_dir_for(file_stem: str) -> str:
    return os.path.join(STORAGE_PATH, f"{file_stem}_vector_index")


def _read_text_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def _normalize(s: str) -> str:
    return (s or "").strip().lower()


def _normalize_category_name(name: str) -> str:
    n = _normalize(name)
    if n == "method":
        return "methods"
    if n == "experiment":
        return "experiments"
    if n == "result":
        return "results"
    return n


def _build_section_category_map(file_stem: str) -> Dict[str, Dict[str, str]]:
    path = os.path.join(_persist_dir_for(file_stem), "section_tree.md")
    clf = SectionClassifier(use_llm_fallback=True)
    return clf.classify_tree_file(path)


def _map_section_to_category(section_text: str, heading_map: Dict[str, Dict[str, str]]) -> Optional[str]:
    sec = _normalize(section_text)
    if not sec:
        return None
    if sec in heading_map:
        return heading_map[sec].get("category")
    for k, v in heading_map.items():
        if k and (k in sec or sec in k):
            return v.get("category")
    if "method" in sec:
        return "methods"
    if "experiment" in sec:
        return "experiments"
    if "result" in sec:
        return "results"
    if "evaluation" in sec:
        return "evaluation"
    if "dataset" in sec or "data set" in sec:
        return "dataset"
    if "related work" in sec:
        return "related work"
    if "reference" in sec or "bibliograph" in sec:
        return "references"
    return None


def _is_reference_like(section_text: str, section_category: Optional[str]) -> bool:
    sec = _normalize(section_text)
    cat = _normalize(section_category or "")
    bad_keys = [
        "related work",
        "reference",
        "references",
        "bibliograph",
        "acknowledg",
        "appendix",
        "keyword",
        "index term",
        "subject descriptor",
    ]
    if any(k in sec for k in bad_keys):
        return True
    if any(k in cat for k in bad_keys):
        return True
    return False


def _collect_results_for_topic(qe, topic: str, heading_map: Dict[str, Dict[str, str]]) -> List[Dict[str, Any]]:
    try:
        resp = qe.query(topic)
        nodes = getattr(resp, 'source_nodes', [])
    except Exception:
        nodes = []
    results: List[Dict[str, Any]] = []
    for n in nodes[:12]:
        try:
            content = (n.node.get_content() or '').strip()
        except Exception:
            try:
                content = str(getattr(n, 'get_content', lambda: '')()).strip()
            except Exception:
                content = ''
        try:
            meta = getattr(n, 'node', None)
            meta = getattr(meta, 'metadata', None) or {}
        except Exception:
            meta = {}
        section = meta.get('section') if isinstance(meta, dict) else None
        score = getattr(n, 'score', None)
        cat = _map_section_to_category(section or '', heading_map) or ''
        if _is_reference_like(section or '', cat):
            continue
        results.append({
            "text": content,
            "section": section,
            "section_category": cat,
            "score": score,
        })
    return results


def _audit_file(file_stem: str) -> Dict[str, Any]:
    heading_map = _build_section_category_map(file_stem)
    vqc = VectorQueryEngineCreator(
        llama_parse_api_key=LLAMA_CLOUD_API_KEY,
        cohere_api_key=COHERE_API_KEY,
        input_path=INPUT_PATH,
        storage_path=STORAGE_PATH,
        cohere_rerank=False,
        embedding_model_name=EMBEDDING_MODEL,
        response_mode="compact",
    )
    qe = vqc.get_query_engine(file_stem)

    prefer_default = ["method", "approach", "experiment", "result", "evaluation", "dataset", "conclusion"]
    avoid_default = ["related work", "references", "acknowledgment", "appendix", "bibliograph"]
    prefer_cats = [_normalize_category_name(x) for x in prefer_default]
    avoid_cats = [_normalize_category_name(x) for x in avoid_default]

    strong_tokens = 0
    weak_tokens = 0
    total_tokens = 0
    per_topic: Dict[str, Dict[str, Any]] = {}

    for q in QUERIES:
        topic = q.get("topic", "").strip()
        if not topic:
            continue
        results = _collect_results_for_topic(qe, topic, heading_map)
        st = 0
        wt = 0
        tt = 0
        for r in results:
            txt = r.get("text") or ""
            tokens = max(1, len(txt.split()))
            cat = _normalize_category_name(r.get("section_category") or "")
            sec_txt = _normalize(r.get("section") or "")
            is_strong = bool(cat and any(pc in cat for pc in prefer_cats)) or any(ph in sec_txt for ph in prefer_cats)
            is_weak = bool(cat and any(ac in cat for ac in avoid_cats)) or any(ah in sec_txt for ah in avoid_cats)
            tt += tokens
            if is_strong:
                st += tokens
            if is_weak:
                wt += tokens
        strong_tokens += st
        weak_tokens += wt
        total_tokens += tt
        per_topic[topic] = {
            "tokens": tt,
            "strong": st,
            "weak": wt,
            "frac_strong": (st / max(1, tt)),
            "frac_weak": (wt / max(1, tt)),
        }

    summary = {
        "file": file_stem,
        "tokens": total_tokens,
        "strong": strong_tokens,
        "weak": weak_tokens,
        "frac_strong": (strong_tokens / max(1, total_tokens)),
        "frac_weak": (weak_tokens / max(1, total_tokens)),
    }
    return {"summary": summary, "per_topic": per_topic}


def main():
    ts_dir = os.path.join(OUTPUT_PATH, f"{time.strftime('%Y.%m.%d_%H.%M.%S')}_strongweak_audit")
    os.makedirs(ts_dir, exist_ok=True)

    files = [os.path.splitext(f)[0] for f in os.listdir(INPUT_PATH) if f.lower().endswith('.pdf')]
    global_tokens = 0
    global_strong = 0
    global_weak = 0
    per_file_results: Dict[str, Any] = {}

    for stem in files:
        try:
            res = _audit_file(stem)
        except Exception as e:
            res = {"summary": {"file": stem, "error": str(e)}, "per_topic": {}}
        per_file_results[stem] = res
        s = res.get("summary", {})
        global_tokens += int(s.get("tokens") or 0)
        global_strong += int(s.get("strong") or 0)
        global_weak += int(s.get("weak") or 0)

    global_summary = {
        "tokens": global_tokens,
        "strong": global_strong,
        "weak": global_weak,
        "frac_strong": (global_strong / max(1, global_tokens)),
        "frac_weak": (global_weak / max(1, global_tokens)),
        "files": len(files),
        "topics_per_file": len(QUERIES),
    }

    out = {"global": global_summary, "files": per_file_results}
    out_path = os.path.join(ts_dir, "strongweak_audit.json")
    # Ensure directory exists (defensive)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Wrote audit: {out_path}")


if __name__ == "__main__":
    main()


