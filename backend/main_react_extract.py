import os
import time
import threading
from contextlib import contextmanager
import json
import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from llama_index.core import Settings
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

from config.config import (
    API,
    INPUT_PATH,
    OUTPUT_PATH,
    STORAGE_PATH,
    EXECUTION_MODEL,
    EMBEDDING_MODEL,
    EMBEDDING_API,
    OLLAMA_BASE_URL,
    OLLAMA_EXECUTION_MODEL,
    OLLAMA_EMBEDDING_MODEL,
    CONCURRENCY as CFG_CONCURRENCY,
)
from config.config_keys import (
    OPENAI_API_KEY,
    LLAMA_CLOUD_API_KEY,
    OPENROUTER_API_KEY,
)

from config.queries import QUERIES
from utils.VectorQueryEngineCreator import VectorQueryEngineCreator
from utils.TokenTracker import TokenTracker
from utils.SectionClassifier import SectionClassifier

# Reuse core building blocks from heavyweight implementation
from utils.react_agent_utils import (
    ToolLibrary,
    AgentState,
    _synthesize_answer,
    PLANNER_HEURISTICS,
)
from utils.TraceRecorder import TraceRecorder

#s
# ---------------------------------------------
# A. Document IO Utilities (reuse minimal helpers)
# ---------------------------------------------

# Concurrency caps (environment-overridable)
GROUP_MAX_WORKERS = int(os.getenv("GROUP_MAX_WORKERS", "2"))  # concurrent groups per file
TOPIC_MAX_WORKERS = int(os.getenv("TOPIC_MAX_WORKERS", "4"))  # concurrent topics per group

# Global cap for LLM calls (hard upper bound across whole process)
GLOBAL_MAX_WORKERS = int(os.getenv("GLOBAL_MAX_WORKERS", "8"))
try:
    _GLOBAL_LLM_SEM = threading.BoundedSemaphore(GLOBAL_MAX_WORKERS)
    @contextmanager
    def llm_slot():
        _GLOBAL_LLM_SEM.acquire()
        try:
            yield
        finally:
            try:
                _GLOBAL_LLM_SEM.release()
            except Exception:
                pass
except Exception:
    @contextmanager
    def llm_slot():
        yield

def _persist_dir_for(file_stem: str) -> str:
    return os.path.join(STORAGE_PATH, f"{file_stem}_vector_index")


def _read_text_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def _read_section_tree(file_stem: str) -> str:
    return _read_text_file(os.path.join(_persist_dir_for(file_stem), "section_tree.md"))


def _read_raw_markdown(file_stem: str) -> str:
    return _read_text_file(os.path.join(_persist_dir_for(file_stem), "raw_markdown.md"))


def _normalize(s: str) -> str:
    return (s or "").strip().lower()


def _normalize_category_name(name: str) -> str:
    n = _normalize(name)
    # map common singulars to canonical plurals used by SectionClassifier
    if n == "method":
        return "methods"
    if n == "experiment":
        return "experiments"
    if n == "result":
        return "results"
    return n
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


def _filter_non_reference(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in results or []:
        if _is_reference_like(r.get("section") or "", r.get("section_category")):
            continue
        out.append(r)
    return out


def _to_baseline_compatible_results(file_stem: str, topics: List[str], payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    topic_to_options = {q.get("topic", ""): q.get("possible_options", "None") for q in QUERIES}
    def _assign_code_llm(topic: str, full_answer: str, concise_answer: str, possible_options: str) -> str:
        if str(possible_options).lower() == "none":
            return ""
        prompt = f"""
        Your task is to extract and return only the CODES from the provided options that appear in the given answer.

        Topic: {topic}
        Answer (full):
        {full_answer}

        Answer (concise):
        {concise_answer}

        Options (each may be CODE or "CODE | DEFINITION"):
        {possible_options}

        Requirements:
        - Match by either code or definition, but output ONLY the codes.
        - Return a comma-separated string of matching codes (e.g., "RF, LR").
        - Do not add any extra text.
        """.strip()
        try:
            raw = f"{Settings.llm.complete(prompt)!s}"
        except Exception:
            return ""
        alias_to_code: Dict[str, str] = {}
        for seg in str(possible_options).split(','):
            part = seg.strip()
            if not part:
                continue
            if '|' in part:
                left, right = part.split('|', 1)
                code = left.strip()
                definition = right.strip()
            else:
                code = part.strip()
                definition = ""
            if code:
                alias_to_code[code.lower()] = code
            if definition:
                alias_to_code[definition.lower()] = code
        chosen: List[str] = []
        for token in raw.split(','):
            key = token.strip().lower()
            code = alias_to_code.get(key)
            if code and code not in chosen:
                chosen.append(code)
        return ", ".join(chosen)
    out: List[Dict[str, Any]] = []
    for t in topics:
        ent = payload.get(t, {}) or {}
        best_ctx_list = ent.get("best_context") or []
        # Fallback: derive from evidence if best_context missing
        if not best_ctx_list:
            ev = ent.get("evidence") or []
            best_ctx_list = [{
                "text": r.get("text", ""),
                "score": r.get("score"),
                "page": r.get("page"),
                "section": r.get("section"),
            } for r in ev[:5]]
        # ensure list type
        if isinstance(best_ctx_list, dict):
            best_ctx_list = [best_ctx_list]
        best_context_serialized: List[Dict[str, Any]] = []
        for c in best_ctx_list[:5]:
            best_context_serialized.append({
                "context": (c.get("context") or c.get("text") or ""),
                "score": c.get("score"),
                "page": c.get("page"),
                "section": c.get("section"),
            })
        out.append({
            "query": {"topic": t, "possible_options": topic_to_options.get(t, "None")},
            "question": t,
            "answer": ent.get("answer", ""),
            "answer_concise": ent.get("concise_answer", ""),
            "code": _assign_code_llm(t, ent.get("answer", ""), ent.get("concise_answer", ""), topic_to_options.get(t, "None")),
            "best_context": best_context_serialized,
        })
    return out



def _build_section_category_map(file_stem: str) -> Dict[str, Dict[str, str]]:
    """Create a mapping from normalized heading → {raw, category}."""
    tree_path = os.path.join(_persist_dir_for(file_stem), "section_tree.md")
    clf = SectionClassifier(use_llm_fallback=True)
    return clf.classify_tree_file(tree_path)


def _map_section_to_category(section_text: str, heading_map: Dict[str, Dict[str, str]]) -> Optional[str]:
    sec = _normalize(section_text)
    if not sec:
        return None
    # exact match first
    if sec in heading_map:
        return heading_map[sec].get("category")
    # try contains or prefix match
    for k, v in heading_map.items():
        if not k:
            continue
        if k in sec or sec in k:
            return v.get("category")
    # simple keyword fallbacks aligned with classifier
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
    if "data" in sec:
        return "data"
    if "related work" in sec:
        return "related work"
    if "reference" in sec or "bibliograph" in sec:
        return "references"
    if "acknowledg" in sec:
        return "acknowledgments"
    if "appendix" in sec:
        return "appendix"
    return None


class AnnotatedToolLibrary:
    """
    Wrapper over ToolLibrary that annotates each result with a canonical
    section category using SectionClassifier-derived heading map.
    """

    def __init__(self, base_tools: ToolLibrary, heading_map: Dict[str, Dict[str, str]]):
        self._base = base_tools
        self._heading_map = heading_map or {}

    def query_document(self, *args, **kwargs) -> Dict[str, Any]:
        out = self._base.query_document(*args, **kwargs)
        results = list((out or {}).get("results") or [])
        for r in results:
            if "section_category" not in r:
                r["section_category"] = _map_section_to_category(r.get("section", ""), self._heading_map)
        out["results"] = results
        return out

    def validate_answer(self, *args, **kwargs) -> Dict[str, Any]:
        return self._base.validate_answer(*args, **kwargs)



def _extract_gist(llm, raw_md_text: str) -> str:
    sample = raw_md_text[:18000]
    prompt = f"""
        You are given a research paper's markdown content (may be partial).
        Write a concise, high-level summary (3-6 sentences) covering problem, approach, and key findings.
        Output only a single paragraph, no headings.

        Paper (subset):
        {sample}
    """.strip()
    try:
        with llm_slot():
            return f"{llm.complete(prompt)!s}".strip()
    except Exception as e:
        try:
            print(f"[llm-error] gist failed: {e}")
        except Exception:
            pass
        # fallback to first lines of markdown
        return " ".join((sample or "").splitlines()[:5])[:600]


# ---------------------------------------------
# B. Enhanced Planner (priority + avoid + groups)
# ---------------------------------------------

def _plan_topics_enhanced(llm, file_stem: str, topics: List[str]) -> Dict[str, Any]:
    section_tree = _read_section_tree(file_stem)
    raw_md = _read_raw_markdown(file_stem)
    gist = _extract_gist(llm, raw_md)

    prompt = f"""
    {PLANNER_HEURISTICS}

    You must plan the extraction with an explicit strategy:
    - Choose a priority order of topics to look into first based on likely availability in Methods/Experiments/Results.
    - Choose topics to avoid (e.g., vague, redundant, or likely only in Related Work/References).
    - Group topics that are tightly interdependent so they can run concurrently. Groups should be disjoint.
    - Keep groups small (2-5 topics) and logical

    Paper gist:
    {gist}

    Section tree (subset):
    {section_tree[:2000]}

    Target fields (topics):
    {json.dumps(topics, ensure_ascii=False)}

    Return ONLY strict JSON with keys:
    {{
      "order": [string],
      "prefer": [lowercase section hints],
      "avoid_sections": [lowercase section hints],
      "avoid_topics": [string],
      "groups": [[string]],
      "queries_by_topic": {{ topic: [subquery strings] }}
    }}
    """.strip()

    try:
        with llm_slot():
            raw = f"{llm.complete(prompt)!s}".strip()
    except Exception as e:
        try:
            print(f"[llm-error] planner failed: {e}")
        except Exception:
            pass
        raw = "{}"
    start = raw.find('{'); end = raw.rfind('}')
    json_str = raw[start:end+1] if start != -1 and end != -1 and end > start else raw

    data: Dict[str, Any]
    try:
        data = json.loads(json_str)
    except Exception:
        data = {}

    # Defaults and normalization
    prefer_default = ["method", "approach", "experiment", "result", "evaluation", "dataset", "conclusion"]
    avoid_sections_default = ["related work", "references", "acknowledgment", "appendix", "bibliograph"]

    order = data.get("order") if isinstance(data.get("order"), list) else list(topics)
    avoid_topics = data.get("avoid_topics") if isinstance(data.get("avoid_topics"), list) else []
    groups = data.get("groups") if isinstance(data.get("groups"), list) else []
    qbt = data.get("queries_by_topic") if isinstance(data.get("queries_by_topic"), dict) else {}
    queries_by_topic: Dict[str, List[str]] = {}
    for t in topics:
        arr = qbt.get(t) if isinstance(qbt.get(t), list) else []
        if not arr:
            arr = [t]
        seen: set = set(); seq: List[str] = []
        for q in arr:
            qn = str(q)
            if qn not in seen:
                seen.add(qn); seq.append(qn)
        queries_by_topic[t] = seq

    return {
        "gist": gist,
        "guide": {"prefer": data.get("prefer") or prefer_default, "avoid": data.get("avoid_sections") or avoid_sections_default},
        "section_tree": section_tree,
        "raw_md_present": bool(raw_md),
        "order": order,
        "groups": groups,
        "avoid_topics": avoid_topics,
        "queries_by_topic": queries_by_topic,
    }


# ---------------------------------------------
# C. Grouped Agent Execution
# ---------------------------------------------

def _uniq_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen_ids = set(); seen_txt = set(); out = []
    for r in results:
        nid = r.get("node_id")
        t = (r.get("text") or "")[:200]
        key_txt = t.strip().lower()
        if nid is not None:
            if nid in seen_ids:
                continue
            seen_ids.add(nid); out.append(r); continue
        if key_txt in seen_txt:
            continue
        seen_txt.add(key_txt); out.append(r)
    return out


def _deterministic_confidence(answer: str, evidence: List[Dict[str, Any]], prefer_hints: List[str], avoid_hints: List[str]) -> Tuple[float, str]:
    if not (answer or "").strip():
        return 0.2, "empty answer → 0.2"
    # normalize hints to canonical categories
    prefer_cats = [_normalize_category_name(h) for h in (prefer_hints or [])]
    avoid_cats = [_normalize_category_name(h) for h in (avoid_hints or [])]

    # Precompute lexical tokens from answer and (light) topic proxy from first sentence of answer
    ans_tokens = set([t for t in (str(answer or "").lower().split()) if len(t) > 2])

    # Prepare normalization for retrieval score
    raw_scores = [float(r.get("score") or 0.0) for r in (evidence or [])]
    max_score = max(raw_scores) if raw_scores else 0.0

    total_cap_tokens = 0.0
    strong_weight = 0.0
    weak_weight = 0.0

    details = []
    for r in (evidence or [])[:8]:
        txt = str(r.get("text") or "")
        # Cap per-snippet token influence to avoid one long chunk dominating
        tokens = min(120, max(1, len(txt.split())))
        total_cap_tokens += tokens

        sec_cat = _normalize_category_name(r.get("section_category") or "")
        sec_txt = _normalize(r.get("section") or "")
        is_preferred = bool(sec_cat and any(pc in sec_cat for pc in prefer_cats)) or any(ph in sec_txt for ph in prefer_cats)
        is_avoided = bool(sec_cat and any(ac in sec_cat for ac in avoid_cats)) or any(ah in sec_txt for ah in avoid_cats)

        # Retrieval score normalized
        rscore = float(r.get("score") or 0.0)
        rnorm = (rscore / max_score) if max_score > 0 else 0.0

        # Lexical overlap with answer
        ev_tokens = set([t for t in txt.lower().split() if len(t) > 2])
        overlap_ans = 0.0
        if ans_tokens:
            overlap_ans = len(ans_tokens & ev_tokens) / max(1.0, float(len(ans_tokens)))

        # Section-based signal: +1 for preferred, -1 for avoided, neutral 0 otherwise
        sec_signal = (1.0 if is_preferred else 0.0) - (1.0 if is_avoided else 0.0)

        # Combine signals (weights tuned conservatively)
        # Emphasize section match and retrieval score; include lexical grounding
        signal = 0.50 * sec_signal + 0.35 * rnorm + 0.15 * overlap_ans

        if signal >= 0:
            strong_weight += tokens * signal
        else:
            weak_weight += tokens * (-signal)

        details.append({
            "sec_cat": sec_cat,
            "section": sec_txt,
            "tokens": tokens,
            "rnorm": round(rnorm, 3),
            "overlap": round(overlap_ans, 3),
            "sec_signal": sec_signal,
            "signal": round(signal, 3),
        })

    denom = max(1.0, strong_weight + weak_weight)
    frac_strong = min(1.0, strong_weight / denom)
    frac_weak = min(1.0, weak_weight / denom)

    # Base 0.5, boost by strong, penalize by weak; keep within [0, 0.95]
    score = 0.5 + 0.4 * frac_strong - 0.1 * frac_weak
    score = max(0.0, min(0.95, score))

    log = (
        f"conf=0.5+0.4*{frac_strong:.2f}-0.1*{frac_weak:.2f}→{score:.2f} "
        f"[cap_tokens={int(total_cap_tokens)}, strong_w={strong_weight:.1f}, weak_w={weak_weight:.1f}]"
    )
    return score, log


def _select_best_context(answer: str, topic: str, evidence: List[Dict[str, Any]], prefer_hints: List[str], avoid_hints: List[str]) -> List[Dict[str, Any]]:
    if not evidence:
        return []
    ans_tokens = set([t for t in (str(answer or "").lower().split()) if len(t) > 2])
    topic_tokens = set([t for t in (str(topic or "").lower().split()) if len(t) > 2])
    prefer_cats = set(_normalize_category_name(p) for p in (prefer_hints or []))
    avoid_cats = set(_normalize_category_name(a) for a in (avoid_hints or []))
    # normalize retrieval scores for combination
    raw_scores = [float(r.get("score") or 0.0) for r in evidence]
    max_score = max(raw_scores) if raw_scores else 0.0
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for r in evidence:
        txt = str(r.get("text") or "").lower()
        ev_tokens = set([t for t in txt.split() if len(t) > 2])
        # lexical overlap with answer and topic
        common_ans = len(ans_tokens & ev_tokens)
        common_topic = len(topic_tokens & ev_tokens)
        overlap = 0.0
        if ans_tokens:
            overlap += common_ans / max(1.0, float(len(ans_tokens)))
        if topic_tokens:
            overlap += 0.5 * (common_topic / max(1.0, float(len(topic_tokens))))
        # retrieval score normalized
        rscore = float(r.get("score") or 0.0)
        rnorm = (rscore / max_score) if max_score > 0 else 0.0
        # section preference bonus
        sec_cat = _normalize_category_name(r.get("section_category") or "")
        sec_txt = _normalize(r.get("section") or "")
        bonus = 0.0
        if sec_cat and sec_cat in prefer_cats:
            bonus += 0.1
        if any(p in sec_txt for p in prefer_cats if p):
            bonus += 0.05
        if sec_cat and sec_cat in avoid_cats:
            bonus -= 0.05
        if any(a in sec_txt for a in avoid_cats if a):
            bonus -= 0.03
        # total combined score
        # Increase emphasis on embedding similarity vs lexical overlap
        total = 0.35 * overlap + 0.55 * rnorm + 0.10 * bonus
        scored.append((total, r))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[:5]]


def _process_topic_with_tools(tools: AnnotatedToolLibrary, topic: str, topic_idx: int, qlist: List[str], prefer: List[str], avoid: List[str], trace=None) -> Tuple[str, Dict[str, Any]]:
    search_term = qlist[0] if qlist else topic
    section_hint = None
    params = {"search_term": search_term, "section": section_hint}

    # First pass: conservative top_k=5
    obs_main = tools.query_document(**params, first_pass=True, top_k=5)
    params_b = dict(params); params_b["section"] = None
    obs_b = tools.query_document(**params_b, first_pass=True, top_k=5)
    merged_all = _uniq_results((obs_main.get("results", []) or []) + (obs_b.get("results", []) or []))
    merged = _filter_non_reference(list(merged_all))
    contexts = [r.get("text", "") for r in merged]
    with llm_slot():
        answer = _synthesize_answer(Settings.llm, topic, contexts)
    evidence = merged[:5]
    # Compute confidence on unfiltered evidence to capture weak (e.g., references) signal
    conf, clog = _deterministic_confidence(answer, merged_all[:8], prefer, avoid)
    try:
        if trace is not None:
            trace.record(
                "initial_answer",
                {
                    "topic": topic,
                    "query": search_term,
                    "results": [
                        {
                            "score": r.get("score"),
                            "section": r.get("section"),
                            "excerpt": (r.get("text") or "")[:300],
                        }
                        for r in merged[:8]
                    ],
                    "answer_preview": (answer or "")[:600],
                    "confidence": conf,
                    "confidence_log": clog,
                },
                topic=topic,
                step="initial",
            )
    except Exception:
        pass

    # One-step expansion if below threshold: expand neighborhood via higher top_k
    if conf < 0.75:
        obs_exp = tools.query_document(search_term, section=section_hint, top_k=12, first_pass=False)
        merged2_all = _uniq_results(merged_all + (obs_exp.get("results", []) or []))
        merged2 = _filter_non_reference(list(merged2_all))
        contexts = [r.get("text", "") for r in merged2]
        with llm_slot():
            answer = _synthesize_answer(Settings.llm, topic, contexts)
        evidence = merged2[:5]
        conf, clog = _deterministic_confidence(answer, merged2_all[:8], prefer, avoid)
        try:
            if trace is not None:
                trace.record(
                    "expansion_update",
                    {
                        "topic": topic,
                        "query": search_term,
                        "results": [
                            {
                                "score": r.get("score"),
                                "section": r.get("section"),
                                "excerpt": (r.get("text") or "")[:300],
                            }
                            for r in merged2[:10]
                        ],
                        "answer_preview": (answer or "")[:600],
                        "confidence": conf,
                        "confidence_log": clog,
                    },
                    topic=topic,
                    step="expand",
                )
        except Exception:
            pass

    result = {
        "answer": answer,
        "confidence": conf,
        "confidence_log": clog,
        "evidence": evidence,
        "best_context": _select_best_context(answer, topic, evidence, prefer, avoid),
        "validated": conf >= 0.75,
    }
    try:
        if trace is not None:
            trace.record(
                "topic_result",
                {
                    "topic": topic,
                    "result": {
                        "answer_preview": (result.get("answer") or "")[:600],
                        "confidence": result.get("confidence"),
                        "validated": result.get("validated"),
                    },
                },
                topic=topic,
                step="topic_final",
            )
    except Exception:
        pass
    return topic, result


def _split_list_items(text: str) -> List[str]:
    if not text:
        return []
    raw = text.strip()
    # try common bullet/separator patterns
    seps = ["\n- ", "\n* ", "\n• ", "\n", ";", ","]
    candidates: List[str] = []
    for sep in seps:
        if sep in raw:
            candidates = [p.strip(" -•*\t ") for p in raw.split(sep) if p.strip()]
            break
    if not candidates:
        candidates = [raw]
    out: List[str] = []
    seen: set = set()
    for c in candidates:
        k = c.lower()
        if k not in seen and c:
            seen.add(k)
            out.append(c)
    return out


def _has_inconsistency(a: str, b: str) -> bool:
    la = _split_list_items(a)
    lb = _split_list_items(b)
    if len(la) <= 1 and len(lb) <= 1:
        sa = (a or "").strip()
        sb = (b or "").strip()
        return bool(sa and sb and sa != sb)
    return len(la) != len(lb) or (set(x.lower() for x in la) != set(y.lower() for y in lb))


def _refine_due_to_inconsistency(tools: ToolLibrary, topic: str, current_answer: str, counterpart_answer: str) -> Dict[str, Any]:
    section_hint = None
    subqs = [
        f"{topic} exact list",
        f"{topic} reconcile: {current_answer} vs {counterpart_answer}",
    ]
    best_merge: List[Dict[str, Any]] = []
    for q in subqs:
        obs = tools.query_document(q, section=section_hint, top_k=5, first_pass=False)
        best_merge = _uniq_results(best_merge + (obs.get("results", []) or []))
    evidence = best_merge[:5]
    # Verify the existing answer without changing it
    with llm_slot():
        verdict = tools.validate_answer(topic, current_answer)
    conf = 0.85 if verdict.get("supported") else (0.5 if current_answer else 0.2)
    merged_ev = (verdict.get("evidence") or [])[:4]
    if len(merged_ev) < 4:
        merged_ev.extend(evidence[: max(0, 4 - len(merged_ev))])
    return {"answer": current_answer, "confidence": conf, "evidence": merged_ev, "validated": verdict.get("supported", False)}

def _merge_results(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base or {})
    # preserve original answer
    out["answer"] = (base or {}).get("answer") or (extra or {}).get("answer") or ""
    # merge evidence with dedupe
    ev_a = list((base or {}).get("evidence") or [])
    ev_b = list((extra or {}).get("evidence") or [])
    merged = _uniq_results(ev_a + ev_b)
    out["evidence"] = merged[:5]
    # confidence: take the max
    out["confidence"] = max(float((base or {}).get("confidence") or 0.0), float((extra or {}).get("confidence") or 0.0))
    # validated if either is validated
    out["validated"] = bool((base or {}).get("validated") or (extra or {}).get("validated"))
    # track number of consistency checks
    checks = int((base or {}).get("consistency_checks") or 0)
    out["consistency_checks"] = checks + 1
    return out

def run_agent_grouped(file_stem: str, query_engine, topics: List[str], plan: Dict[str, Any], out_dir: Optional[str] = None, trace=None) -> Dict[str, Any]:
    # Build section classification map and annotated tool wrapper
    heading_map = _build_section_category_map(file_stem)
    tools = AnnotatedToolLibrary(ToolLibrary(query_engine), heading_map)
    state = AgentState(
        file_stem=file_stem,
        topics=topics,
        gist=plan.get("gist", ""),
        guide=plan.get("guide", {"prefer": [], "avoid": []}),
        max_steps=24,
        queries_by_topic=plan.get("queries_by_topic", {}),
    )

    # Prepare groups: prioritize the provided topics order; don't re-derive from plan to avoid mismatches
    ordered = list(topics)
    # Do not exclude topics based on avoid list during execution; use avoid only as guidance
    avoid_topics = set()
    base_groups: List[List[str]] = []
    raw_groups = plan.get("groups") or []
    for g in raw_groups:
        if not isinstance(g, list):
            continue
        gg = [t for t in g if t in topics]
        if gg:
            base_groups.append(gg)
    # Add any remaining topics (preserving order) as singleton groups
    covered = {t for grp in base_groups for t in grp}
    for t in ordered:
        if t in covered or t not in topics:
            continue
        base_groups.append([t])
    # Fallback: if planner groups resolved to nothing, run all topics as one group
    if not base_groups:
        base_groups = [ordered]

    def _run_group(grp: List[str]) -> Dict[str, Dict[str, Any]]:
        group_results: Dict[str, Dict[str, Any]] = {}
        # Initial per-topic runs (single-step expansion if needed)
        with ThreadPoolExecutor(max_workers=min(TOPIC_MAX_WORKERS, max(1, len(grp)))) as pool:
            futs = {}
            for idx, t in enumerate(grp):
                qlist = plan.get("queries_by_topic", {}).get(t, [t])
                futs[pool.submit(
                    _process_topic_with_tools,
                    tools,
                    t,
                    idx,
                    qlist,
                    plan.get("guide", {}).get("prefer", []),
                    plan.get("guide", {}).get("avoid", []),
                    trace,
                )] = t
            for fut in as_completed(futs):
                t = futs[fut]
                try:
                    k, v = fut.result()
                    group_results[k] = v
                except Exception:
                    group_results[t] = {"answer": "", "confidence": 0.0, "evidence": [], "validated": False}

        # ReAct-style reconciliation within the group (max 2 rounds)
        def _meeting_and_followups(round_idx: int) -> Dict[str, str]:
            def _to_claim(text: str) -> str:
                tx = (text or "").strip().replace("\n", " ")
                # take first sentence or first ~200 chars as a concise claim
                sep_idx = tx.find('.')
                if 0 < sep_idx <= 200:
                    return tx[:sep_idx+1]
                return tx[:200]

            topics_desc: List[str] = []
            topic_to_evsecs: Dict[str, List[str]] = {}
            low_conf_or_avoid: Dict[str, bool] = {}
            avoid_hints = [(_normalize_category_name(a) or "") for a in (plan.get("guide", {}).get("avoid", []) or [])]
            for t in grp:
                res = group_results.get(t) or {}
                ev_secs: List[str] = []
                for e in (res.get("evidence") or [])[:3]:
                    cat = e.get("section_category") or _map_section_to_category(e.get("section"), heading_map) or ""
                    if cat:
                        ev_secs.append(_normalize_category_name(cat))
                topic_to_evsecs[t] = ev_secs
                conf_val = float(res.get("confidence") or 0.0)
                # mark if all anchors belong to avoided sections or confidence is low
                all_avoid = bool(ev_secs) and all(any(a in (sec or "") for a in avoid_hints) for sec in ev_secs)
                low_conf_or_avoid[t] = (conf_val < 0.75) or all_avoid
                topics_desc.append(
                    f"Topic: {t}\nClaim: {_to_claim(res.get('answer',''))}\nConfidence: {conf_val:.2f}\nEvidenceAnchors: {', '.join([s for s in ev_secs if s])}"
                )
            topics_block = "\n\n".join(topics_desc)
            prompt = f"""
            You are chairing a fast reconciliation meeting (round {round_idx+1}).
            Compare concise claim cards, confidence, and evidence anchors (section categories).

            Guidelines (be strict and practical):
            - Prefer Methods/Results/Dataset anchors; avoid using Related Work/References/Keywords to justify claims.
            - Ensure measurements are comparable (same definition/unit/threshold/procedure/stage).
            - Clarify entity/denominator (items/participants/documents) and whether subsets/filters/dedup changed totals.
            - Distinguish broad approaches from configured instances; don’t conflate auxiliary steps with main method.
            - Prefer later concrete sections over early intentions; resolve table vs narrative conflicts.
            - Retain caveats (no improvement/only under X/except Y); don’t upgrade qualified findings.
            - Be cautious of parsing/formatting artifacts; check units and magnitudes.

            Task:
            - Identify concrete issues (contradictions, gaps, missing anchors).
            - Propose at most ONE targeted follow-up query per topic to resolve issues.
            - If a topic needs no follow-up, set it to null.

            Return ONLY JSON with keys: {{ "issues": [string], "followups": {{ topic: string|null }} }}

            Items:
            {topics_block}
            """.strip()
            try:
                with llm_slot():
                    raw = f"{Settings.llm.complete(prompt)!s}".strip()
                s = raw.find('{'); e = raw.rfind('}')
                j = raw[s:e+1] if s != -1 and e != -1 and e > s else raw
                data = json.loads(j)
            except Exception:
                data = {"issues": [], "followups": {}}
            fups = data.get("followups") if isinstance(data.get("followups"), dict) else {}
            out: Dict[str, str] = {}
            for t in grp:
                v = fups.get(t)
                if isinstance(v, str) and v.strip():
                    out[t] = v.strip()
            # Enforce a follow-up if confidence is low or anchors are suspicious (avoided sections)
            if not out:
                enforced: Dict[str, str] = {}
                for t in grp:
                    if low_conf_or_avoid.get(t):
                        # simple targeted prompt to bias retrieval towards preferred sections
                        enforced[t] = f"verify {t} exact details in Methods/Results (units, thresholds, scope)"
                out = enforced or out
            try:
                if trace is not None:
                    trace.record(
                        "meeting_round",
                        {
                            "round": round_idx + 1,
                            "group": list(grp),
                            "proposed_followups": out,
                        },
                        step="meeting",
                    )
            except Exception:
                pass
            return out

        for ridx in range(2):
            followups = _meeting_and_followups(ridx)
            if not followups:
                break
            # Execute follow-ups serially per topic (fast) to avoid API spikes
            for t, q in followups.items():
                try:
                    obs = tools.query_document(f"{t} {q}", section=None, top_k=6, first_pass=False)
                    merged_all = _uniq_results((group_results.get(t, {}).get("evidence") or []) + (obs.get("results", []) or []))
                    merged = _filter_non_reference(list(merged_all))
                    contexts = [r.get("text", "") for r in merged]
                    with llm_slot():
                        ans = _synthesize_answer(Settings.llm, t, contexts)
                    ev = merged[:5]
                    conf, clog = _deterministic_confidence(ans, merged_all[:8], plan.get("guide", {}).get("prefer", []), plan.get("guide", {}).get("avoid", []))
                    prev = group_results.get(t, {})
                    if conf >= (prev.get("confidence") or 0.0) or (not prev.get("answer") and ans):
                        group_results[t] = {
                            "answer": ans,
                            "confidence": conf,
                            "confidence_log": clog,
                            "evidence": ev,
                            "best_context": _select_best_context(ans, t, ev, plan.get("guide", {}).get("prefer", []), plan.get("guide", {}).get("avoid", [])),
                            "validated": conf >= 0.75,
                        }
                        try:
                            if trace is not None:
                                trace.record(
                                    "followup_update",
                                    {
                                        "topic": t,
                                        "followup_query": q,
                                        "results": [
                                            {
                                                "score": r.get("score"),
                                                "section": r.get("section"),
                                                "excerpt": (r.get("text") or "")[:300],
                                            }
                                            for r in merged[:8]
                                        ],
                                        "answer_preview": (ans or "")[:600],
                                        "confidence": conf,
                                        "confidence_log": clog,
                                    },
                                    topic=t,
                                    step="followup",
                                )
                        except Exception:
                            pass
                except Exception:
                    continue
        return group_results

    # Execute groups concurrently; topics inside each group are processed concurrently by _run_group
    group_results_list: List[Dict[str, Dict[str, Any]]] = []
    with ThreadPoolExecutor(max_workers=min(GROUP_MAX_WORKERS, max(1, len(base_groups)))) as pool:
        futs = {pool.submit(_run_group, grp): tuple(grp) for grp in base_groups if grp}
        for fut in as_completed(futs):
            try:
                group_results_list.append(fut.result())
            except Exception:
                group_results_list.append({})

    # Merge and persist
    results_map: Dict[str, Dict[str, Any]] = {}
    for gm in group_results_list:
        results_map.update(gm)
    if out_dir:
        try:
            for t, v in results_map.items():
                if t in ordered:
                    with open(os.path.join(out_dir, f"topic_{ordered.index(t):02d}.json"), "w", encoding="utf-8") as f:
                        json.dump({"topic": t, "result": v}, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # Board-wide quick consistency check (non-destructive; at most 2 rounds with deterministic confidence)
    payload = {t: results_map.get(t, {}) for t in topics}
    for _ in range(2):
        to_refine: Dict[str, str] = {}
        valid_keys = [k for k, v in payload.items() if (v or {}).get("answer")]
        for i in range(len(valid_keys)):
            for j in range(i + 1, len(valid_keys)):
                ki, kj = valid_keys[i], valid_keys[j]
                ai = (payload.get(ki, {}).get("answer") or "")
                aj = (payload.get(kj, {}).get("answer") or "")
                if not ai or not aj:
                    continue
                if _has_inconsistency(ai, aj):
                    to_refine.setdefault(ki, aj)
                    to_refine.setdefault(kj, ai)
        if not to_refine:
            break
        # Single follow-up per item, recompute deterministic confidence
        for t, counterpart in to_refine.items():
            try:
                obs = tools.query_document(f"{t} reconcile {counterpart}", section=None, top_k=6, first_pass=False)
                merged_all = _uniq_results((payload.get(t, {}).get("evidence") or []) + (obs.get("results", []) or []))
                merged = _filter_non_reference(list(merged_all))
                contexts = [r.get("text", "") for r in merged]
                with llm_slot():
                    ans = _synthesize_answer(Settings.llm, t, contexts)
                ev = merged[:5]
                conf, clog = _deterministic_confidence(ans, merged_all[:8], plan.get("guide", {}).get("prefer", []), plan.get("guide", {}).get("avoid", []))
                prev = payload.get(t, {})
                if conf >= (prev.get("confidence") or 0.0) or (not prev.get("answer") and ans):
                    payload[t] = {
                        "answer": ans,
                        "confidence": conf,
                        "confidence_log": clog,
                        "evidence": ev,
                        "best_context": _select_best_context(ans, t, ev, plan.get("guide", {}).get("prefer", []), plan.get("guide", {}).get("avoid", [])),
                        "validated": conf >= 0.75,
                    }
                    try:
                        if trace is not None:
                            trace.record(
                                "consistency_update",
                                {
                                    "topic": t,
                                    "counterpart_answer_preview": (counterpart or "")[:300],
                                    "results": [
                                        {
                                            "score": r.get("score"),
                                            "section": r.get("section"),
                                            "excerpt": (r.get("text") or "")[:300],
                                        }
                                        for r in merged[:8]
                                    ],
                                    "answer_preview": (ans or "")[:600],
                                    "confidence": conf,
                                    "confidence_log": clog,
                                },
                                topic=t,
                                step="consistency",
                            )
                    except Exception:
                        pass
            except Exception:
                continue

    # Final succinct summarization per topic referencing the original query
    def _summarize(topic: str, answer: str) -> str:
        if not (answer or "").strip():
            return ""
        prompt = f"""
                Provide a concise answer to the topic using the provided text.

                Topic: {topic}
                Text: {answer}

                Requirements:
                - If the topic implies an enumeration (e.g., "list/which/name/keywords/methods/algorithms/etc."), output ONLY a comma-separated list of the items (short words or phrases), no extra words.
                - Otherwise, output ONLY a single short sentence or phrase directly answering the topic.
                - Do not add any prefixes, labels, or explanations.
                - Return just the answer as plain text.
                """.strip()
        try:
            with llm_slot():
                return f"{Settings.llm.complete(prompt)!s}".strip()
        except Exception:
            return answer
    for t, v in payload.items():
        v["concise_answer"] = _summarize(t, v.get("answer", ""))
        try:
            if trace is not None:
                trace.record(
                    "final",
                    {
                        "topic": t,
                        "answer": v.get("answer", ""),
                        "concise": v.get("concise_answer", ""),
                        "confidence": v.get("confidence"),
                        "validated": v.get("validated"),
                        "evidence": [
                            {
                                "score": r.get("score"),
                                "section": r.get("section"),
                                "excerpt": (r.get("text") or "")[:300],
                            }
                            for r in (v.get("evidence") or [])[:5]
                        ],
                    },
                    topic=t,
                    step="final",
                )
        except Exception:
            pass

        # Final best_context re-rank mirroring react_meetings (strictly over union of seen evidence + one concise-answer retrieval)
        try:
            ca = v.get("concise_answer") or ""
            obs_final = tools.query_document(ca or t, section=None, top_k=6, first_pass=False)
            union_prev = list(v.get("evidence") or [])
            union_all = _uniq_results(union_prev + (obs_final.get("results", []) or []))
            # Optional: retain union snapshot for future diagnostics
            v["_union_evidence"] = union_all[:50]
            def _score_meetings_style(r: Dict[str, Any]) -> float:
                txt = str(r.get("text") or "").lower()
                if not txt or ("reference" in (r.get("section") or "").lower()):
                    return 0.0
                ca_tokens = set([tok for tok in (ca.lower().split()) if len(tok) > 2])
                ev_tokens = set([tok for tok in txt.split() if len(tok) > 2])
                overlap = len(ca_tokens & ev_tokens) / max(1.0, float(len(ca_tokens))) if ca_tokens else 0.0
                rscore = float(r.get("score") or 0.0)
                return 0.6 * overlap + 0.4 * rscore
            ranked = sorted((_ for _ in union_all), key=_score_meetings_style, reverse=True)
            v["best_context"] = [{
                "context": r.get("text", ""),
                "score": r.get("score"),
                "page": r.get("page"),
                "section": r.get("section"),
            } for r in ranked[:5]]
            v["all_context"] = [{
                "context": (r.get("text") or "")[:500],
                "score": r.get("score"),
                "page": r.get("page"),
                "section": r.get("section"),
            } for r in union_all[:20]]
        except Exception:
            pass

    return {"extracted_data": payload, "gist": state.gist}


# ---------------------------------------------
# D. Main Entrypoint
# ---------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ReAct Extract Pipeline")
    parser.add_argument("--file", dest="file_stem", type=str, default=None, help="Specific PDF stem to process (e.g., '[52] 3-691-698')")
    parser.add_argument("--topics", dest="topics", type=str, default=None, help="Comma-separated topic indices or substrings to filter")
    parser.add_argument("--concurrency", dest="concurrency", type=int, default=None, help="Max PDFs to process concurrently")
    args = parser.parse_args()
    start = time.time()

    if API == "openrouter":
        print(f"Using OpenRouter execution model: {EXECUTION_MODEL} [REACT_EXTRACT]")
        Settings.llm = OpenAILike(
            model=EXECUTION_MODEL,
            api_base="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
            is_chat_model=True,
        )
    elif API == "ollama":
        exec_model = OLLAMA_EXECUTION_MODEL or EXECUTION_MODEL
        print(f"Using Ollama execution model: {exec_model} @ {OLLAMA_BASE_URL} [REACT_EXTRACT]")
        Settings.llm = Ollama(model=exec_model, base_url=OLLAMA_BASE_URL)
    else:
        raise ValueError("Unsupported API. Choose 'openrouter' or 'ollama'.")

    if EMBEDDING_API == "openai":
        Settings.embed_model = OpenAIEmbedding(
            model=EMBEDDING_MODEL,
            api_base="https://api.openai.com/v1",
            api_key=OPENAI_API_KEY,
        )
    elif EMBEDDING_API == "ollama":
        emb_model = OLLAMA_EMBEDDING_MODEL or EMBEDDING_MODEL
        Settings.embed_model = OllamaEmbedding(model_name=emb_model, base_url=OLLAMA_BASE_URL)
    else:
        raise ValueError("Unsupported EMBEDDING_API. Choose 'openai' or 'ollama'.")

    # Quick environment checks
    try:
        print(f"[check] api={API} emb_api={EMBEDDING_API} exec_model={EXECUTION_MODEL} emb_model={EMBEDDING_MODEL}")
        print(f"[check] keys: OPENROUTER={bool(OPENROUTER_API_KEY)} OPENAI={bool(OPENAI_API_KEY)} LLAMA_CLOUD={bool(LLAMA_CLOUD_API_KEY)}")
    except Exception:
        pass

    tracker = TokenTracker()
    tracker.install()

    run_tag = "react_extract"
    output_path = os.path.join(OUTPUT_PATH, f"{time.strftime('%Y.%m.%d_%H.%M.%S')}_{run_tag}")
    os.makedirs(output_path, exist_ok=True)
    recorder = TraceRecorder(output_path)

    files = [os.path.splitext(f)[0] for f in os.listdir(INPUT_PATH) if f.lower().endswith('.pdf')]
    if args.file_stem:
        files = [f for f in files if f == args.file_stem]
        if not files:
            print(f"No file matched --file={args.file_stem}. Available examples: {', '.join(sorted(files)[:5])}")
            return
    print(f"[check] files={len(files)} input_path={INPUT_PATH}")
    topics_all = [q.get("topic", "") for q in QUERIES]
    topics = list(topics_all)
    if args.topics:
        tokens = [t.strip() for t in str(args.topics).split(",") if t.strip()]
        selected: List[str] = []
        for tok in tokens:
            if tok.isdigit():
                idx = int(tok)
                if 0 <= idx < len(topics_all):
                    selected.append(topics_all[idx])
            else:
                sel = [t for t in topics_all if tok.lower() in t.lower()]
                selected.extend(sel)
        topics = list(dict.fromkeys(selected)) or topics_all
    max_conc = int(args.concurrency if args.concurrency is not None else (CFG_CONCURRENCY if isinstance(CFG_CONCURRENCY, int) else 3))
    print(f"[check] topics={len(topics)} concurrency={max_conc}")

    def _process_file(file_stem: str) -> None:
        print(f"\nProcessing file: {file_stem}")
        try:
            print(f"[parse] {file_stem}: starting markdown parsing")
        except Exception:
            pass
        query_engine = VectorQueryEngineCreator(
            llama_parse_api_key=LLAMA_CLOUD_API_KEY,
            cohere_api_key=os.getenv('COHERE_API_KEY',''),
            input_path=INPUT_PATH,
            storage_path=STORAGE_PATH,
            cohere_rerank=False,
            embedding_model_name=EMBEDDING_MODEL,
            response_mode="compact",
        ).get_query_engine(file_stem)
        try:
            print(f"[engine] {file_stem}: query engine ready")
        except Exception:
            pass

        trace = recorder.for_file(file_stem)
        try:
            print(f"[plan] {file_stem}: planning extraction strategy")
        except Exception:
            pass
        plan = _plan_topics_enhanced(Settings.llm, file_stem, topics)
        try:
            print(f"[plan] {file_stem}: groups={len(plan.get('groups') or [])} order_len={len(plan.get('order') or [])}")
        except Exception:
            pass
        try:
            trace.record(
                "plan",
                {
                    "order": plan.get("order"),
                    "groups": plan.get("groups"),
                    "queries_by_topic": plan.get("queries_by_topic"),
                    "prefer": (plan.get("guide", {}) or {}).get("prefer"),
                    "avoid": (plan.get("guide", {}) or {}).get("avoid"),
                    "gist_preview": (plan.get("gist") or "")[:600],
                },
                step="plan",
            )
        except Exception:
            pass
        out_dir = os.path.join(output_path, file_stem)
        os.makedirs(out_dir, exist_ok=True)

        # Determine concrete topic list to process (ignore avoid list to prevent empty answers)
        ordered = [t for t in plan.get("order", topics) if t in topics]
        if not ordered:
            ordered = list(topics)

        try:
            print(f"[execute] {file_stem}: executing queries — please wait…")
        except Exception:
            pass
        result = run_agent_grouped(file_stem, query_engine, ordered, plan, out_dir=out_dir)
        # Quick result stats
        try:
            payload_dbg = (result or {}).get("extracted_data", {})
            answered = sum(1 for v in (payload_dbg or {}).values() if (v or {}).get("answer"))
            print(f"[result] {file_stem}: answered_topics={answered}/{len(ordered)}")
        except Exception:
            pass
        # Save native result
        try:
            res_path = os.path.join(out_dir, f"{file_stem}_result.json")
            with open(res_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"[save] wrote {res_path}")
        except Exception as e:
            print(f"[save-error] result.json: {e}")
        # Save baseline-compatible result alongside
        try:
            payload = (result or {}).get("extracted_data", {})
            baseline_like = _to_baseline_compatible_results(file_stem, ordered, payload)
            bl_path = os.path.join(out_dir, f"{file_stem}_baseline_like.json")
            with open(bl_path, "w", encoding="utf-8") as f:
                json.dump(baseline_like, f, ensure_ascii=False, indent=2)
            print(f"[save] wrote {bl_path}")
        except Exception as e:
            print(f"[save-error] baseline_like.json: {e}")

    # Process files concurrently (configurable)
    with ThreadPoolExecutor(max_workers=max_conc) as pool:
        futs = {pool.submit(_process_file, f): f for f in files}
        for _ in as_completed(futs):
            pass

    total = time.time() - start
    print("END [REACT_EXTRACT]")
    print(f"Execution time: {total} seconds")
    tracker.write_report(output_path)
    rep = tracker.report()
    print(f"Token usage → LLM: {rep.get('total_llm_token_count')} | Embed: {rep.get('total_embedding_token_count')} | Total: {rep.get('total_token_count')}")
    print(f"Usage report: {os.path.join(output_path, 'usage.json')}")


if __name__ == "__main__":
    main()


