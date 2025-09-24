import os
import time
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from llama_index.core import Settings
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.schema import NodeWithScore

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
)
from config.config_keys import (
    OPENAI_API_KEY,
    LLAMA_CLOUD_API_KEY,
    COHERE_API_KEY,
    OPENROUTER_API_KEY,
)

from config.queries import QUERIES
from utils.VectorQueryEngineCreator import VectorQueryEngineCreator
from utils.TokenTracker import TokenTracker


# ---------------------------------------------
# A. Document IO Utilities
# ---------------------------------------------

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


def _get_content_safe(n: NodeWithScore) -> str:
    try:
        return (n.node.get_content() or "").strip()
    except Exception:
        pass
    try:
        getter = getattr(n, "get_content", None)
        if callable(getter):
            return str(getter() or "").strip()
    except Exception:
        pass
    return ""


# ---------------------------------------------
# B. Contextualizer: Gist extraction
# ---------------------------------------------

def _extract_gist(llm, raw_md_text: str) -> str:
    sample = raw_md_text[:18000]
    prompt = f"""
        You are given a research paper's markdown content (may be partial).
        Write a concise, high-level summary (3-6 sentences) covering problem, approach, and key findings.
        Output only a single paragraph, no headings.

        Paper (subset):
        {sample}
    """.strip()
    return f"{llm.complete(prompt)!s}".strip()


# ---------------------------------------------
# C. Planner & Heuristics (lightweight)
# ---------------------------------------------

PLANNER_HEURISTICS = (
    "You are a meticulous research assistant using ReAct. Plan how to extract the target fields.\n"
    "Heuristics: (1) Methods/Experiments are ground truth for what was done.\n"
    "(2) Results contain performance metrics close to the model/dataset.\n"
    "(3) Related Work/References should not be primary evidence.\n"
    "(4) If ML paper, expect metrics, dataset details, model/algorithm, features, platform.\n"
    "(5) Use living-canvas facts already found as anchors for sub-queries.\n"
    "(6) Prefer sections: Methods/Experiments/Results/Dataset/Conclusion; avoid: Related Work/References/Acknowledgments/Appendix.\n"
    "(7) Generate sub-queries that chain previously found entities (algorithm ↔ metrics, dataset ↔ metrics).\n"
)

def _plan_topics(llm, file_stem: str, topics: List[str]) -> Dict[str, Any]:
    section_tree = _read_section_tree(file_stem)
    raw_md = _read_raw_markdown(file_stem)
    gist = _extract_gist(llm, raw_md)
    # LLM-driven planner: returns order and per-topic sub-queries using natural-language heuristics
    prompt = f"""
    {PLANNER_HEURISTICS}

    Paper gist:
    {gist}

    Section tree (subset):
    {section_tree[:2000]}

    Target fields (topics):
    {json.dumps(topics, ensure_ascii=False)}

    Return ONLY valid JSON with keys:
    {{
      "order": [string],
      "queries_by_topic": {{ topic: [subquery strings] }},
      "prefer": [lowercase section hints],
      "avoid": [lowercase section hints]
    }}
    """.strip()
    raw = f"{llm.complete(prompt)!s}".strip()
    start = raw.find('{'); end = raw.rfind('}')
    json_str = raw[start:end+1] if start != -1 and end != -1 and end > start else raw
    plan: Dict[str, Any]
    try:
        plan = json.loads(json_str)
    except Exception:
        plan = {}
    # Defaults
    prefer = ["method", "approach", "experiment", "result", "evaluation", "dataset", "conclusion"]
    avoid = ["related work", "references", "acknowledgment", "appendix"]
    order = plan.get("order") if isinstance(plan.get("order"), list) else list(topics)
    qbt = plan.get("queries_by_topic") if isinstance(plan.get("queries_by_topic"), dict) else {}
    # ensure minimal subqueries per topic
    queries_by_topic: Dict[str, List[str]] = {}
    for t in topics:
        arr = qbt.get(t) if isinstance(qbt.get(t), list) else []
        if not arr:
            arr = [t]
        seen: set = set(); seq: List[str] = []
        for q in arr:
            if q not in seen:
                seen.add(q); seq.append(q)
        queries_by_topic[t] = seq
    return {
        "gist": gist,
        "guide": {"prefer": plan.get("prefer") or prefer, "avoid": plan.get("avoid") or avoid},
        "section_tree": section_tree,
        "raw_md_present": bool(raw_md),
        "order": order,
        "queries_by_topic": queries_by_topic,
    }


def _filter_by_sections(nodes: List[NodeWithScore], prefer: List[str], avoid: List[str], top_k: int) -> List[NodeWithScore]:
    avoid_lower = [a.lower() for a in avoid]
    prefer_lower = [p.lower() for p in prefer]
    kept: List[NodeWithScore] = []
    for n in nodes:
        try:
            sec = ((n.node.metadata or {}).get("section") or "").lower()
        except Exception:
            sec = ""
        if any(a in sec or sec in a for a in avoid_lower if a):
            continue
        kept.append(n)
    if not kept:
        kept = list(nodes)
    for n in kept:
        try:
            sec = ((n.node.metadata or {}).get("section") or "").lower()
            if any(p in sec or sec in p for p in prefer_lower if p):
                n.score = (n.score or 0.0) * 1.2
        except Exception:
            pass
    kept.sort(key=lambda x: x.score or 0.0, reverse=True)
    return kept[:top_k]


# ---------------------------------------------
# D. Tool Library
# ---------------------------------------------

class ToolLibrary:
    def __init__(self, query_engine):
        self.query_engine = query_engine
        self._cache: Dict[Tuple[str, Optional[str], int], Dict[str, Any]] = {}
        self._node_visits: Dict[str, int] = {}
        self._node_meta: Dict[str, Dict[str, Any]] = {}

    def query_document(self, search_term: str, section: Optional[str] = None, top_k: int = 5, first_pass: bool = False) -> Dict[str, Any]:
        # simple in-memory cache per process
        ck = (str(search_term or ""), f"{section}|fp={first_pass}", int(top_k))
        if ck in self._cache:
            return self._cache[ck]
        # 1) Initial retrieval
        try:
            response = self.query_engine.query(search_term)
        except Exception as e:
            return {"results": [], "error": str(e)}
        nodes: List[NodeWithScore] = list(getattr(response, "source_nodes", []) or [])

        # 2) Deterministic section policy (skip on first pass): avoid references/related work unless query requires
        query_lc = (search_term or "").lower()
        require_related_refs = any(k in query_lc for k in ["related work", "reference", "references", "citation", "bibliograph", "acknowledg"])
        avoid_sections = ["related work", "references", "bibliograph", "acknowledg"]

        def section_of(n: NodeWithScore) -> str:
            try:
                return str((getattr(n.node, "metadata", {}) or {}).get("section", "") or "")
            except Exception:
                return ""

        def page_of(n: NodeWithScore):
            try:
                md = (getattr(n.node, "metadata", {}) or {})
                if not isinstance(md, dict):
                    return None
                pg = md.get("page_label")
                if pg is None:
                    pg = md.get("page")
                if isinstance(pg, (int, float)):
                    return int(pg)
                if isinstance(pg, str) and pg.isdigit():
                    return int(pg)
                return pg
            except Exception:
                return None

        if not first_pass and not require_related_refs:
            nodes = [n for n in nodes if not any(a in section_of(n).lower() for a in avoid_sections)] or nodes

        # 3) Optional hint: keep to a desired section if provided
        if section:
            try:
                filtered = [n for n in nodes if section.lower() in section_of(n).lower()]
                if filtered:
                    nodes = filtered
            except Exception:
                pass

        # 4) Neighbor-like expansion: add up to 2 extra hits from same section as top seeds
        try:
            seeds = nodes[:3]
            extra: List[NodeWithScore] = []
            if seeds:
                # fetch another batch and filter by seed sections
                resp2 = self.query_engine.query(search_term)
                cand2: List[NodeWithScore] = list(getattr(resp2, "source_nodes", []) or [])
                seed_secs = [section_of(s) for s in seeds]
                for ssec in seed_secs:
                    if not ssec:
                        continue
                    same_sec = [c for c in cand2 if section_of(c) == ssec]
                    for c in same_sec:
                        if len(extra) >= 6:
                            break
                        # dedupe by node id/text
                        try:
                            nid = getattr(getattr(c, 'node', None), 'node_id', None) or getattr(getattr(c, 'node', None), 'id_', None)
                        except Exception:
                            nid = None
                        if nid is not None:
                            seen = False
                            for existing in (nodes + extra):
                                try:
                                    eid = getattr(getattr(existing, 'node', None), 'node_id', None) or getattr(getattr(existing, 'node', None), 'id_', None)
                                except Exception:
                                    eid = None
                                if eid is not None and eid == nid:
                                    seen = True
                                    break
                            if seen:
                                continue
                        extra.append(c)
            if extra:
                nodes = nodes + extra
        except Exception:
            pass

        # 5) Soft preference boost for likely informative sections (skip on first pass)
        prefer = ["method", "approach", "experiment", "result", "evaluation", "dataset", "conclusion"]
        if not first_pass:
            for n in nodes:
                try:
                    sec = section_of(n).lower()
                    if any(p in sec for p in prefer):
                        n.score = (getattr(n, "score", 0.0) or 0.0) * 1.2
                except Exception:
                    pass

        # 5b) Visit-based boost: frequently revisited nodes get a small bump
        if not first_pass:
            for n in nodes:
                try:
                    nid = getattr(getattr(n, 'node', None), 'node_id', None) or getattr(getattr(n, 'node', None), 'id_', None)
                    if nid is not None:
                        cnt = int(self._node_visits.get(nid, 0))
                        if cnt > 0:
                            n.score = (getattr(n, "score", 0.0) or 0.0) * (1.0 + min(0.1 * cnt, 0.5))
                except Exception:
                    continue

        # 6) Truncate to top_k
        try:
            nodes.sort(key=lambda x: getattr(x, "score", 0.0) or 0.0, reverse=True)
        except Exception:
            pass
        nodes = nodes[:top_k]

        # 7) Serialize
        results: List[Dict[str, Any]] = []
        for n in nodes:
            txt = _get_content_safe(n)
            if not txt:
                continue
            try:
                nid = getattr(getattr(n, 'node', None), 'node_id', None) or getattr(getattr(n, 'node', None), 'id_', None)
            except Exception:
                nid = None
            # track visits and store minimal meta for canvas
            if nid is not None:
                self._node_visits[nid] = self._node_visits.get(nid, 0) + 1
                if nid not in self._node_meta:
                    try:
                        sec_name = section_of(n)
                    except Exception:
                        sec_name = ""
                    self._node_meta[nid] = {"section": sec_name, "sample": txt[:200]}
            results.append({
                "text": txt,
                "score": getattr(n, "score", None),
                "section": section_of(n),
                "page": page_of(n),
                "node_id": nid,
                "visits": self._node_visits.get(nid) if nid is not None else None,
            })
        out = {"results": results}
        self._cache[ck] = out
        return out

    def validate_answer(self, topic: str, candidate: str) -> Dict[str, Any]:
        verdict = {"supported": False, "evidence": [], "notes": ""}
        if not candidate or str(candidate).strip().lower() == "insufficient evidence":
            verdict["notes"] = "empty or insufficient"
            return verdict
        strong_sections = ["method", "experiment", "result", "evaluation"]
        hits_strong: List[Dict[str, Any]] = []
        hits_related: List[Dict[str, Any]] = []
        try:
            combo = f"{candidate} {topic}".strip()
            for sec_hint in ["Methods", "Experiments", "Results", None]:
                obs = self.query_document(combo, section=sec_hint, top_k=8)
                for r in obs.get("results", []):
                    sec = str(r.get("section", "")).lower()
                    if any(s in sec for s in ["related work", "reference", "bibliograph"]):
                        hits_related.append(r)
                    if any(s in sec for s in strong_sections):
                        hits_strong.append(r)
                if len(hits_strong) >= 2:
                    break
        except Exception:
            pass
        if hits_strong:
            verdict["supported"] = True
            verdict["evidence"] = hits_strong[:4]
            if hits_related:
                verdict["notes"] = "also appears in related work"
            return verdict
        try:
            obs2 = self.query_document(candidate, section="Methods", top_k=6)
            hits_strong.extend([r for r in obs2.get("results", [])])
        except Exception:
            pass
        verdict["supported"] = bool(hits_strong)
        verdict["evidence"] = hits_strong[:4]
        return verdict

    def finish(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        return {"final": extracted_data}


# ---------------------------------------------
# E. Agent State & Loop
# ---------------------------------------------

@dataclass
class AgentState:
    file_stem: str
    topics: List[str]
    gist: str
    guide: Dict[str, List[str]]
    living_canvas: str = ""
    canvas_history: List[str] = field(default_factory=list)
    found: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # topic -> {answer, confidence, evidence}
    log: List[Dict[str, Any]] = field(default_factory=list)
    max_steps: int = 24
    queries_by_topic: Dict[str, List[str]] = field(default_factory=dict)
    query_idx: Dict[str, int] = field(default_factory=dict)

    def add_log(self, thought: str, action: str, observation: Any) -> None:
        self.log.append({"thought": thought, "action": action, "observation": observation})

    def update_canvas(self) -> None:
        synopsis_parts: List[str] = ["GIST: " + self.gist]
        for t in self.topics:
            ent = self.found.get(t)
            if ent and ent.get("answer"):
                synopsis_parts.append(f"- {t}: {ent.get('answer')}")
        # Append a compact visited-nodes summary if available
        try:
            visited_lines: List[str] = []
            # Pull from ToolLibrary if present on this process (best-effort, optional)
            # We avoid tight coupling; the agent will re-create a short view from evidence
            recent_evidence: List[str] = []
            for t in self.topics:
                ev = (self.found.get(t) or {}).get("evidence") or []
                for e in ev[:2]:
                    sec = (e.get("section") or "").strip()
                    sample = (e.get("text") or "")[:120]
                    if sample:
                        visited_lines.append(f"  • [{sec}] {sample}")
            if visited_lines:
                synopsis_parts.append("Visited nodes (samples):")
                synopsis_parts.extend(visited_lines[:6])
        except Exception:
            pass
        self.living_canvas = "\n".join(synopsis_parts)
        if not self.canvas_history or self.canvas_history[-1] != self.living_canvas:
            self.canvas_history.append(self.living_canvas)


def _synthesize_answer(llm, topic: str, contexts: List[str]) -> str:
    ctx_text = "\n\n---\n\n".join(contexts[:10])
    prompt = f"""
        Using only the provided contexts, answer the topic concisely and directly.
        If the answer is not fully specified in the contexts, provide your best supported answer
        and briefly note uncertainty (one short clause). Do not invent facts not grounded in contexts.are 

        Topic: {topic}
        Contexts:
        {ctx_text}
    """.strip()
    return f"{llm.complete(prompt)!s}".strip()


def _decide_next_action(state: AgentState) -> Tuple[str, Dict[str, Any]]:
    # If all topics have answers, finish
    def _needs_more_work(t: str) -> bool:
        ent = state.found.get(t) or {}
        ans = str(ent.get("answer") or "").strip().lower()
        return (not ans) or (ans == "insufficient evidence")
    remaining = [t for t in state.topics if _needs_more_work(t)]
    if not remaining:
        return "finish", {"extracted_data": {t: state.found.get(t, {}) for t in state.topics}}
    # Pick next topic (first remaining)
    topic = remaining[0]
    # Choose next derived query phrase for that topic
    qlist = state.queries_by_topic.get(topic, [topic])
    idx = state.query_idx.get(topic, 0)
    if idx >= len(qlist):
        idx = 0
    search_term = qlist[idx]
    state.query_idx[topic] = idx + 1
    # Section hint by topic intent
    tl = topic.lower()
    if any(k in tl for k in ["model", "architecture", "approach", "method"]):
        section_hint = "Methods"
    elif any(k in tl for k in ["dataset", "data set", "corpus"]):
        section_hint = "Experiments"
    elif any(k in tl for k in ["metric", "score", "accuracy", "auc", "f1", "dice", "result"]):
        section_hint = "Results"
    elif any(k in tl for k in ["limitation", "drawback", "challenge", "future work"]):
        section_hint = "Conclusion"
    else:
        section_hint = None
    return "query_document", {"search_term": search_term, "section": section_hint, "_topic": topic}


def run_agent(file_stem: str, query_engine, topics: List[str], plan: Dict[str, Any], out_dir: Optional[str] = None) -> Dict[str, Any]:
    tools = ToolLibrary(query_engine)
    state = AgentState(
        file_stem=file_stem,
        topics=topics,
        gist=plan.get("gist", ""),
        guide=plan.get("guide", {"prefer": [], "avoid": []}),
        max_steps=24,
        queries_by_topic=plan.get("queries_by_topic", {}),
    )
    # Intent classifier for targeted retrieval
    def _intent_of(topic: str) -> str:
        tl = topic.lower()
        if any(k in tl for k in ["metric", "score", "accuracy", "auc", "f1", "dice", "result", "performance"]):
            return "metric"
        if any(k in tl for k in ["dataset", "data set", "corpus", "size"]):
            return "dataset"
        if any(k in tl for k in ["model", "architecture", "approach", "method", "algorithm"]):
            return "algorithm"
        if any(k in tl for k in ["feature", "signal"]):
            return "features"
        if any(k in tl for k in ["social", "twitter", "facebook", "youtube", "reddit"]):
            return "platform"
        return "other"

    def _preferred_section_for(intent: str) -> Optional[str]:
        return {
            "metric": "Results",
            "dataset": "Experiments",
            "algorithm": "Methods",
            "features": "Methods",
            "platform": "Data",
        }.get(intent)

    def _process_topic(topic: str, topic_idx: int) -> Tuple[str, Dict[str, Any]]:
        qlist = state.queries_by_topic.get(topic, [topic])
        search_term = qlist[0] if qlist else topic
        intent = _intent_of(topic)
        section_hint = _preferred_section_for(intent)
        params = {"search_term": search_term, "section": section_hint}
        # First pass: no rescoring to build initial feel
        obs_main = tools.query_document(**params, first_pass=True)
        params_b = dict(params); params_b["section"] = None
        obs_b = tools.query_document(**params_b, first_pass=True)
        def _uniq(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
        merged = _uniq((obs_main.get("results", []) or []) + (obs_b.get("results", []) or []))
        contexts = [r.get("text", "") for r in merged]
        answer = _synthesize_answer(Settings.llm, topic, contexts)
        evidence = merged[:5]
        verdict = tools.validate_answer(topic, answer)
        conf = 0.85 if verdict.get("supported") else (0.5 if answer else 0.2)
        merged_ev = (verdict.get("evidence") or [])[:4]
        if len(merged_ev) < 4:
            merged_ev.extend(evidence[: max(0, 4 - len(merged_ev))])
        # Iterative refinement with subquestions if uncertain
        tries = 0
        while conf < 0.75 and tries < 2:
            tries += 1
            anchors: List[str] = []
            # Pull anchors from already found answers to specialize the query
            for k, v in state.found.items():
                if not v or not v.get("answer"):
                    continue
                k_int = _intent_of(k)
                if k_int in ("algorithm", "dataset"):
                    anchors.append(str(v.get("answer"))[:80])
            # Generate simple subqueries
            subqs: List[str] = []
            if intent == "metric":
                subqs = [f"results {a}" for a in anchors] or ["results table", "achieved", topic]
            elif intent == "dataset":
                subqs = [f"dataset {a}" for a in anchors] or ["trained on", topic]
            elif intent == "algorithm":
                subqs = ["we propose", "our model", topic]
            else:
                subqs = [topic]
            # Intensify within preferred section with rescoring
            best_merge: List[Dict[str, Any]] = list(merged)
            for q in subqs[:2]:
                obs_sec = tools.query_document(q, section=section_hint, top_k=5, first_pass=False)
                best_merge = _uniq(best_merge + (obs_sec.get("results", []) or []))
            # Re-synthesize and re-validate
            contexts = [r.get("text", "") for r in best_merge]
            answer = _synthesize_answer(Settings.llm, topic, contexts)
            evidence = best_merge[:5]
            verdict = tools.validate_answer(topic, answer)
            conf = 0.85 if verdict.get("supported") else (0.5 if answer else 0.2)
            merged_ev = (verdict.get("evidence") or [])[:4]
            if len(merged_ev) < 4:
                merged_ev.extend(evidence[: max(0, 4 - len(merged_ev))])
        result = {"answer": answer, "confidence": conf, "evidence": merged_ev, "validated": verdict.get("supported", False)}
        if out_dir:
            try:
                with open(os.path.join(out_dir, f"topic_{topic_idx:02d}.json"), "w", encoding="utf-8") as f:
                    json.dump({"topic": topic, "result": result}, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
        return topic, result

    # Run all topics concurrently
    results_map: Dict[str, Dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=min(8, max(1, os.cpu_count() or 4))) as pool:
        futs = {pool.submit(_process_topic, t, idx): t for idx, t in enumerate(topics)}
        for fut in as_completed(futs):
            t = futs[fut]
            try:
                k, v = fut.result()
                results_map[k] = v
            except Exception:
                results_map[t] = {"answer": "", "confidence": 0.0, "evidence": [], "validated": False}
    state.found = dict(results_map)

    # Triangulation (after a brief wait for any missing answers)
    payload = {t: state.found.get(t, {}) for t in topics}
    missing = [k for k, v in payload.items() if not (v or {}).get("answer")]
    tries = 0
    while missing and tries < 3:
        time.sleep(1.0)
        for m in list(missing):
            cur = state.found.get(m)
            if cur and cur.get("answer"):
                payload[m] = cur
                missing.remove(m)
        tries += 1

    keys = [k for k, v in payload.items() if (v or {}).get("answer")]
    # Canvas-aware validation: use anchors from canvas for one more pass on low-confidence answers
    anchors = []
    for k in keys:
        k_int = _intent_of(k)
        if k_int in ("algorithm", "dataset"):
            anchors.append((k, (payload[k].get("answer") or "")[:80]))
    for k in keys:
        if (payload[k] or {}).get("confidence", 0.0) >= 0.75:
            continue
        intent = _intent_of(k)
        sec = _preferred_section_for(intent)
        for _, a in anchors:
            if not a:
                continue
            try:
                obs = tools.query_document(f"{k} {a}", section=sec, top_k=5)
                hits = obs.get("results", [])
                if hits:
                    payload[k]["confidence"] = max(0.75, float(payload[k].get("confidence") or 0.0))
                    break
            except Exception:
                pass
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            ai = (payload[keys[i]].get("answer") or "").strip()
            aj = (payload[keys[j]].get("answer") or "").strip()
            if ai and aj:
                try:
                    tri = tools.query_document(f"{ai} {aj}", section=None, top_k=5)
                    tri_hits = tri.get("results", [])
                    if tri_hits:
                        payload[keys[i]]["triangulated_with"] = keys[j]
                        payload[keys[j]]["triangulated_with"] = keys[i]
                except Exception:
                    pass

    return {"extracted_data": payload, "log": state.log, "gist": state.gist, "canvas_history": state.canvas_history}


# ---------------------------------------------
# F. Main Entrypoint
# ---------------------------------------------

def main():
    start = time.time()

    if API == "openrouter":
        print(f"Using OpenRouter execution model: {EXECUTION_MODEL} [REACT_HEAVYWEIGHT]")
        Settings.llm = OpenAILike(
            model=EXECUTION_MODEL,
            api_base="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
            is_chat_model=True,
        )
    elif API == "ollama":
        exec_model = OLLAMA_EXECUTION_MODEL or EXECUTION_MODEL
        print(f"Using Ollama execution model: {exec_model} @ {OLLAMA_BASE_URL} [REACT_HEAVYWEIGHT]")
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

    # Install token tracker callbacks
    tracker = TokenTracker()
    tracker.install()

    run_tag = "react_heavyweight"
    output_path = os.path.join(OUTPUT_PATH, f"{time.strftime('%Y.%m.%d_%H.%M.%S')}_{run_tag}")
    os.makedirs(output_path, exist_ok=True)

    files = [os.path.splitext(f)[0] for f in os.listdir(INPUT_PATH) if f.lower().endswith('.pdf')]
    topics = [q.get("topic", "") for q in QUERIES]

    def _process_file(file_stem: str) -> None:
        print(f"\nProcessing file: {file_stem}")
        query_engine = VectorQueryEngineCreator(
            llama_parse_api_key=LLAMA_CLOUD_API_KEY,
            cohere_api_key=COHERE_API_KEY,
            input_path=INPUT_PATH,
            storage_path=STORAGE_PATH,
            cohere_rerank=False,
            embedding_model_name=EMBEDDING_MODEL,
            response_mode="compact",
        ).get_query_engine(file_stem)
        plan = _plan_topics(Settings.llm, file_stem, topics)
        out_dir = os.path.join(output_path, file_stem)
        os.makedirs(out_dir, exist_ok=True)
        result = run_agent(file_stem, query_engine, topics, plan, out_dir=out_dir)
        with open(os.path.join(out_dir, f"{file_stem}_result.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    # up to 3 files concurrently
    with ThreadPoolExecutor(max_workers=3) as pool:
        futs = {pool.submit(_process_file, f): f for f in files}
        for _ in as_completed(futs):
            pass

    total = time.time() - start
    print("END [REACT_HEAVYWEIGHT]")
    print(f"Execution time: {total} seconds")
    # Persist usage once over the entire run
    tracker.write_report(output_path)
    rep = tracker.report()
    print(f"Token usage → LLM: {rep.get('total_llm_token_count')} | Embed: {rep.get('total_embedding_token_count')} | Total: {rep.get('total_token_count')}")
    print(f"Usage report: {os.path.join(output_path, 'usage.json')}")


if __name__ == "__main__":
    main()
