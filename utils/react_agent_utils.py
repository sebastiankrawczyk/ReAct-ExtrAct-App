from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import os


# Lightweight planner guidance reused by ReAct-Extract
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

# Allow environment-based override or augmentation of planner heuristics without touching mains/UI.
# Two mechanisms:
# 1) PLANNER_HEURISTICS_OVERRIDE: full replacement text
# 2) HUMAN_HEURISTIC (+ HUMAN_HEURISTIC_TEXT): when enabled, append HUMAN_HEURISTIC_TEXT to defaults
try:
    override_text = os.getenv("PLANNER_HEURISTICS_OVERRIDE")
    if not override_text:
        flag = str(os.getenv("HUMAN_HEURISTIC") or "").strip().lower() in ("1", "true", "yes", "y", "on")
        extra = os.getenv("HUMAN_HEURISTIC_TEXT")
        if flag and extra:
            override_text = (PLANNER_HEURISTICS + "\n" + extra.strip()).strip()
    if override_text:
        PLANNER_HEURISTICS = override_text
except Exception:
    # If anything goes wrong, keep the default heuristics
    pass


def _synthesize_answer(llm: Any, topic: str, contexts: List[str]) -> str:
    ctx_text = "\n\n---\n\n".join(contexts[:10])
    prompt = f"""
        Using only the provided contexts, answer the topic concisely and directly.
        If the answer is not fully specified in the contexts, provide your best supported answer
        and briefly note uncertainty (one short clause). Do not invent facts not grounded in contexts.are 

        Topic: {topic}
        Contexts:
        {ctx_text}
    """.strip()
    try:
        return f"{llm.complete(prompt)!s}".strip()
    except Exception as e:
        try:
            print(f"[llm-error] synthesis failed: {e}")
        except Exception:
            pass
        return ""


@dataclass
class AgentState:
    file_stem: str
    topics: List[str]
    gist: str
    guide: Dict[str, List[str]]
    living_canvas: str = ""
    canvas_history: List[str] = field(default_factory=list)
    found: Dict[str, Dict[str, Any]] = field(default_factory=dict)
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
        try:
            visited_lines: List[str] = []
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


class ToolLibrary:
    def __init__(self, query_engine: Any) -> None:
        self.query_engine = query_engine
        self._cache: Dict[Tuple[str, Optional[str], int], Dict[str, Any]] = {}
        self._node_visits: Dict[str, int] = {}
        self._node_meta: Dict[str, Dict[str, Any]] = {}

    def _get_content_safe(self, n: Any) -> str:
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

    def _section_of(self, n: Any) -> str:
        try:
            return str((getattr(n.node, "metadata", {}) or {}).get("section", "") or "")
        except Exception:
            return ""

    def _page_of(self, n: Any) -> Optional[int]:
        try:
            md = (getattr(n, "node", None) or {}).metadata if hasattr(n, "node") else {}
            if not isinstance(md, dict):
                return None
            pg = md.get("page_label")
            if pg is None:
                pg = md.get("page")
            if isinstance(pg, (int, float)):
                return int(pg)
            if isinstance(pg, str) and pg.isdigit():
                return int(pg)
            return None
        except Exception:
            return None

    def query_document(self, search_term: str, section: Optional[str] = None, top_k: int = 5, first_pass: bool = False) -> Dict[str, Any]:
        ck = (str(search_term or ""), f"{section}|fp={first_pass}", int(top_k))
        if ck in self._cache:
            return self._cache[ck]
        try:
            response = self.query_engine.query(search_term)
        except Exception as e:
            return {"results": [], "error": str(e)}
        nodes: List[Any] = list(getattr(response, "source_nodes", []) or [])

        query_lc = (search_term or "").lower()
        require_related_refs = any(k in query_lc for k in ["related work", "reference", "references", "citation", "bibliograph", "acknowledg"])
        avoid_sections = ["related work", "references", "bibliograph", "acknowledg"]

        if not first_pass and not require_related_refs:
            nodes = [n for n in nodes if not any(a in self._section_of(n).lower() for a in avoid_sections)] or nodes

        if section:
            try:
                filtered = [n for n in nodes if section.lower() in self._section_of(n).lower()]
                if filtered:
                    nodes = filtered
            except Exception:
                pass

        try:
            seeds = nodes[:3]
            extra: List[Any] = []
            if seeds:
                resp2 = self.query_engine.query(search_term)
                cand2: List[Any] = list(getattr(resp2, "source_nodes", []) or [])
                seed_secs = [self._section_of(s) for s in seeds]
                for ssec in seed_secs:
                    if not ssec:
                        continue
                    same_sec = [c for c in cand2 if self._section_of(c) == ssec]
                    for c in same_sec:
                        if len(extra) >= 6:
                            break
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

        prefer = ["method", "approach", "experiment", "result", "evaluation", "dataset", "conclusion"]
        if not first_pass:
            for n in nodes:
                try:
                    sec = self._section_of(n).lower()
                    if any(p in sec for p in prefer):
                        n.score = (getattr(n, "score", 0.0) or 0.0) * 1.2
                except Exception:
                    pass

        try:
            nodes.sort(key=lambda x: getattr(x, "score", 0.0) or 0.0, reverse=True)
        except Exception:
            pass
        nodes = nodes[:top_k]

        results: List[Dict[str, Any]] = []
        for n in nodes:
            txt = self._get_content_safe(n)
            if not txt:
                continue
            try:
                nid = getattr(getattr(n, 'node', None), 'node_id', None) or getattr(getattr(n, 'node', None), 'id_', None)
            except Exception:
                nid = None
            if nid is not None:
                self._node_visits[nid] = self._node_visits.get(nid, 0) + 1
                if nid not in self._node_meta:
                    try:
                        sec_name = self._section_of(n)
                    except Exception:
                        sec_name = ""
                    self._node_meta[nid] = {"section": sec_name, "sample": txt[:200]}
            results.append({
                "text": txt,
                "score": getattr(n, "score", None),
                "section": self._section_of(n),
                "page": self._page_of(n),
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


