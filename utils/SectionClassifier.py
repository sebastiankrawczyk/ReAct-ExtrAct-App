import os
import re
from typing import Dict, List, Optional, Tuple

from llama_index.core import Settings
from llama_index.llms.openai_like import OpenAILike

from config.config import SMALL_MODEL
from config.config_keys import OPENROUTER_API_KEY


CanonicalCategory = str


def _strip_heading_markers(line: str) -> str:
    text = str(line or "").strip()
    # remove leading markdown bullets/headers
    text = re.sub(r"^\s*#+\s*", "", text)
    text = re.sub(r"^\s*[-*•]\s*", "", text)
    return text.strip(" #\t\r\n")


def _normalize_name(name: str) -> str:
    n = (name or "").strip().lower()
    n = re.sub(r"\s+", " ", n)
    return n


class SectionClassifier:
    """
    Classifies paper section headings into canonical categories and strength levels.

    - Canonical categories only (no strength):
      methods, experiments, results, evaluation, dataset, data, introduction, background,
      conclusion, discussion, related work, references, acknowledgments, appendix, limitations, approach, analysis.

    If a heading doesn't match rules, an optional small LLM fallback picks from the canonical categories.
    """

    def __init__(self, use_llm_fallback: bool = True):
        self.use_llm_fallback = use_llm_fallback
        self._ensure_small_llm()

        # keyword → canonical category
        self.alias_to_category: Dict[str, CanonicalCategory] = {
            # strong families
            "method": "methods",
            "methodology": "methods",
            "materials and methods": "methods",
            "experimental": "experiments",
            "experiment": "experiments",
            "evaluation": "evaluation",
            "result": "results",
            "approach": "approach",
            # neutral families
            "dataset": "dataset",
            "data set": "dataset",
            "data": "data",
            "introduction": "introduction",
            "background": "background",
            "conclusion": "conclusion",
            "discussion": "discussion",
            "analysis": "analysis",
            # avoid families
            "related work": "related work",
            "reference": "references",
            "bibliograph": "references",
            "acknowledg": "acknowledgments",
            "appendix": "appendix",
        }

        # strength mapping removed: planner will decide which categories are strong/weak per use case

    def _ensure_small_llm(self) -> None:
        # Configure a small LLM client once if requested
        if not self.use_llm_fallback:
            return
        try:
            # If Settings.llm is already configured elsewhere, we reuse it.
            if getattr(Settings, "llm", None) is None:
                Settings.llm = OpenAILike(
                    model=SMALL_MODEL,
                    api_base="https://openrouter.ai/api/v1",
                    api_key=OPENROUTER_API_KEY,
                    is_chat_model=True,
                )
        except Exception:
            # Fail soft; fallback classification will be skipped
            self.use_llm_fallback = False

    def _rule_category(self, heading_norm: str) -> Optional[CanonicalCategory]:
        for key, cat in self.alias_to_category.items():
            if key in heading_norm:
                return cat
        return None

    def _llm_category(self, heading: str) -> Optional[CanonicalCategory]:
        if not self.use_llm_fallback or getattr(Settings, "llm", None) is None:
            return None
        prompt = f"""
        You map a section heading to one canonical category from this set:
        [methods, experiments, results, evaluation, approach, dataset, data, introduction, background, conclusion,
         discussion, analysis, related work, references, acknowledgments, appendix].

        Heading: {heading}

        Respond with ONLY the category word from the set. If unsure, choose the closest.
        """.strip()
        try:
            raw = f"{Settings.llm.complete(prompt)!s}".strip().lower()
            raw = raw.strip().strip("` ")
            # keep only first token-like word if model returns extra
            raw = re.split(r"[^a-z ]+", raw)[0].strip()
            valid = set(self.category_to_strength.keys())
            return raw if raw in valid else None
        except Exception:
            return None

    def classify_heading(self, heading: str) -> Dict[str, str]:
        h = _strip_heading_markers(heading)
        hn = _normalize_name(h)
        cat = self._rule_category(hn)
        if cat is None:
            cat = self._llm_category(h) or "analysis"
        return {"raw": h, "category": cat}

    def classify_headings(self, headings: List[str]) -> List[Dict[str, str]]:
        return [self.classify_heading(h) for h in headings]

    def classify_tree_file(self, path: str) -> Dict[str, Dict[str, str]]:
        """
        Reads a markdown tree file (e.g., section_tree_md_only_[id]_adjusted.md) and returns
        a mapping: normalized heading → {raw, category, strength}.
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = [l.rstrip("\n") for l in f.readlines()]
        except Exception:
            return {}
        # Keep only non-empty lines; assume each is a heading-like entry
        headings = [l for l in lines if str(l or "").strip()]
        out: Dict[str, Dict[str, str]] = {}
        for h in headings:
            item = self.classify_heading(h)
            out[_normalize_name(item["raw"]) or item["raw"]] = item
        return out


