import csv
import json
import os
from typing import Any, Dict, List, Optional, Tuple

try:
    from langchain_openai import ChatOpenAI
except Exception:
    ChatOpenAI = None  # type: ignore


def _default_csv_path() -> str:
    return os.path.join(os.path.dirname(__file__), "..", "eval", "Annotation JCDL - Seba eval.csv")


def parse_seba_csv(csv_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Parse the Seba eval CSV to extract binary criteria definitions.

    Returns a dict with:
      - criteria: List[Dict[name, question, yes_means, no_means, ragas_key?]]
      - sample: Optional example block if present (question, human_answer, ai_answer, labels)
    """
    path = csv_path or _default_csv_path()
    if not os.path.exists(path):
        return {"criteria": [], "sample": None}

    criteria: List[Dict[str, str]] = []
    sample: Dict[str, Any] = {"question": None, "human_answer": None, "ai_answer": None, "labels": {}}

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    # Phase 1: top criteria section (has headers: a Criterion, Question for Annotator, Yes Means, No Means)
    # Collect contiguous rows under this header until a blank delimiter.
    header_found = False
    for idx, row in enumerate(rows):
        if not row:
            continue
        first = (row[0] or "").strip()
        if not header_found and first.lower().endswith("criterion"):
            header_found = True
            continue
        if header_found:
            # Break if we hit an empty section separator (many empties) or the 'where i=' marker
            if all((c or "").strip() == "" for c in row) or first.lower().startswith("where i="):
                break
            name = (row[0] or "").strip()
            question = (row[1] or "").strip()
            yes_means = (row[2] or "").strip()
            no_means = (row[3] or "").strip()
            ragas_key = None
            # Extract optional RAGAS key if provided in parentheses in the name
            if "(" in name and ")" in name:
                base, rest = name.split("(", 1)
                name = base.strip()
                inner = rest.split(")", 1)[0]
                if ":" in inner:
                    _, ragas_key = inner.split(":", 1)
                    ragas_key = (ragas_key or "").strip()
            criteria.append({
                "name": name,
                "question": question,
                "yes_means": yes_means,
                "no_means": no_means,
                "ragas_key": ragas_key or "",
            })

    # Phase 2: try to parse the example block if present (Human Answer / AI Answer then binary labels)
    # Find the line that starts with ',Human Answer'
    for i, row in enumerate(rows):
        if len(row) >= 2 and (row[0] or "").strip() == "" and (row[1] or "").strip().lower() == "human answer":
            sample["human_answer"] = (row[2] or "").strip()
            # next row should be AI Answer
            if i + 1 < len(rows):
                ai_row = rows[i + 1]
                if len(ai_row) >= 3 and (ai_row[1] or "").strip().lower().startswith("ai answer"):
                    sample["ai_answer"] = (ai_row[2] or "").strip()
            # After that, collect label rows until blank line
            j = i + 2
            while j < len(rows):
                r = rows[j]
                if not r or all((c or "").strip() == "" for c in r):
                    break
                label_name = (r[1] or r[0] or "").strip()
                val_raw = (r[2] if len(r) > 2 else r[1] if len(r) > 1 else "").strip()
                if label_name and val_raw in {"0", "1"}:
                    sample["labels"][label_name] = (val_raw == "1")
                j += 1
            break

    return {"criteria": criteria, "sample": sample if sample["human_answer"] else None}


class SebaEvaluator:
    """
    Configurable evaluator that grades AI answers against ground truth using
    binary criteria defined in the Seba CSV. Uses an LLM judge to return
    boolean decisions per criterion plus brief justifications.

    Usage:
      from utils.SebaEvaluator import SebaEvaluator, parse_seba_csv
      defs = parse_seba_csv()
      evaluator = SebaEvaluator(model="qwen/qwen-turbo", api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1", criteria=defs["criteria"]) 
      result = evaluator.evaluate_sample(
          query_text="Algorithm: What ai algorithm was used?",
          ground_truth="Naive Bayes, SVM, Decision Tree, Random Forest, K-nearest neighbours and ensamble methods (bagging and boosting)",
          ai_answer="The following machine learning algorithms were used: Support Vector Machine (SVM), naïve Bayes’, decision tree, random forest, K Nearest Neighbors (KNN), and ensemble classification methods (with bagging and boosting).",
          contexts=[],
      )
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str],
        base_url: Optional[str] = None,
        criteria: Optional[List[Dict[str, str]]] = None,
    ) -> None:
        if ChatOpenAI is None:
            raise ImportError("langchain_openai is required for SebaEvaluator")
        self.model = model
        self.api_key = api_key
        self.base_url = base_url or "https://openrouter.ai/api/v1"
        # Load criteria and drop ones we don't want to evaluate
        loaded_criteria = criteria or parse_seba_csv().get("criteria", [])
        def _keep(c: Dict[str, str]) -> bool:
            n = (c.get("name") or "").strip().lower()
            return not (n.startswith("source locatability") or n.startswith("extraneous information"))
        self.criteria = [c for c in loaded_criteria if _keep(c)]
        self._llm = ChatOpenAI(
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=0,
            max_tokens=512,
            response_format={"type": "json_object"},
        )

    def _build_prompt(self, query_text: str, ground_truth: str, ai_answer: str, contexts: Optional[List[str]]) -> List[Dict[str, str]]:
        """Create a JSON-only instruction prompt for the LLM judge."""
        criteria_instructions = []
        for c in self.criteria:
            criteria_instructions.append({
                "name": c.get("name", ""),
                "question": c.get("question", ""),
                "yes_means": c.get("yes_means", ""),
                "no_means": c.get("no_means", ""),
            })

        sys_msg = (
            "You are a strict evaluator. Return a compact JSON object with boolean keys for each criterion name, "
            "a 'justifications' object mapping criterion to a short reason, and an overall 'verdict' in {correct, partially_correct, incorrect} "
            "plus 'verdict_reason'. Only output valid JSON. Base judgments on the provided source contexts if present; "
            "otherwise, treat the human reference answer as the authoritative reference. Avoid speculation. "
            "Do NOT require answers to be verbatim copies: judge semantic alignment. If the core intent is satisfied (e.g., the same social platform, the same set of ML models, or equivalent paraphrases), mark as correct unless other criteria fail."
        )
        user_payload = {
            "query": query_text,
            "reference_human_answer": ground_truth,
            "ai_answer": ai_answer,
            "source_contexts": contexts or [],
            "criteria": criteria_instructions,
            "output_schema": {
                c.get("name", ""): "boolean" for c in self.criteria
            },
            "verdict_rules": {
                "incorrect_if_any": [
                    "Presence == false",
                    "Relevance / Answerable == false",
                    "Factual Accuracy / Correctness == false",
                    "Hallucination Detection == true",
                ],
                "correct_if_all": [
                    "Presence == true",
                    "Relevance / Answerable == true",
                    "Factual Accuracy / Correctness == true",
                    "Hallucination Detection == false",
                    "Completeness == true",
                ],
                "otherwise": "partially_correct"
            }
        }
        return [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ]

    def evaluate_sample(
        self,
        query_text: str,
        ground_truth: str,
        ai_answer: str,
        contexts: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        # Fast exact-match safeguard: if normalized answers match, short-circuit as correct
        def _norm(s: str) -> str:
            return ' '.join((s or '').strip().lower().replace('\n', ' ').replace('\t', ' ').split())
        if _norm(ai_answer) == _norm(ground_truth) and _norm(ai_answer) != '':
            result: Dict[str, Any] = {}
            for c in self.criteria:
                name = c.get("name", "")
                val: Optional[bool] = None
                if name.lower().startswith('presence'):
                    val = True
                elif name.lower().startswith('relevance'):
                    val = True
                elif name.lower().startswith('factual accuracy'):
                    val = True
                elif name.lower().startswith('hallucination'):
                    val = False
                elif name.lower().startswith('completeness'):
                    val = True
                # leave others None
                result[name] = val
            result["justifications"] = {"Factual Accuracy / Correctness": "Exact match with human answer."}
            result["verdict"] = "correct"
            result["verdict_reason"] = "Exact match with human answer."
            return result

        messages = self._build_prompt(query_text, ground_truth, ai_answer, contexts)
        raw = self._llm.invoke(messages)
        try:
            data = raw.content if hasattr(raw, "content") else (raw["content"] if isinstance(raw, dict) else str(raw))
            result = json.loads(data)
        except Exception:
            # Fallback: return empty booleans
            result = {c.get("name", ""): None for c in self.criteria}
            result["justifications"] = {}
        # Ensure verdict derived if missing
        if "verdict" not in result:
            # normalize map
            def getb(name: str) -> Optional[bool]:
                for k, v in result.items():
                    if isinstance(v, bool) and k.lower().startswith(name.lower()):
                        return v
                return None
            presence = getb("Presence")
            relevance = getb("Relevance / Answerable")
            factual = getb("Factual Accuracy / Correctness")
            halluc = getb("Hallucination Detection")
            complete = getb("Completeness")
            verdict = "partially_correct"
            if (presence is False) or (relevance is False) or (factual is False) or (halluc is True):
                verdict = "incorrect"
            elif (presence is True) and (relevance is True) and (factual is True) and (halluc is False) and (complete is True):
                verdict = "correct"
            result["verdict"] = verdict
            result["verdict_reason"] = result.get("justifications", {}).get("Factual Accuracy / Correctness", "")
        return result


