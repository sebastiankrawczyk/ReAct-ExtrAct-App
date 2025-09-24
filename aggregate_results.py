import os
import re
import json
import argparse
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd

from config.queries import QUERIES


def _list_run_dirs(output_root: str) -> List[str]:
    if not os.path.isdir(output_root):
        return []
    return [d for d in os.listdir(output_root) if os.path.isdir(os.path.join(output_root, d))]


def _latest_run_dir(output_root: str, suffix: str) -> str | None:
    candidates = [d for d in _list_run_dirs(output_root) if d.endswith(suffix)]
    if not candidates:
        return None
    # Directories are prefixed by timestamp YYYY.MM.DD_HH.MM.SS_ so lexicographic sort works
    candidates.sort(reverse=True)
    return os.path.join(output_root, candidates[0])


def _read_results_from_run(run_dir: str) -> Dict[str, Dict[str, str]]:
    """
    Return mapping: paper -> { question_topic -> answer_text }
    """
    paper_to_q_to_ans: Dict[str, Dict[str, str]] = {}
    if not run_dir or not os.path.isdir(run_dir):
        return paper_to_q_to_ans
    for paper_dir in os.listdir(run_dir):
        pdir = os.path.join(run_dir, paper_dir)
        if not os.path.isdir(pdir):
            continue
        # Expect file like <paper>/<paper>_result.json
        base = paper_dir
        result_path = os.path.join(pdir, f"{base}_result.json")
        if not os.path.isfile(result_path):
            # Try to find any *_result.json file
            alt = [f for f in os.listdir(pdir) if f.endswith("_result.json")]
            if alt:
                result_path = os.path.join(pdir, alt[0])
            else:
                continue
        try:
            with open(result_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        results = data.get("results") or []
        q_to_ans: Dict[str, str] = {}
        for r in results:
            topic = None
            # Prefer 'query': {'topic': ...}
            try:
                topic = (r.get("query") or {}).get("topic")
            except Exception:
                topic = None
            # Fallback to 'question'
            if not topic:
                topic = r.get("question")
            if not isinstance(topic, str) or not topic.strip():
                continue
            answer = r.get("answer") if isinstance(r.get("answer"), str) else ""
            q_to_ans[topic] = answer
        paper_to_q_to_ans[paper_dir] = q_to_ans
    return paper_to_q_to_ans


def _sanitize_filename(name: str) -> str:
    s = re.sub(r"\s+", "_", name.strip())
    s = re.sub(r"[^A-Za-z0-9_\-]+", "", s)
    return s[:140] or "question"


def aggregate_latest(output_root: str, dest_dir: str | None = None) -> Tuple[str, List[str]]:
    """
    Build per-question CSVs with rows=papers and columns=[naive,guided,react].
    Returns (destination_directory, list_of_csv_paths).
    """
    latest_baseline = _latest_run_dir(output_root, "_baseline")
    latest_guided = _latest_run_dir(output_root, "_guided")
    latest_react = _latest_run_dir(output_root, "_react")

    col_to_run = {
        "naive": latest_baseline,
        "guided": latest_guided,
        "react": latest_react,
    }

    if dest_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest_dir = os.path.join(output_root, f"aggregates_{ts}")
    os.makedirs(dest_dir, exist_ok=True)

    # Load mappings for each run
    run_answers: Dict[str, Dict[str, Dict[str, str]]] = {}
    for col, rdir in col_to_run.items():
        run_answers[col] = _read_results_from_run(rdir) if rdir else {}

    # Known topics from config to ensure alignment and stable ordering
    topics = [q.get("topic") for q in QUERIES if isinstance(q, dict) and q.get("topic")]
    csv_paths: List[str] = []

    for topic in topics:
        # Collect union of papers across runs for this topic
        papers = set()
        for col in ("naive", "guided", "react"):
            for paper, qmap in run_answers[col].items():
                if topic in qmap:
                    papers.add(paper)
        if not papers:
            continue
        sorted_papers = sorted(papers)

        rows = []
        for paper in sorted_papers:
            row = {"paper": paper}
            for col in ("naive", "guided", "react"):
                ans = run_answers[col].get(paper, {}).get(topic, "")
                row[col] = ans
            rows.append(row)

        df = pd.DataFrame(rows).set_index("paper")
        fname = f"answers_{_sanitize_filename(topic)}.csv"
        out_path = os.path.join(dest_dir, fname)
        df.to_csv(out_path)
        csv_paths.append(out_path)

    return dest_dir, csv_paths


def main():
    parser = argparse.ArgumentParser(description="Aggregate latest baseline/guided/react answers into per-question CSVs.")
    parser.add_argument("--output-root", default=os.path.join(".", "output"), help="Root output directory containing run folders")
    parser.add_argument("--dest-dir", default=None, help="Destination directory for aggregated CSVs")
    args = parser.parse_args()

    dest, files = aggregate_latest(args.output_root, args.dest_dir)
    print(f"Aggregated CSVs written to: {dest}")
    for p in files:
        print(f" - {p}")


if __name__ == "__main__":
    main()


