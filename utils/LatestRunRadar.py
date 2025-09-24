import os
import re
import json
import math
from typing import Dict, Any, List, Tuple

import plotly.graph_objects as go


OUTPUT_BASE = os.path.join('.', 'output')


EXCLUDE_KEYS = set()


def is_number(value: Any) -> bool:
    try:
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return False
        return True
    except Exception:
        return False


def find_latest_run_dir(base_dir: str) -> str | None:
    if not os.path.isdir(base_dir):
        return None
    entries: List[Tuple[float, str]] = []
    for name in os.listdir(base_dir):
        path = os.path.join(base_dir, name)
        if os.path.isdir(path):
            # Only consider timestamped run directories like YYYY.MM.DD_HH.MM.SS
            if name == 'aggregated' or not re.match(r'^\d{4}\.\d{2}\.\d{2}_\d{2}\.\d{2}\.\d{2}$', name):
                continue
            try:
                mtime = os.path.getmtime(path)
            except Exception:
                continue
            entries.append((mtime, path))
    if not entries:
        return None
    entries.sort(reverse=True)
    return entries[0][1]


def collect_samples(run_dir: str) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for file_dir in os.listdir(run_dir):
        sub_path = os.path.join(run_dir, file_dir)
        if not os.path.isdir(sub_path):
            continue
        result_path = os.path.join(sub_path, f"{file_dir}_result.json")
        if not os.path.isfile(result_path):
            continue
        try:
            with open(result_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            continue
        for r in data.get('results', []):
            ev = r.get('evaluation') or {}
            ragas = ev.get('ragas') if isinstance(ev, dict) else None
            geval = ev.get('geval') if isinstance(ev, dict) else None
            samples.append({'ragas': ragas if isinstance(ragas, dict) else {}, 'geval': geval if isinstance(geval, dict) else {}})
    return samples


def average_metrics(samples: List[Dict[str, Any]]) -> Dict[str, float]:
    if not samples:
        return {}
    # Target display metrics and their mapping from sources
    # Prefer RAGAS where available, fall back to Deepeval names
    target_map = {
        'Faithfulness': [('ragas', 'faithfulness')],
        'Semantic Similarity': [('ragas', 'semantic_similarity')],
        'Answer Correctness': [('ragas', 'answer_correctness')],
        'Context Precision': [('ragas', 'llm_context_precision_without_reference')],
    }
    sums: Dict[str, float] = {k: 0.0 for k in target_map.keys()}
    counts: Dict[str, int] = {k: 0 for k in target_map.keys()}
    # precompute contextual precision to derive noise sensitivity if not present
    cp_values: List[float] = []
    for s in samples:
        ragas = s.get('ragas', {})
        geval = s.get('geval', {})
        # sum known metrics
        for display_key, paths in target_map.items():
            val = None
            for source_name, metric_key in paths:
                source = ragas if source_name == 'ragas' else geval
                if metric_key in source and is_number(source[metric_key]):
                    val = float(source[metric_key])
                    break
            if val is not None:
                sums[display_key] += val
                counts[display_key] += 1
        # capture context precision candidate
        cp = None
        if is_number(ragas.get('llm_context_precision_without_reference')):
            cp = float(ragas['llm_context_precision_without_reference'])
        elif is_number(geval.get('contextual_precision')):
            cp = float(geval['contextual_precision'])
        if cp is not None:
            cp_values.append(cp)
    # remove any unused computed state
    # finalize averages clipped to [0,1]
    averages: Dict[str, float] = {}
    for k in target_map.keys():
        if counts[k] > 0:
            v = sums[k] / counts[k]
            if not math.isnan(v) and not math.isinf(v):
                averages[k] = max(0.0, min(1.0, float(v)))
    return averages


def render_radar(averages: Dict[str, float], out_png: str, title: str) -> None:
    if not averages:
        return
    metrics = list(averages.keys())
    values = [averages[m] for m in metrics]
    # Close the radar loop by repeating first value and metric
    metrics_closed = metrics + metrics[:1]
    values_closed = values + values[:1]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=values_closed, theta=metrics_closed, fill='toself', name='Average'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=False, title=title)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.write_image(out_png, scale=2)


def main() -> None:
    latest_dir = find_latest_run_dir(OUTPUT_BASE)
    if not latest_dir:
        print('No output run directory found.')
        return
    samples = collect_samples(latest_dir)
    averages = average_metrics(samples)
    # write under per-run aggregated subfolder
    agg_dir = os.path.join(latest_dir, 'aggregated')
    os.makedirs(agg_dir, exist_ok=True)
    out_json = os.path.join(agg_dir, 'run_ragas_average.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump({'averages': averages}, f, ensure_ascii=False, indent=4)
    out_png = os.path.join(agg_dir, 'run_ragas_average.png')
    render_radar(averages, out_png, title='RAGAS Average Metrics (Requested) - This Run')
    print(f'Latest run: {latest_dir}')
    print(f'Samples counted: {len(samples)}')
    print(f'Wrote: {out_json}')
    print(f'Wrote: {out_png}')


if __name__ == '__main__':
    main()


