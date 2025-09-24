import os
import json
import math
from typing import Dict, List, Any

try:
    import plotly.graph_objects as go
except Exception:
    go = None
#s

OUTPUT_BASE = os.path.join('.', 'output')


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not (isinstance(value, float) and math.isnan(value))


def coerce_float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except Exception:
        return None


def collect_result_files(base_dir: str) -> List[Dict[str, str]]:
    collected: List[Dict[str, str]] = []
    if not os.path.isdir(base_dir):
        return collected
    for run_name in os.listdir(base_dir):
        run_path = os.path.join(base_dir, run_name)
        if not os.path.isdir(run_path):
            continue
        # each run contains subfolders named by file
        for file_dir in os.listdir(run_path):
            sub_path = os.path.join(run_path, file_dir)
            if not os.path.isdir(sub_path):
                continue
            result_json_path = os.path.join(sub_path, f"{file_dir}_result.json")
            if os.path.isfile(result_json_path):
                collected.append({
                    'run': run_name,
                    'file': file_dir,
                    'path': result_json_path,
                })
    return collected


def aggregate_samples(files: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for item in files:
        try:
            with open(item['path'], 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            continue
        results = data.get('results') or []
        for r in results:
            eval_obj = r.get('evaluation')
            if isinstance(eval_obj, dict) and eval_obj.get('ragas') and isinstance(eval_obj['ragas'], dict):
                ragas_metrics = eval_obj['ragas']
                sample: Dict[str, Any] = {
                    'run': item['run'],
                    'file': item['file'],
                    'topic': (r.get('query') or {}).get('topic')
                }
                for k, v in ragas_metrics.items():
                    sample[k] = coerce_float_or_none(v)
                samples.append(sample)
    return samples


def write_aggregated_dataset(samples: List[Dict[str, Any]], out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'aggregated_radar_dataset.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'samples': samples}, f, ensure_ascii=False, indent=4)
    return out_path


def get_metric_keys(samples: List[Dict[str, Any]]) -> List[str]:
    allowed = {
        'faithfulness',
        'semantic_similarity',
        'answer_correctness',
        'llm_context_precision_without_reference',
    }
    keys = set()
    for s in samples:
        for k, v in s.items():
            if k in ('run', 'file', 'topic'):
                continue
            if (is_number(v) or v is None) and k in allowed:
                keys.add(k)
    return sorted(list(keys))


def render_charts(samples: List[Dict[str, Any]], out_dir: str) -> None:
    if go is None or not samples:
        return
    charts_dir = os.path.join(out_dir, 'radar_charts_aggregated')
    os.makedirs(charts_dir, exist_ok=True)
    metric_keys = get_metric_keys(samples)
    # group by file
    by_file: Dict[str, List[Dict[str, Any]]] = {}
    for s in samples:
        by_file.setdefault(s['file'], []).append(s)
    for file_name, group in by_file.items():
        fig = go.Figure()
        theta = metric_keys
        for s in group:
            r_vals = []
            for k in metric_keys:
                val = s.get(k)
                r_vals.append(0 if (val is None or (isinstance(val, float) and math.isnan(val))) else float(val))
            name = f"{s.get('run', '-')}: {s.get('topic', '-') }"
            fig.add_trace(go.Scatterpolar(r=r_vals, theta=theta, fill='toself', name=name))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title=f"Aggregated RAGAS Metrics: {file_name}"
        )
        out_html = os.path.join(charts_dir, f"{file_name}.html")
        fig.write_html(out_html, include_plotlyjs='cdn')


def average_metrics(samples: List[Dict[str, Any]]) -> Dict[str, float]:
    metrics: Dict[str, List[float]] = {}
    for s in samples:
        for k, v in s.items():
            if k in ('run', 'file', 'topic'):
                continue
            fv = coerce_float_or_none(v)
            if fv is None:
                continue
            metrics.setdefault(k, []).append(fv)
    averages: Dict[str, float] = {}
    for k, vals in metrics.items():
        if vals:
            avg = sum(vals) / len(vals)
            if not math.isnan(avg) and not math.isinf(avg):
                # clip to [0,1]
                averages[k] = max(0.0, min(1.0, float(avg)))
    return averages


def render_aggregate_png(averages: Dict[str, float], out_path: str, title: str) -> None:
    if go is None or not averages:
        return
    metrics = list(averages.keys())
    values = [averages[m] for m in metrics]
    # close loop
    metrics_closed = metrics + metrics[:1]
    values_closed = values + values[:1]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=values_closed, theta=metrics_closed, fill='toself', name='Average'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=False, title=title)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.write_image(out_path, scale=2)


def main() -> None:
    files = collect_result_files(OUTPUT_BASE)
    samples = aggregate_samples(files)
    out_dir = os.path.join(OUTPUT_BASE, 'aggregated')
    dataset_path = write_aggregated_dataset(samples, out_dir)
    render_charts(samples, out_dir)
    # also write aggregate PNG across all runs/files
    averages = average_metrics(samples)
    avg_json_path = os.path.join(out_dir, 'aggregated_ragas_average.json')
    with open(avg_json_path, 'w', encoding='utf-8') as f:
        json.dump({'averages': averages}, f, ensure_ascii=False, indent=4)
    png_path = os.path.join(out_dir, 'aggregated_ragas_average.png')
    render_aggregate_png(averages, png_path, title='Aggregated RAGAS Metrics (All Runs)')
    print(f"Aggregated samples: {len(samples)}")
    print(f"Dataset: {dataset_path}")
    charts_dir = os.path.join(out_dir, 'radar_charts_aggregated')
    if os.path.isdir(charts_dir):
        print(f"Charts dir: {charts_dir}")
    if os.path.isfile(png_path):
        print(f"Aggregate PNG: {png_path}")


if __name__ == '__main__':
    main()


