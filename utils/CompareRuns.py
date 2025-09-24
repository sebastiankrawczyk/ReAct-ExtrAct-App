import os
import json
import csv
import time
from typing import Dict, List, Any, Tuple

try:
    import plotly.graph_objects as go
except Exception:
    go = None

from config.queries import QUERIES
from eval.ground_truth_dual import get_dual, GROUND_TRUTH_DUAL


OUTPUT_BASE = os.path.join('.', 'output')

# Target SEBA criteria to aggregate (ordered)
CRITERIA = [
    'Factual Accuracy / Correctness',
    'Hallucination Detection',
    'Completeness',
    'Relevance / Answerable',
    'Presence',
]


def _latest_run_with_suffix(suffix: str) -> str | None:
    candidates = []
    if not os.path.isdir(OUTPUT_BASE):
        return None
    for name in os.listdir(OUTPUT_BASE):
        if name.endswith(suffix) and os.path.isdir(os.path.join(OUTPUT_BASE, name)):
            p = os.path.join(OUTPUT_BASE, name)
            try:
                mtime = os.path.getmtime(p)
            except Exception:
                mtime = 0
            candidates.append((mtime, name))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def _run_has_results(run_name: str) -> bool:
    base_path = os.path.join(OUTPUT_BASE, run_name)
    if not os.path.isdir(base_path):
        return False
    try:
        for file_dir in os.listdir(base_path):
            sub = os.path.join(base_path, file_dir)
            if not os.path.isdir(sub):
                continue
            bl = os.path.join(sub, f"{file_dir}_baseline_like.json")
            if os.path.isfile(bl):
                try:
                    with open(bl, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    if isinstance(data, list) and len(data) > 0:
                        return True
                except Exception:
                    pass
            jf = os.path.join(sub, f"{file_dir}_result.json")
            if os.path.isfile(jf):
                try:
                    with open(jf, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    if isinstance(data, dict):
                        if (isinstance(data.get('results'), list) and len(data.get('results')) > 0) or (isinstance(data.get('extracted_data'), dict) and len(data.get('extracted_data')) > 0):
                            return True
                except Exception:
                    pass
    except Exception:
        return False
    return False


def _latest_nonempty_run_with_suffix(suffix: str) -> str | None:
    candidates: List[Tuple[float, str]] = []
    if not os.path.isdir(OUTPUT_BASE):
        return None
    for name in os.listdir(OUTPUT_BASE):
        if name.endswith(suffix) and os.path.isdir(os.path.join(OUTPUT_BASE, name)):
            p = os.path.join(OUTPUT_BASE, name)
            try:
                mtime = os.path.getmtime(p)
            except Exception:
                mtime = 0
            candidates.append((mtime, name))
    candidates.sort(reverse=True)
    for _, run_name in candidates:
        if _run_has_results(run_name):
            return run_name
    return candidates[0][1] if candidates else None


def _load_results(run_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    """Load results for a run, preferring baseline_like schema when available.

    Returns mapping: file_stem -> list[baseline_like entries]
    """
    out: Dict[str, List[Dict[str, Any]]] = {}
    base_path = os.path.join(OUTPUT_BASE, run_dir)
    if not os.path.isdir(base_path):
        return out
    for file_dir in os.listdir(base_path):
        sub = os.path.join(base_path, file_dir)
        if not os.path.isdir(sub):
            continue
        # Prefer baseline_like.json for uniform comparison across pipelines
        bl = os.path.join(sub, f"{file_dir}_baseline_like.json")
        if os.path.isfile(bl):
            try:
                with open(bl, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    out[file_dir] = data
                    continue
            except Exception:
                pass
        # Fallback to native result structure
        jf = os.path.join(sub, f"{file_dir}_result.json")
        if os.path.isfile(jf):
            try:
                with open(jf, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # normalize to baseline_like list if needed
                if isinstance(data, dict) and isinstance(data.get('results'), list):
                    out[file_dir] = data.get('results') or []
                elif isinstance(data, dict) and isinstance(data.get('extracted_data'), dict):
                    # react_* pipelines: map topic->entry to baseline_like entries
                    entries: List[Dict[str, Any]] = []
                    topic_to_options = {q.get('topic', ''): q.get('possible_options', 'None') for q in QUERIES}
                    for topic, ent in (data.get('extracted_data') or {}).items():
                        best_ctx = ent.get('best_context') or []
                        if isinstance(best_ctx, dict):
                            best_ctx = [best_ctx]
                        best_context_serialized = []
                        for c in best_ctx[:5]:
                            best_context_serialized.append({
                                'context': (c.get('context') or c.get('text') or ''),
                                'score': c.get('score'),
                                'page': c.get('page'),
                                'section': c.get('section'),
                            })
                        entries.append({
                            'query': {'topic': topic, 'possible_options': topic_to_options.get(topic, 'None')},
                            'question': topic,
                            'answer': ent.get('answer', ''),
                            'answer_concise': ent.get('concise_answer', ''),
                            'code': ent.get('code', ''),
                            'best_context': best_context_serialized,
                            'evaluation': ent.get('evaluation'),
                        })
                    # Keep original QUERIES order where possible
                    order = {QUERIES[i].get('topic'): i for i in range(len(QUERIES))}
                    entries.sort(key=lambda r: order.get(r.get('question', ''), 0))
                    out[file_dir] = entries
                else:
                    out[file_dir] = []
            except Exception:
                continue
    return out


def _pick_gt(file_key: str, idx: int, topic: str | None = None) -> str:
    def _norm(s: str) -> str:
        return ''.join(ch for ch in s.lower() if ch.isalnum())

    # Try exact key first
    dual = get_dual(file_key)
    # If empty, try fuzzy match over known GT keys
    if not (dual.get('human_answer') or dual.get('questions')):
        best_key = None
        best_score = -1
        fk_norm = _norm(file_key)
        fk_tokens = set(''.join(ch if ch.isalnum() else ' ' for ch in file_key.lower()).split())
        for k in (GROUND_TRUTH_DUAL or {}).keys():
            kn = _norm(k)
            # basic similarity: token overlap + prefix bonus
            ktoks = set(''.join(ch if ch.isalnum() else ' ' for ch in k.lower()).split())
            overlap = len(fk_tokens & ktoks)
            score = overlap
            if kn.startswith(fk_norm) or fk_norm.startswith(kn):
                score += 3
            if score > best_score:
                best_score = score
                best_key = k
        if best_key:
            dual = get_dual(best_key)
    q = QUERIES[idx]
    possible = str(q.get('possible_options', 'none')).lower()
    # Always use human_answer now per latest policy
    arr = dual.get('human_answer') or []
    # Prefer mapping by topic text if provided and available in dual['questions']
    if topic:
        questions = dual.get('questions') or []
        try:
            j = questions.index(topic)
            return (arr[j] if j < len(arr) else '') or ''
        except ValueError:
            pass
    return (arr[idx] if idx < len(arr) else '') or ''


def _extract_texts(one: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
    # Prefer natural language answer; fallback to code if answer empty
    ans = one.get('answer') or one.get('code') or ''
    concise = one.get('answer_concise') or ''
    evaluation = one.get('evaluation') or {}
    return str(ans), str(concise), (evaluation if isinstance(evaluation, dict) else {})


def _is_correct(evaluation: Dict[str, Any]) -> bool | None:
    # Prefer SEBA criterion: "Factual Accuracy / Correctness"
    seba = evaluation.get('seba')
    if isinstance(seba, dict):
        # try to find by key
        for k, v in seba.items():
            if isinstance(v, bool) and k.lower().startswith('factual accuracy'):
                return v
        # or if seba contains nested 'criteria' mapping
    # Fallback to RAGAS answer_correctness >= 0.5
    ragas = evaluation.get('ragas')
    if isinstance(ragas, dict):
        val = ragas.get('answer_correctness')
        try:
            if val is not None and float(val) >= 0.5:
                return True
            if val is not None:
                return False
        except Exception:
            pass
    return None


def build_comparison() -> str:
    baseline_run = _latest_nonempty_run_with_suffix('_baseline')
    iter_run = _latest_nonempty_run_with_suffix('_iter_retgen')
    meetings_run = _latest_nonempty_run_with_suffix('_react_meetings')
    extract_run = _latest_nonempty_run_with_suffix('_react_extract')
    if not baseline_run or not iter_run:
        raise RuntimeError('Missing baseline or iter_retgen run in output')
    base_results = _load_results(baseline_run)
    iter_results = _load_results(iter_run)
    meet_results = _load_results(meetings_run) if meetings_run else {}
    extr_results = _load_results(extract_run) if extract_run else {}
    files = sorted(set(base_results.keys()) & set(iter_results.keys()) & (set(meet_results.keys()) if meet_results else set(base_results.keys())) & (set(extr_results.keys()) if extr_results else set(base_results.keys())))

    ts = time.strftime('%Y.%m.%d_%H.%M.%S')
    out_dir = os.path.join(OUTPUT_BASE, 'aggregated', f'compare_{ts}')
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'comparison.csv')
    html_path = os.path.join(out_dir, 'comparison.html')

    headers = [
        'file', 'question', 'ground_truth',
        'baseline_answer', 'baseline_concise',
        'iter_answer', 'iter_concise',
        'react_meetings', 'meetings_concise',
        'react_extract', 'extract_concise',
    ]
    rows: List[List[str]] = []

    for file_key in files:
        b_list = base_results.get(file_key) or []
        i_list = iter_results.get(file_key) or []
        m_list = meet_results.get(file_key) or []
        e_list = extr_results.get(file_key) or []
        # map by topic text to align
        topic_to_b = {((r.get('query') or {}).get('topic') or r.get('question') or ''): r for r in b_list}
        topic_to_i = {((r.get('query') or {}).get('topic') or r.get('question') or ''): r for r in i_list}
        topic_to_m = {((r.get('query') or {}).get('topic') or r.get('question') or ''): r for r in m_list}
        topic_to_e = {((r.get('query') or {}).get('topic') or r.get('question') or ''): r for r in e_list}
        for idx, q in enumerate(QUERIES):
            topic = q.get('topic') or ''
            b = topic_to_b.get(topic, {})
            it = topic_to_i.get(topic, {})
            mt = topic_to_m.get(topic, {})
            et = topic_to_e.get(topic, {})
            gt = _pick_gt(file_key, idx, topic)
            b_ans, b_con, b_eval = _extract_texts(b)
            i_ans, i_con, i_eval = _extract_texts(it)
            m_ans, m_con, m_eval = _extract_texts(mt)
            e_ans, e_con, e_eval = _extract_texts(et)
            rows.append([file_key, topic, gt, b_ans, b_con, i_ans, i_con, m_ans, m_con, e_ans, e_con])

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    # HTML with color coding for correctness
    def td(text: str, cls: str = '') -> str:
        cl = f' class="{cls}"' if cls else ''
        return f'<td{cl}>{(text or "").replace("&", "&amp;").replace("<", "&lt;")}</td>'

    styles = (
        '<style>table{border-collapse:collapse;font-family:system-ui,Arial,sans-serif;font-size:14px}'
        'th,td{border:1px solid #ddd;padding:6px 8px;cursor:default}th{background:#fafafa;position:sticky;top:0}'
        '.mono{font-family:ui-monospace,Menlo,monospace}.gt{background:#fffef0}'
        '.state-ok{background:#e9f7ef}.state-bad{background:#fdecea}.state-none{background:#ffffff}'
        'td.toggle{cursor:pointer;user-select:none}'
        '.muted{color:#555}'
        '</style>'
    )
    html = ['<html><head>', styles, '</head><body>']
    html.append(f'<h2>Comparison: {baseline_run} vs {iter_run}' + (f' vs {meetings_run}' if meetings_run else '') + (f' vs {extract_run}' if extract_run else '') + '</h2>')
    html.append('<table>')
    html.append('<tr>' + ''.join(f'<th>{h}</th>' for h in headers) + '</tr>')
    for ridx, r in enumerate(rows):
        cells = [
            td(r[0]),
            td(r[1]),
            td(r[2], 'gt mono'),
            f'<td class="toggle state-none mono" data-col="baseline" data-row="{ridx}">{r[3]}</td>',
            td(r[4], 'mono muted'),
            f'<td class="toggle state-none mono" data-col="iter" data-row="{ridx}">{r[5]}</td>',
            td(r[6], 'mono muted'),
        ]
        # meetings and extract columns
        cells.append(f'<td class="toggle state-none mono" data-col="react_meetings" data-row="{ridx}">{r[7]}</td>')
        cells.append(td(r[8], 'mono muted'))
        cells.append(f'<td class="toggle state-none mono" data-col="react_extract" data-row="{ridx}">{r[9]}</td>')
        cells.append(td(r[10], 'mono muted'))
        html.append('<tr>' + ''.join(cells) + '</tr>')
    html.append('</table>')

    # Add a small script to toggle cell state: none -> green -> red -> none
    html.append('<script>(function(){\n'
                'function cycle(td){\n'
                ' if(td.classList.contains("state-none")){td.classList.remove("state-none");td.classList.add("state-ok");}\n'
                ' else if(td.classList.contains("state-ok")){td.classList.remove("state-ok");td.classList.add("state-bad");}\n'
                ' else if(td.classList.contains("state-bad")){td.classList.remove("state-bad");td.classList.add("state-none");}\n'
                '}\n'
                'document.querySelectorAll("td.toggle").forEach(function(td){\n'
                ' td.addEventListener("click", function(){cycle(td);});\n'
                '});\n'
                '})();</script>')

    # Aggregate verdict bars per run
    def summarize_verdicts(run_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, int]:
        counts = {"correct": 0, "partially_correct": 0, "incorrect": 0}
        for lst in run_results.values():
            for r in lst:
                seba = (r.get('evaluation') or {}).get('seba')
                if isinstance(seba, dict):
                    v = str(seba.get('verdict') or '').lower()
                    if v in counts:
                        counts[v] += 1
        return counts

    base_counts = summarize_verdicts(base_results)
    iter_counts = summarize_verdicts(iter_results)
    meet_counts = summarize_verdicts(meet_results) if meet_results else {"correct":0,"partially_correct":0,"incorrect":0}
    extr_counts = summarize_verdicts(extr_results) if extr_results else {"correct":0,"partially_correct":0,"incorrect":0}

    def render_bar(title: str, counts: Dict[str, int]) -> str:
        total = sum(counts.values()) or 1
        pc = {k: (counts[k] / total * 100.0) for k in counts}
        return (
            f"<div><b>{title}</b> — total {total}</div>"
            f"<div style='display:flex;height:16px;border:1px solid #ddd;width:520px'>"
            f"<div title='correct {counts['correct']}' style='width:{pc['correct']}%;background:#2ecc71'></div>"
            f"<div title='partial {counts['partially_correct']}' style='width:{pc['partially_correct']}%;background:#f1c40f'></div>"
            f"<div title='incorrect {counts['incorrect']}' style='width:{pc['incorrect']}%;background:#e74c3c'></div>"
            f"</div>"
        )

    html.append('<h3>LLM-as-judge (SEBA) aggregate verdicts</h3>')
    html.append(render_bar(baseline_run, base_counts))
    html.append(render_bar(iter_run, iter_counts))
    if meetings_run:
        html.append(render_bar(meetings_run, meet_counts))
    if extract_run:
        html.append(render_bar(extract_run, extr_counts))

    # SEBA per-criterion aggregates per run
    def criteria_aggregate(run_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for crit in CRITERIA:
            t = f = 0
            for lst in run_results.values():
                for r in lst:
                    seba = (r.get('evaluation') or {}).get('seba')
                    if not isinstance(seba, dict):
                        continue
                    # Find criterion key by prefix match to be robust to variations
                    val = None
                    for k, v in seba.items():
                        if isinstance(v, bool) and k.lower().startswith(crit.lower()):
                            val = v
                            break
                    if val is True:
                        t += 1
                    elif val is False:
                        f += 1
            total = t + f
            rate = (t / total) if total else 0.0
            out[crit] = {"true": t, "false": f, "total": total, "true_rate": rate}
        return out

    base_crit = criteria_aggregate(base_results)
    iter_crit = criteria_aggregate(iter_results)
    meet_crit = criteria_aggregate(meet_results) if meet_results else {crit: {"true_rate": 0.0} for crit in CRITERIA}
    extr_crit = criteria_aggregate(extr_results) if extr_results else {crit: {"true_rate": 0.0} for crit in CRITERIA}

    # Write CSV of per-criterion aggregates
    agg_csv = os.path.join(out_dir, 'seba_criteria_aggregate.csv')
    with open(agg_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['run', 'criterion', 'true', 'false', 'total', 'true_rate'])
        for crit in CRITERIA:
            bc = base_crit.get(crit, {"true":0,"false":0,"total":0,"true_rate":0.0})
            ic = iter_crit.get(crit, {"true":0,"false":0,"total":0,"true_rate":0.0})
            mc = meet_crit.get(crit, {"true":0,"false":0,"total":0,"true_rate":0.0})
            ec = extr_crit.get(crit, {"true":0,"false":0,"total":0,"true_rate":0.0})
            w.writerow([baseline_run, crit, bc['true'], bc['false'], bc['total'], f"{bc['true_rate']:.3f}"])
            w.writerow([iter_run, crit, ic['true'], ic['false'], ic['total'], f"{ic['true_rate']:.3f}"])
            if meetings_run:
                w.writerow([meetings_run, crit, mc['true'], mc['false'], mc['total'], f"{mc['true_rate']:.3f}"])
            if extract_run:
                w.writerow([extract_run, crit, ec['true'], ec['false'], ec['total'], f"{ec['true_rate']:.3f}"])

    # Render simple bars for per-criterion true_rate per run
    html.append('<h3>SEBA per-criterion true rate</h3>')
    html.append('<table>')
    hdr = '<tr><th>Criterion</th><th>' + baseline_run + '</th><th>' + iter_run + '</th>'
    if meetings_run:
        hdr += '<th>' + meetings_run + '</th>'
    if extract_run:
        hdr += '<th>' + extract_run + '</th>'
    hdr += '</tr>'
    html.append(hdr)
    def rate_bar(rate: float, color: str) -> str:
        pct = max(0.0, min(1.0, float(rate))) * 100.0
        return (f"<div style='width:260px;border:1px solid #ddd;height:14px'>"
                f"<div style='width:{pct:.1f}%;height:14px;background:{color}'></div>"
                f"</div> <span style='font-family:ui-monospace'>{pct:.1f}%</span>")
    for crit in CRITERIA:
        br = base_crit.get(crit, {"true_rate": 0.0})['true_rate']
        ir = iter_crit.get(crit, {"true_rate": 0.0})['true_rate']
        mr = meet_crit.get(crit, {"true_rate": 0.0})['true_rate'] if meetings_run else 0.0
        er = extr_crit.get(crit, {"true_rate": 0.0})['true_rate'] if extract_run else 0.0
        row_cells = [
            f'<td>{crit}</td>',
            f'<td>{rate_bar(br, "#2ecc71")}</td>',
            f'<td>{rate_bar(ir, "#3498db")}</td>',
        ]
        if meetings_run:
            row_cells.append(f'<td>{rate_bar(mr, "#9b59b6")}</td>')
        if extract_run:
            row_cells.append(f'<td>{rate_bar(er, "#e67e22")}</td>')
        html.append('<tr>' + ''.join(row_cells) + '</tr>')
    html.append('</table>')

    # Per-run aggregate RAGAS plots
    if go is not None:
        def collect_metrics(run_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, float]:
            sums: Dict[str, float] = {k: 0.0 for k in ('faithfulness','semantic_similarity','answer_correctness','llm_context_precision_without_reference')}
            counts: Dict[str, int] = {k: 0 for k in sums.keys()}
            for lst in run_results.values():
                for r in lst:
                    ragas = (r.get('evaluation') or {}).get('ragas')
                    if isinstance(ragas, dict):
                        for k in sums.keys():
                            v = ragas.get(k)
                            try:
                                if v is not None:
                                    fv = float(v)
                                    if not (fv != fv or fv == float('inf') or fv == float('-inf')):
                                        sums[k] += fv
                                        counts[k] += 1
                            except Exception:
                                pass
            return {k: (sums[k]/counts[k] if counts[k] else 0.0) for k in sums.keys()}

        base_avg = collect_metrics(base_results)
        iter_avg = collect_metrics(iter_results)
        metrics = list(base_avg.keys())
        html.append('<h3>RAGAS aggregate (averages)</h3>')
        fig = go.Figure()
        metrics_closed = metrics + metrics[:1]
        base_vals = [base_avg[m] for m in metrics] + [base_avg[metrics[0]]]
        iter_vals = [iter_avg[m] for m in metrics] + [iter_avg[metrics[0]]]
        fig.add_trace(go.Scatterpolar(r=base_vals, theta=metrics_closed, fill='toself', name=baseline_run))
        fig.add_trace(go.Scatterpolar(r=iter_vals, theta=metrics_closed, fill='toself', name=iter_run))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), showlegend=True)
        try:
            agg_png = os.path.join(out_dir, 'ragas_aggregate_compare.png')
            fig.write_image(agg_png, scale=2)
        except Exception:
            pass
        out_html = os.path.join(out_dir, 'ragas_aggregate_compare.html')
        fig.write_html(out_html, include_plotlyjs='cdn')
        html.append(f'<p>Saved RAGAS aggregate comparison: {out_html}</p>')

    html.append('</body></html>')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html))

    return out_dir


if __name__ == '__main__':
    out = build_comparison()
    print(f'Wrote comparison outputs to: {out}')


