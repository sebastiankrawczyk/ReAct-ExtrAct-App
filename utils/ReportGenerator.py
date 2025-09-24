import os
import re
import csv
import json
import math
from typing import List
from datetime import datetime
try:
    import plotly.graph_objects as go
except Exception:
    go = None

from config.config import (
    INPUT_PATH, 
    OUTPUT_PATH, 
    STORAGE_PATH, 
    API,
    EXECUTION_MODEL,
    EVAL_MODEL,
    EMBEDDING_MODEL,
    MAX_STEPS, 
    DESABLE_SECOND_LOOP,
    EVALUATION, 
    RAGAS,
    G_EVAL,
    GROUND_TRUTH,
    CLEAR_STORAGE, 
    COHERE_RERANK,
    CSV_ENCODING,
    CSV_DELIMITER,
)

class ReportGenerator:
    def __init__(self, queries, output_dir):
        self.queries = queries
        self.output_dir = output_dir
        self.files_results = []
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"[save] output_dir ready → {self.output_dir}")
        except Exception:
            pass

    def generate_partial_report(self, file, info, results):
        self.files_results.append({
            "file_name": file,
            "info": info,
            "result": results
        })
        os.makedirs(f"{self.output_dir}/{file}", exist_ok=True)
        file_path = f"{self.output_dir}/{file}/{file}_result"
        content = ""
        
        # markdown
        for i, result in enumerate(results):
            content += f"## Question {i+1}\n"
            content += f"```json\n{json.dumps(result, indent=4)}\n```\n"
            content += "___\n"
        with open(f"{file_path}.md", "w", encoding="utf-8") as f:
            f.write(content)
        try:
            print(f"[save] wrote → {file_path}.md")
        except Exception:
            pass

        # json: maintain baseline format; also allow react_phd normalization upstream
        data = {'results': results}
        with open(f"{file_path}.json", 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
        try:
            print(f"[save] wrote → {file_path}.json")
        except Exception:
            pass
        return 

    def generate_main_report(self):
        # todo add info (author, title, etc.)
        def _one_line(text: str) -> str:
            try:
                s = str(text)
            except Exception:
                s = ""
            s = s.replace("\r", " ").replace("\n", " ")
            s = re.sub(r"\s+", " ", s).strip()
            return s

        def _short(text: str, max_len: int = 140) -> str:
            s = _one_line(text)
            return s if len(s) <= max_len else (s[: max_len - 1] + "…")

        header_line_1 = ["file"]+[item for q in self.queries for item in (q["topic"], "", "", "")]
        header_line_2 = [""]+[item for q in self.queries for item in ("[context = original text]", "[LLM answer]", "[LLM concise]", f"[kod] opts: {_short(q.get('possible_options', 'none'))}")]

        data_rows = []
        for entry in self.files_results:
            data_row = [entry["file_name"]]
            for r in entry["result"]:
                contexts = []
                try:
                    bc_list = r.get("best_context") or []
                except Exception:
                    bc_list = []
                for i, c in enumerate(bc_list):
                    ctx = _one_line((c or {}).get("context", ""))
                    if ctx:
                        contexts.append(f"{i+1}. {ctx}")
                context = " | ".join(contexts)
                answer = _one_line(r.get("answer", ""))
                concise = _one_line(r.get("answer_concise", ""))
                code = _one_line(r.get("code", ""))
                data_row.append(context)
                data_row.append(answer)
                data_row.append(concise)
                data_row.append(code)
            data_rows.append(data_row)

        with open(f"{self.output_dir}/raport.csv", mode='w', newline='', encoding=CSV_ENCODING) as file:
            writer = csv.writer(file, delimiter=CSV_DELIMITER, quoting=csv.QUOTE_MINIMAL)

            writer.writerow(header_line_1)
            writer.writerow(header_line_2)

            writer.writerows(data_rows)

        try:
            print(f"[save] wrote → {self.output_dir}/raport.csv")
        except Exception:
            pass

        return
    
    def generate_config_report(self, execution_time):

        obj = {
            "config": {
                "api": API,
                "execution_model": EXECUTION_MODEL,
                "eval_model": EVAL_MODEL,
                "embedding_model": EMBEDDING_MODEL,
                "input_path": INPUT_PATH,
                "output_path": OUTPUT_PATH,
                "storage_path": STORAGE_PATH,
                "max_steps": MAX_STEPS,
                "disable_second_loop": DESABLE_SECOND_LOOP,
                "evaluation": EVALUATION,
                "ragas": RAGAS,
                "g_eval": G_EVAL,
                "ground_truth": GROUND_TRUTH,
                "clear_storage": CLEAR_STORAGE,
                "cohere_rerank": COHERE_RERANK,
            },
            "queries": self.queries,
            "files": [r["file_name"] for r in self.files_results],
            "execution_time": execution_time
        }

        with open(f"{self.output_dir}/start_up_detail.json", 'w', encoding='utf-8') as file:
            json.dump(obj, file, ensure_ascii=False, indent=4)
        try:
            print(f"[save] wrote → {self.output_dir}/start_up_detail.json")
        except Exception:
            pass
        
        # If evaluations exist in files_results, create a radar chart dataset for plotting
        radar_data = []
        allowed_keys = {
            "faithfulness",
            "semantic_similarity",
            "answer_correctness",
            "llm_context_precision_without_reference",
        }
        for entry in self.files_results:
            for r in entry["result"]:
                if isinstance(r.get("evaluation"), dict) and r["evaluation"].get("ragas"):
                    sample = r["evaluation"]["ragas"]
                    if isinstance(sample, dict):
                        obj = {
                            "file": entry["file_name"],
                            "topic": r["query"]["topic"],
                        }
                        for k in allowed_keys:
                            v = sample.get(k)
                            if isinstance(v, (int, float)):
                                obj[k] = float(v)
                        radar_data.append(obj)
        if radar_data:
            with open(f"{self.output_dir}/radar_dataset.json", 'w', encoding='utf-8') as f:
                json.dump({"samples": radar_data}, f, ensure_ascii=False, indent=4)
            try:
                print(f"[save] wrote → {self.output_dir}/radar_dataset.json")
            except Exception:
                pass
        
        # Additionally, render radar (spider) charts if Plotly is available
        if radar_data and go is not None:
            os.makedirs(f"{self.output_dir}/radar_charts", exist_ok=True)
            # Determine metric axes by collecting numeric keys across samples
            numeric_keys = []
            for key in radar_data[0].keys():
                if key not in ("file", "topic") and isinstance(radar_data[0][key], (int, float)):
                    numeric_keys.append(key)
            # Fallback: infer numeric keys by checking all keys across samples
            if not numeric_keys:
                all_keys = set()
                for s in radar_data:
                    all_keys.update(k for k, v in s.items() if k not in ("file", "topic") and isinstance(v, (int, float)))
                numeric_keys = sorted(all_keys)
            # Plot per file a multi-trace radar over topics
            file_to_samples = {}
            for s in radar_data:
                file_to_samples.setdefault(s["file"], []).append(s)
            for file_name, samples in file_to_samples.items():
                fig = go.Figure()
                theta = numeric_keys
                for s in samples:
                    r_vals = [float(s.get(k, 0) or 0) for k in numeric_keys]
                    fig.add_trace(go.Scatterpolar(r=r_vals, theta=theta, fill='toself', name=s["topic"]))
                fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True, title=f"RAGAS Metrics: {file_name}")
                out_path = f"{self.output_dir}/radar_charts/{file_name}.html"
                fig.write_html(out_path, include_plotlyjs='cdn')
                try:
                    print(f"[save] wrote → {out_path}")
                except Exception:
                    pass

            # Per-run aggregate PNG (average over all samples in this run)
            try:
                agg_dir = f"{self.output_dir}/aggregated"
                os.makedirs(agg_dir, exist_ok=True)
                # compute averages
                sums = {k: 0.0 for k in numeric_keys}
                counts = {k: 0 for k in numeric_keys}
                for s in radar_data:
                    for k in numeric_keys:
                        v = s.get(k)
                        if isinstance(v, (int, float)) and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                            sums[k] += float(v)
                            counts[k] += 1
                averages = {k: (sums[k]/counts[k] if counts[k] else 0.0) for k in numeric_keys}
                # write JSON
                with open(f"{agg_dir}/run_ragas_average.json", 'w', encoding='utf-8') as f:
                    json.dump({'averages': averages}, f, ensure_ascii=False, indent=4)
                try:
                    print(f"[save] wrote → {agg_dir}/run_ragas_average.json")
                except Exception:
                    pass
                # render PNG
                metrics_order = list(averages.keys())
                values = [averages[m] for m in metrics_order]
                metrics_closed = metrics_order + metrics_order[:1]
                values_closed = values + values[:1]
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(r=values_closed, theta=metrics_closed, fill='toself', name='Average'))
                fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=False, title='RAGAS Average Metrics (This Run)')
                fig.write_image(f"{agg_dir}/run_ragas_average.png", scale=2)
                try:
                    print(f"[save] wrote → {agg_dir}/run_ragas_average.png")
                except Exception:
                    pass
            except Exception:
                pass
        
        return
