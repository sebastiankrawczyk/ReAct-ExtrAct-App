import os
import re
import csv
import json
import math
from typing import List
from datetime import datetime

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
        
        return
