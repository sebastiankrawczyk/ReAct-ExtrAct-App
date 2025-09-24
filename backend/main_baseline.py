import os
import shutil
import time
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

from llama_index.core import Settings
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

import config.config as conf

from config.config import (
    API,
    INPUT_PATH,
    OUTPUT_PATH,
    STORAGE_PATH,
    EXECUTION_MODEL,
    EVAL_MODEL,
    EMBEDDING_MODEL,
    EMBEDDING_API,
    OLLAMA_BASE_URL,
    OLLAMA_EXECUTION_MODEL,
    OLLAMA_EMBEDDING_MODEL,
    CONCURRENCY,
    EVALUATION,
    RAGAS,
    G_EVAL,
    GROUND_TRUTH,
    CLEAR_STORAGE,
    COHERE_RERANK,
)
from config.config_keys import (
    OPENAI_API_KEY,
    LLAMA_CLOUD_API_KEY,
    COHERE_API_KEY,
    OPENROUTER_API_KEY,
)

from config.queries import QUERIES
try:
    from config.ground_truth import GROUND_TRUTH_LIST
except Exception:
    GROUND_TRUTH_LIST = {}

from utils.ReportGenerator import ReportGenerator
from utils.VectorQueryEngineCreator import VectorQueryEngineCreator
from utils.RAGEvaluator import RAGEvaluator
from utils.TokenTracker import TokenTracker


# Ensure ReAct features are disabled for baseline context
try:
    conf.REACT_SUBQUESTIONS = False
    conf.REACT_REFINEMENT = False
    conf.REACT_INCLUDE_OBSERVATIONS = False
except Exception:
    pass

RUN_TAG = "baseline"


def _extract_code_with_options(llm, answer_text: str, possible_options: str, concise_text: str = "") -> str:
    prompt = f"""
                Your task is to extract and return only the CODES from the provided options that appear in the given answer.

                Answer (full):
                {answer_text}

                Answer (concise):
                {concise_text}

                Options (each may be CODE or "CODE | DEFINITION"):
                {possible_options}

                Requirements:
                - Match by either code or definition, but output ONLY the codes.
                - Return a comma-separated string of matching codes (e.g., "RF, LR").
                - Do not add any extra text.
            """
    raw = f"{llm.complete(prompt)!s}"
    # Build alias→code map supporting "CODE | DEFINITION" pairs
    alias_to_code: dict[str, str] = {}
    for seg in str(possible_options).split(','):
        part = seg.strip()
        if not part:
            continue
        if '|' in part:
            left, right = part.split('|', 1)
            code = left.strip()
            definition = right.strip()
        else:
            code = part.strip()
            definition = ""
        if code:
            alias_to_code[code.lower()] = code
        if definition:
            alias_to_code[definition.lower()] = code
    chosen: List[str] = []
    for token in raw.split(','):
        key = token.strip().lower()
        code = alias_to_code.get(key)
        if code and code not in chosen:
            chosen.append(code)
    return ", ".join(chosen)


def main():
    start = time.time()

    if API == "openrouter":
        print(f"Using OpenRouter execution model: {EXECUTION_MODEL} [BASELINE]")
        Settings.llm = OpenAILike(
            model=EXECUTION_MODEL,
            api_base="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
            is_chat_model=True,
        )
    elif API == "ollama":
        exec_model = OLLAMA_EXECUTION_MODEL or EXECUTION_MODEL
        print(f"Using Ollama execution model: {exec_model} @ {OLLAMA_BASE_URL} [BASELINE]")
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

    # Install token tracker callbacks (counts LLM + embedding tokens)
    tracker = TokenTracker()
    tracker.install()

    if CLEAR_STORAGE:
        for item in os.listdir(STORAGE_PATH):
            item_path = os.path.join(STORAGE_PATH, item)
            if os.path.basename(item_path) == ".gitkeep":
                continue
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)

    output_path = os.path.join(OUTPUT_PATH, f"{time.strftime('%Y.%m.%d_%H.%M.%S')}_{RUN_TAG}")
    os.makedirs(output_path, exist_ok=True)

    # Plain questions: use topics directly; no generation/reformulation
    plain_questions = [q for q in QUERIES]
    raportGenerator = ReportGenerator(QUERIES, output_path)

    files = os.listdir(INPUT_PATH)
    pdf_files = []
    for file in files:
        if file.lower().endswith('.pdf'):
            pdf_files.append(os.path.splitext(file)[0])

    def process_file(file: str):
        print(f"\nProcessing file: {file}")
        # Choose engine based on USE_GROBID
        use_grobid = str(os.getenv('USE_GROBID') or '0').strip().lower() in ('1','true','yes','y','on')
        EngineCls = VectorQueryEngineCreator
        if use_grobid:
            try:
                from utils.VectorQueryEngineCreatorGrobid import VectorQueryEngineCreatorGrobid as _G  # type: ignore
                EngineCls = _G
            except Exception:
                EngineCls = VectorQueryEngineCreator
        query_engine = EngineCls(
            llama_parse_api_key=LLAMA_CLOUD_API_KEY,
            cohere_api_key=COHERE_API_KEY,
            input_path=INPUT_PATH,
            storage_path=STORAGE_PATH,
            cohere_rerank=COHERE_RERANK,
            embedding_model_name=EMBEDDING_MODEL,
            response_mode='compact',
        ).get_query_engine(file)

        def run_one(qi: int):
            q = plain_questions[qi]
            topic = q["topic"]
            possible_options = q.get("possible_options", "None")
            question_text = topic

            print(f"  - [{file}] Query [{qi+1}/{len(plain_questions)}]: {topic}")
            response = query_engine.query(question_text)
            answer_text = f"{response!s}"

            # Produce a concise one-liner summary of the answer (kept alongside the full answer)
            try:
                summary_prompt = f"""
                Provide a concise answer to the topic using the provided text.

                Topic: {topic}
                Text: {answer_text}

                Requirements:
                - If the topic implies an enumeration (e.g., "list/which/name/keywords/methods/algorithms/etc."), output ONLY a comma-separated list of the items (short words or phrases), no extra words.
                - Otherwise, output ONLY a single short sentence or phrase directly answering the topic.
                - Do not add any prefixes, labels, or explanations.
                - Return just the answer as plain text.
                """.strip()
                concise_text = f"{Settings.llm.complete(summary_prompt)!s}".strip()
            except Exception:
                concise_text = ""

            source_nodes = getattr(response, 'source_nodes', []) or []
            contexts = []
            try:
                contexts = [c.node.get_content() for c in source_nodes]
            except Exception:
                try:
                    contexts = [str(getattr(c, 'get_content', lambda: '')()) for c in source_nodes]
                except Exception:
                    contexts = []

            try:
                best = sorted(source_nodes, key=lambda c: c.score, reverse=True)[:5]
            except Exception:
                best = source_nodes[:5]
            best_context = []
            for c in best:
                try:
                    txt = c.node.get_content().strip()
                except Exception:
                    try:
                        txt = str(getattr(c, 'get_content', lambda: '')()).strip()
                    except Exception:
                        txt = ''
                if not txt:
                    continue
                try:
                    md = getattr(c.node, 'metadata', {}) or {}
                except Exception:
                    md = {}
                page = md.get('page_label') if isinstance(md, dict) else None
                if page is None and isinstance(md, dict):
                    page = md.get('page')
                section = md.get('section') if isinstance(md, dict) else None
                best_context.append({
                    "context": txt,
                    "score": getattr(c, 'score', None),
                    "page": page,
                    "section": section,
                })

            if str(possible_options).lower() != "none":
                code = _extract_code_with_options(Settings.llm, answer_text, possible_options, concise_text)
            else:
                code = ""

            result = {
                "query": {"topic": topic, "possible_options": possible_options},
                "question": question_text,
                "answer": answer_text,
                "answer_concise": concise_text,
                "code": code,
                "best_context": best_context,
                "contexts": contexts,
            }

            # immediate evaluation per-question
            if EVALUATION:
                try:
                    gt = [""]
                    if GROUND_TRUTH:
                        try:
                            full_gt = GROUND_TRUTH_LIST.get(file, [])
                            idx = {plain_questions[i]["topic"]: i for i in range(len(plain_questions))}.get(topic, None)
                            if idx is not None and idx < len(full_gt):
                                gt = [full_gt[idx]]
                        except Exception:
                            gt = [""]
                    eval_one = RAGEvaluator(
                        start_questions=[topic],
                        llm=Settings.llm,
                        results=[result],
                        eval_model=EVAL_MODEL,
                        evaluate_ragas=RAGAS,
                        evaluate_geval=G_EVAL,
                        ground_truth=gt,
                        filename=file,
                        embedding_model_name=EMBEDDING_MODEL,
                    ).evaluate()
                    result["evaluation"] = eval_one[0] if eval_one else {"ragas": None, "geval": None}
                except Exception:
                    result["evaluation"] = {"ragas": None, "geval": None}
            else:
                result["evaluation"] = None

            # drop contexts in final output
            if "contexts" in result:
                del result["contexts"]

            return result

        max_workers_q = min(8, max(1, os.cpu_count() or 4))
        results = []
        with ThreadPoolExecutor(max_workers=max_workers_q) as pool:
            futures = {pool.submit(run_one, qi): qi for qi in range(len(plain_questions))}
            for fut in as_completed(futures):
                try:
                    results.append(fut.result())
                except Exception:
                    # Skip failed question; continue with others
                    continue
        order = {plain_questions[i]["topic"]: i for i in range(len(plain_questions))}
        results.sort(key=lambda r: order.get(r["question"], 0))

        info = {}
        raportGenerator.generate_partial_report(file, info, results)
        # Additionally write a baseline-compatible flat list for UI consumption
        try:
            out_dir = os.path.join(output_path, file)
            os.makedirs(out_dir, exist_ok=True)
            baseline_like = []
            for r in results:
                baseline_like.append({
                    "query": r.get("query", {}),
                    "question": r.get("question", ""),
                    "answer": r.get("answer", ""),
                    "answer_concise": r.get("answer_concise", ""),
                    "code": r.get("code", ""),
                    "best_context": r.get("best_context", []),
                })
            with open(os.path.join(out_dir, f"{file}_baseline_like.json"), "w", encoding="utf-8") as f:
                import json as _json
                _json.dump(baseline_like, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # Run up to CONCURRENCY files in parallel
    # Start counting from the beginning of question answering
    tracker.start()
    with ThreadPoolExecutor(max_workers=max(1, int(CONCURRENCY) if str(CONCURRENCY).isdigit() else 3)) as pool:
        futures = {pool.submit(process_file, f): f for f in pdf_files}
        for _ in as_completed(futures):
            pass

    raportGenerator.generate_main_report()
    end = time.time()
    execution_time = end - start
    raportGenerator.generate_config_report(execution_time)

    print("END [BASELINE]")
    print(f"Execution time: {execution_time} seconds")
    # Persist token/time usage
    tracker.write_report(output_path)
    rep = tracker.report()
    print(f"Token usage → LLM: {rep.get('total_llm_token_count')} | Embed: {rep.get('total_embedding_token_count')} | Total: {rep.get('total_token_count')}")
    print(f"Usage report: {os.path.join(output_path, 'usage.json')}")
    return


if __name__ == "__main__":
    main()



