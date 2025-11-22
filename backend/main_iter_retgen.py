import os
import time
import json
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from llama_index.core import Settings
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

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
    MAX_STEPS,
    CONCURRENCY,
    GROUND_TRUTH,
    CLEAR_STORAGE,
    COHERE_RERANK,
)
from config.config_keys import (
    OPENAI_API_KEY,
    LLAMA_CLOUD_API_KEY,
    OPENROUTER_API_KEY,
)

from config.queries import QUERIES
try:
    from config.ground_truth import GROUND_TRUTH_LIST
except Exception:
    GROUND_TRUTH_LIST = {}

from utils.ReportGenerator import ReportGenerator
from utils.VectorQueryEngineCreator import VectorQueryEngineCreator
from utils.TokenTracker import TokenTracker


RUN_TAG = "iter_retgen"


def _expand_query(llm, topic: str, prev_answer: str, iteration: int) -> str:
    prompt = f"""
        You are iteratively expanding the retrieval query for the topic.
        Return a SHORT clause (<= 12 words) that should be APPENDED to the current query to better target missing details.
        Output only the clause. If nothing useful, reply exactly: STOP.

        Topic: {topic}
        Iteration: {iteration}
        Current answer: {prev_answer}
    """
    raw = f"{llm.complete(prompt)!s}".strip()
    if not raw or raw.lower().startswith("stop"):
        return ""
    return raw.splitlines()[0].strip()


def _synthesize(llm, topic: str, snippets: List[str]) -> str:
    ctx_text = "\n\n---\n\n".join(snippets[:16])
    prompt = f"""
        Using only the provided contexts, answer the topic concisely and directly.
        If the answer is not fully specified in the contexts, provide your best supported answer and briefly note uncertainty.
        Do not fabricate details that are not grounded in the contexts.

        Topic: {topic}
        Contexts:
        {ctx_text}
        Return just the answer text.
    """
    raw = f"{llm.complete(prompt)!s}".strip()
    return raw


def _node_id_or_hash(n) -> Tuple[str, int]:
    try:
        nid = getattr(n.node, 'node_id', None) or getattr(n.node, 'id_', None)
    except Exception:
        nid = None
    if nid:
        return ("id", hash(str(nid)))
    try:
        txt = (n.node.get_content() or '').strip()
    except Exception:
        txt = str(getattr(n, 'get_content', lambda: '')()).strip()
    return ("txt", hash(txt))


def main():
    start = time.time()

    if API == "openrouter":
        print(f"Using OpenRouter execution model: {EXECUTION_MODEL} [ITER-RETGEN]")
        Settings.llm = OpenAILike(
            model=EXECUTION_MODEL,
            api_base="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
            is_chat_model=True,
        )
    elif API == "ollama":
        exec_model = OLLAMA_EXECUTION_MODEL or EXECUTION_MODEL
        print(f"Using Ollama execution model: {exec_model} @ {OLLAMA_BASE_URL} [ITER-RETGEN]")
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

    # Install token tracker callbacks
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
                import shutil
                shutil.rmtree(item_path)

    output_path = os.path.join(OUTPUT_PATH, f"{time.strftime('%Y.%m.%d_%H.%M.%S')}_{RUN_TAG}")
    os.makedirs(output_path, exist_ok=True)
    raportGenerator = ReportGenerator(QUERIES, output_path)

    files = [os.path.splitext(f)[0] for f in os.listdir(INPUT_PATH) if f.lower().endswith('.pdf')]
    plain_questions = [q for q in QUERIES]

    def process_file(file: str):
        print(f"\nProcessing file: {file}")
        query_engine = VectorQueryEngineCreator(
            llama_parse_api_key=LLAMA_CLOUD_API_KEY,
            cohere_api_key=os.getenv('COHERE_API_KEY',''),
            input_path=INPUT_PATH,
            storage_path=STORAGE_PATH,
            cohere_rerank=COHERE_RERANK,
            embedding_model_name=EMBEDDING_MODEL,
            enable_section_reasoner=False,
            response_mode='compact',
        ).get_query_engine(file)

        def run_one(qi: int):
            q = plain_questions[qi]
            topic = q["topic"]
            possible_options = q.get("possible_options", "None")

            print(f"  - [{file}] Query [{qi+1}/{len(plain_questions)}]: {topic}")
            # Iterative retrieve-generate (classic expansion): query += clause
            collected_nodes = []
            seen_keys = set()
            answer = ""
            query = topic
            for it in range(max(1, int(MAX_STEPS))):
                resp = query_engine.query(query)
                answer = f"{resp!s}"
                nodes = getattr(resp, 'source_nodes', []) or []
                new_nodes_added = 0
                for n in nodes:
                    key = _node_id_or_hash(n)
                    if key not in seen_keys:
                        seen_keys.add(key)
                        collected_nodes.append(n)
                        new_nodes_added += 1
                # Expand query for next loop
                if it < int(MAX_STEPS) - 1:
                    if new_nodes_added == 0:
                        break
                    clause = _expand_query(Settings.llm, topic, answer, it + 1)
                    if not clause:
                        break
                    # Append clause to current query
                    query = (query + " " + clause).strip()

            # Final synthesis over collected contexts
            snippets = []
            dedup: Dict[Tuple[str, int], object] = {}
            for n in collected_nodes:
                key = _node_id_or_hash(n)
                prev = dedup.get(key)
                if prev is None or (float(getattr(n, 'score', 0.0) or 0.0) > float(getattr(prev, 'score', 0.0) or 0.0)):
                    dedup[key] = n
            uniq_nodes = list(dedup.values())
            try:
                uniq_nodes.sort(key=lambda c: c.score, reverse=True)
            except Exception:
                pass
            for n in uniq_nodes[:16]:
                try:
                    snippets.append((n.node.get_content() or '').strip())
                except Exception:
                    snippets.append(str(getattr(n, 'get_content', lambda: '')()).strip())
            final_answer = _synthesize(Settings.llm, topic, snippets) or answer

            # Produce a concise version: list if enumeration implied, else one short sentence/phrase
            try:
                summary_prompt = f"""
                Provide a concise answer to the topic using the provided text.

                Topic: {topic}
                Text: {final_answer}

                Requirements:
                - If the topic implies an enumeration (e.g., "list/which/name/keywords/methods/algorithms/etc."), output ONLY a comma-separated list of the items (short words or phrases), no extra words.
                - Otherwise, output ONLY a single short sentence or phrase directly answering the topic.
                - Do not add any prefixes, labels, or explanations.
                - Return just the answer as plain text.
                """.strip()
                concise_text = f"{Settings.llm.complete(summary_prompt)!s}".strip()
            except Exception:
                concise_text = ""

            # Best context (include page/section)
            best_context = []
            for n in uniq_nodes[:5]:
                try:
                    txt = (n.node.get_content() or '').strip()
                except Exception:
                    txt = str(getattr(n, 'get_content', lambda: '')()).strip()
                md = {}
                try:
                    md = getattr(n.node, 'metadata', {}) or {}
                except Exception:
                    md = {}
                page = md.get('page_label') if isinstance(md, dict) else None
                if page is None and isinstance(md, dict):
                    page = md.get('page')
                section = md.get('section') if isinstance(md, dict) else None
                best_context.append({"context": txt, "score": getattr(n, 'score', None), "page": page, "section": section})

            if str(possible_options).lower() != "none":
                code = _extract_code_with_options(Settings.llm, final_answer, possible_options)
            else:
                code = ""

            result = {
                "query": {"topic": topic, "possible_options": possible_options},
                "question": topic,
                "answer": final_answer,
                "answer_concise": concise_text,
                "code": code,
                "best_context": best_context,
            }

            # evaluation disabled/removed
            result["evaluation"] = None

            return result

        # Per-question concurrency
        results = []
        with ThreadPoolExecutor(max_workers=min(8, max(1, os.cpu_count() or 4))) as pool:
            futures = {pool.submit(run_one, qi): qi for qi in range(len(plain_questions))}
            for f in as_completed(futures):
                try:
                    results.append(f.result())
                except Exception:
                    continue
        order = {plain_questions[i]["topic"]: i for i in range(len(plain_questions))}
        results.sort(key=lambda r: order.get(r["question"], 0))

        raportGenerator.generate_partial_report(file, {}, results)
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

    max_workers_files = max(1, int(CONCURRENCY) if str(CONCURRENCY).isdigit() else 3)
    with ThreadPoolExecutor(max_workers=max_workers_files) as pool:
        # Start counting from the beginning of question answering
        tracker.start()
        futs = {pool.submit(process_file, f): f for f in files}
        for _ in as_completed(futs):
            pass

    raportGenerator.generate_main_report()
    end = time.time()
    raportGenerator.generate_config_report(end - start)

    print("END [ITER-RETGEN]")
    print(f"Execution time: {end - start} seconds")
    tracker.write_report(output_path)
    rep = tracker.report()
    print(f"Token usage → LLM: {rep.get('total_llm_token_count')} | Embed: {rep.get('total_embedding_token_count')} | Total: {rep.get('total_token_count')}")
    print(f"Usage report: {os.path.join(output_path, 'usage.json')}")
    return


def _extract_code_with_options(llm, answer_text: str, possible_options: str) -> str:
    prompt = f"""
                Your task is to extract and return only the items from the provided options that appear in the given answer.

                Answer:
                {answer_text}

                Options:
                {possible_options}

                Requirements:
                Return a comma-separated string of matching items (e.g., "Option1, Option2, ...").
                The output must include only the matching items without any additional text, words, or formatting.
            """
    raw = f"{llm.complete(prompt)!s}"
    allowed = [opt.strip() for opt in str(possible_options).split(',')]
    lower_to_canonical = {opt.lower(): opt for opt in allowed}
    chosen: List[str] = []
    for token in raw.split(','):
        key = token.strip().lower()
        if key in lower_to_canonical:
            canonical = lower_to_canonical[key]
            if canonical not in chosen:
                chosen.append(canonical)
    return ", ".join(chosen)


if __name__ == "__main__":
    main()


