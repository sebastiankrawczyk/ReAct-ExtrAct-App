import os
import time
import json
from typing import List, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import QueryBundle, NodeWithScore

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
)
from config.config_keys import (
    OPENAI_API_KEY,
    OPENROUTER_API_KEY,
)

from config.queries import QUERIES
from eval.ground_truth_dual import get_dual
try:
    from config.ground_truth import GROUND_TRUTH_LIST
except Exception:
    GROUND_TRUTH_LIST = {}

from utils.ReportGenerator import ReportGenerator
from utils.RAGEvaluator import RAGEvaluator


RUN_TAG = "react"


def _read_section_tree(storage_path: str, file: str) -> str:
    persist_dir = os.path.join(storage_path, f"{file}_vector_index")
    path = os.path.join(persist_dir, "section_tree.md")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def _read_raw_markdown(storage_path: str, file: str) -> str:
    persist_dir = os.path.join(storage_path, f"{file}_vector_index")
    path = os.path.join(persist_dir, "raw_markdown.md")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def _topic_relationship_map(llm, topics: List[str], paper_context: str) -> Dict:
    prompt = (
        "You are building a topic relationship map for extracting facts from a single research paper.\n"
        "Given the list of extraction items (topics), infer likely co-occurrence relationships in papers.\n"
        "Return ONLY JSON with fields: relationships (list of \"A->B\" implications),\n"
        "strong_pairs (pairs likely found together), and notes (short guidance).\n\n"
        f"topics: {json.dumps(topics, ensure_ascii=False)}\n"
        "paper_context (abstract and goals, may be truncated):\n"
        f"{paper_context[:2500]}\n"
        "Example output: {\n  \"relationships\": [\"Algorithm->Metrics\"],\n  \"strong_pairs\": [[\"Dataset Size\", \"Data Source\"]],\n  \"notes\": \"Metrics reported per algorithm; dataset details near data source.\"\n}"
    )
    raw = f"{llm.complete(prompt)!s}".strip()
    start = raw.find("{"); end = raw.rfind("}")
    json_str = raw[start:end+1] if start != -1 and end != -1 and end > start else raw
    try:
        obj = json.loads(json_str)
        if not isinstance(obj, dict):
            return {"relationships": [], "strong_pairs": [], "notes": ""}
        obj.setdefault("relationships", [])
        obj.setdefault("strong_pairs", [])
        obj.setdefault("notes", "")
        return obj
    except Exception:
        return {"relationships": [], "strong_pairs": [], "notes": ""}


def _section_guide(llm, section_tree_md: str) -> Dict[str, List[str]]:
    prompt = f"""
        Identify which sections of the paper should be prioritized vs avoided for fact extraction.\n
        Section Tree (subset):\n
        {section_tree_md[:4000]}\n
        Rules:\n
        - Prefer Methods/Approach, Experiments/Results/Evaluation, Dataset/Data, Conclusions\n
        - Avoid References, Acknowledgments, Appendix, Related Work (unless explicitly relevant)\n
        Output ONLY JSON: {{"prefer": [..], "avoid": [..]}}
    """
    raw = f"{llm.complete(prompt)!s}".strip()
    start = raw.find("{"); end = raw.rfind("}")
    json_str = raw[start:end+1] if start != -1 and end != -1 and end > start else raw
    try:
        obj = json.loads(json_str)
        prefer = [str(s).strip() for s in (obj.get("prefer") or []) if str(s).strip()]
        avoid = [str(s).strip() for s in (obj.get("avoid") or []) if str(s).strip()]
        return {"prefer": prefer, "avoid": avoid}
    except Exception:
        return {"prefer": [], "avoid": []}


def _attack_order(llm, topics: List[str], relationships: Dict, paper_context: str) -> List[str]:
    prompt = (
        "Create a smart attack order for extracting topics from a paper.\n"
        "Use the relationships and typical paper structure to decide the sequence.\n"
        "Return ONLY JSON with array 'order'.\n\n"
        f"topics: {json.dumps(topics, ensure_ascii=False)}\n"
        f"relationships_hint: {json.dumps(relationships, ensure_ascii=False)}\n"
        "paper_context (abstract and goals, may be truncated):\n"
        f"{paper_context[:2500]}\n"
    )
    raw = f"{llm.complete(prompt)!s}".strip()
    start = raw.find("{"); end = raw.rfind("}")
    json_str = raw[start:end+1] if start != -1 and end != -1 and end > start else raw
    try:
        obj = json.loads(json_str)
        arr = obj.get("order") or []
        seq: List[str] = []
        for t in arr:
            s = str(t).strip()
            if s and s in topics and s not in seq:
                seq.append(s)
        # fallback to original order if llm output is empty
        return seq or topics
    except Exception:
        return topics


def _get_content_safe(n: NodeWithScore) -> str:
    try:
        return (n.node.get_content() or "").strip()
    except Exception:
        pass
    try:
        getter = getattr(n, 'get_content', None)
        if callable(getter):
            return str(getter() or '').strip()
    except Exception:
        pass
    return ""


def _filter_by_sections(nodes: List[NodeWithScore], prefer: List[str], avoid: List[str], top_k: int) -> List[NodeWithScore]:
    avoid_lower = [a.lower() for a in avoid]
    prefer_lower = [p.lower() for p in prefer]
    kept: List[NodeWithScore] = []
    for n in nodes:
        try:
            sec = ((n.node.metadata or {}).get("section") or "").lower()
        except Exception:
            sec = ""
        if any(a in sec or sec in a for a in avoid_lower if a):
            continue
        kept.append(n)
    if not kept:
        kept = list(nodes)
    # gentle boost for preferred
    for n in kept:
        try:
            sec = ((n.node.metadata or {}).get("section") or "").lower()
            if any(p in sec or sec in p for p in prefer_lower if p):
                n.score = (n.score or 0.0) * 1.2
        except Exception:
            pass
    kept.sort(key=lambda x: x.score or 0.0, reverse=True)
    return kept[:top_k]


def _synthesize_grounded(llm, topic: str, nodes: List[NodeWithScore]) -> str:
    contexts: List[str] = []
    for n in nodes[:10]:
        txt = _get_content_safe(n)
        if txt:
            contexts.append(txt)
    ctx_text = "\n\n---\n\n".join(contexts)
    prompt = f"""
        Answer the topic using the provided contexts \n
        If insufficient, reply exactly: "insufficient evidence".\n
        Topic: {topic}\n
        Contexts:\n
        {ctx_text}\n
        Output only the final answer as a single plain string.
    """
    ans = f"{llm.complete(prompt)!s}".strip()
    return ans or "insufficient evidence"


def _verify_supported(llm, topic: str, answer_text: str, contexts: List[str]) -> bool:
    joined = "\n\n---\n\n".join([str(c)[:800] for c in contexts[:12]])
    prompt = (
        "Validate if the proposed answer is directly supported by the contexts.\n"
        f"Question: {topic}\n"
        f"Proposed answer: {answer_text}\n"
        f"Contexts: \n{joined}\n\n"
        "Respond strictly with 'yes' or 'no'."
    )
    raw = f"{llm.complete(prompt)!s}".strip().lower()
    return raw.startswith("y")


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


def _extract_abstract_and_goals(raw_md_text: str, retriever=None) -> str:
    if not raw_md_text:
        raw_md_text = ""
    lines = raw_md_text.splitlines()
    abstract_block: List[str] = []
    collecting = False
    for line in lines:
        s = line.strip()
        if s.lower().startswith("#") and "abstract" in s.lower():
            collecting = True
            continue
        if collecting and s.startswith("#"):
            break
        if collecting:
            abstract_block.append(line)
    abstract_text = "\n".join(abstract_block).strip()
    # Heuristic goal sentences (first 2000 chars to keep it light)
    goal_candidates: List[str] = []
    import re
    sample = raw_md_text[:20000]
    for sent in re.split(r"(?<=[.!?])\s+", sample):
        ls = sent.lower()
        if ("in this paper" in ls) or ("this paper" in ls) or ("we propose" in ls) or ("we present" in ls) or ("we introduce" in ls) or ("our goal" in ls) or ("our objective" in ls) or ("our aim" in ls) or ("purpose" in ls):
            goal_candidates.append(sent.strip())
        if len(goal_candidates) >= 6:
            break
    goal_text = " ".join(goal_candidates).strip()
    context_parts: List[str] = []
    if abstract_text:
        context_parts.append("ABSTRACT:\n" + abstract_text)
    if goal_text:
        context_parts.append("GOALS:\n" + goal_text)
    context = "\n\n".join(context_parts).strip()
    # Fallback: light retrieval for objectives if context too short
    if (not context or len(context) < 300) and retriever is not None:
        try:
            queries = [
                "paper abstract",
                "objective of this paper",
                "we propose",
                "we present",
            ]
            from llama_index.core.schema import QueryBundle
            snippets: List[str] = []
            for q in queries:
                try:
                    nodes = retriever.retrieve(QueryBundle(query_str=q))
                    for n in nodes[:3]:
                        txt = _get_content_safe(n)
                        if txt:
                            snippets.append(txt)
                except Exception:
                    continue
            if snippets:
                extra = ("\n\nRETRIEVED:\n" + "\n\n---\n\n".join(snippets[:3]))
                context = (context + "\n\n" + extra).strip()
        except Exception:
            pass
    return context[:4000]


def _plan(llm, file: str, topics: List[str], retriever=None) -> Dict:
    section_tree_md = _read_section_tree(STORAGE_PATH, file)
    raw_md = _read_raw_markdown(STORAGE_PATH, file)
    planning_context = _extract_abstract_and_goals(raw_md, retriever=retriever)
    relationships = _topic_relationship_map(llm, topics, planning_context)
    guide = _section_guide(llm, section_tree_md)
    order = _attack_order(llm, topics, relationships, planning_context)
    return {"relationships": relationships, "guide": guide, "order": order, "context": planning_context}


def main():
    start = time.time()

    if API == "openrouter":
        print(f"Using OpenRouter execution model: {EXECUTION_MODEL} [REACT]")
        Settings.llm = OpenAILike(
            model=EXECUTION_MODEL,
            api_base="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
            is_chat_model=True,
        )
    elif API == "ollama":
        exec_model = OLLAMA_EXECUTION_MODEL or EXECUTION_MODEL
        print(f"Using Ollama execution model: {exec_model} @ {OLLAMA_BASE_URL} [REACT]")
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

    output_path = os.path.join(OUTPUT_PATH, f"{time.strftime('%Y.%m.%d_%H.%M.%S')}_{RUN_TAG}")
    os.makedirs(output_path, exist_ok=True)
    raportGenerator = ReportGenerator(QUERIES, output_path)

    files = [os.path.splitext(f)[0] for f in os.listdir(INPUT_PATH) if f.lower().endswith('.pdf')]
    topics = [q["topic"] for q in QUERIES]

    def process_file(file: str):
        print(f"\nProcessing file: {file}")
        # Load or build vector index and retriever
        persist_dir = os.path.join(STORAGE_PATH, f"{file}_vector_index")
        try:
            if CLEAR_STORAGE and os.path.isdir(persist_dir):
                import shutil
                shutil.rmtree(persist_dir)
        except Exception:
            pass
        try:
            storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
            vector_index = load_index_from_storage(storage_context)
        except Exception:
            # Build from PDFs if not present
            from utils.VectorQueryEngineCreator import VectorQueryEngineCreator
            vqc = VectorQueryEngineCreator(
                llama_parse_api_key=os.getenv('LLAMA_CLOUD_API_KEY') or '',
                cohere_api_key=os.getenv('COHERE_API_KEY') or '',
                input_path=INPUT_PATH,
                storage_path=STORAGE_PATH,
                cohere_rerank=False,
                embedding_model_name=EMBEDDING_MODEL,
                enable_section_reasoner=False,
                response_mode='compact',
            )
            qe = vqc.get_query_engine(file)
            # Extract underlying index from query engine
            try:
                vector_index = getattr(qe, 'retriever', None).index  # type: ignore
            except Exception:
                # As a fallback, try loading again after builder persisted
                storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
                vector_index = load_index_from_storage(storage_context)
        retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=20)

        # Step 1: Game Plan (now includes abstract/goals context)
        plan = _plan(Settings.llm, file, topics, retriever=retriever)
        try:
            print(f"[plan] {file}: plan ready; topics={len(topics)}")
        except Exception:
            pass
        prefer = plan["guide"].get("prefer", [])
        avoid = plan["guide"].get("avoid", [])
        attack = plan.get("order") or topics

        # Shared cache for cross-question hints
        results_by_topic: Dict[str, Dict] = {}

        # Helper for topic index within attack order
        def _idx_of(t: str) -> int:
            try:
                return attack.index(t)
            except Exception:
                return 0

        def run_one(topic: str, possible_options: str):
            # Progress marker for per-topic querying
            try:
                i = _idx_of(topic) + 1
                N = len(attack)
                print(f"  - [[{file}]] Query [{i}/{N}]: {topic}")
            except Exception:
                pass
            # Step 2: hunt for clues (initial retrieval in preferred sections)
            qb = QueryBundle(query_str=topic)
            raw_nodes = retriever.retrieve(qb)
            nodes = _filter_by_sections(list(raw_nodes), prefer, avoid, top_k=5)

            # Intelligent follow-up: if answer insufficient, use relationships to target co-mentions
            answer_text = _synthesize_grounded(Settings.llm, topic, nodes)
            if not answer_text or answer_text.lower().strip() == "insufficient evidence":
                # Use strong pairs and relationships to craft alternate queries
                other_topics = [t for t in attack if t != topic]
                alt_queries: List[str] = []
                try:
                    strong_pairs = plan["relationships"].get("strong_pairs") or []
                    for pair in strong_pairs:
                        if isinstance(pair, list) and topic in pair:
                            alt_queries.extend([t for t in pair if t != topic])
                except Exception:
                    pass
                # Also include previously answered topics as anchors
                alt_queries.extend([t for t, r in results_by_topic.items() if r.get("answer")])
                # Broaden retrieval with these alternates and deduplicate
                try:
                    broader = VectorIndexRetriever(index=vector_index, similarity_top_k=40)
                except Exception:
                    broader = retriever
                aggregated: List[NodeWithScore] = list(raw_nodes)
                for aq in alt_queries[:3]:
                    try:
                        aggregated.extend(list(broader.retrieve(QueryBundle(query_str=aq))))
                    except Exception:
                        continue
                # Dedup and filter again
                dedup = {}
                for n in aggregated:
                    try:
                        nid = getattr(n.node, 'node_id', None) or getattr(n.node, 'id_', None) or id(n.node)
                    except Exception:
                        nid = id(n)
                    if nid not in dedup or ((n.score or 0.0) > (dedup[nid].score or 0.0)):
                        dedup[nid] = n
                nodes = _filter_by_sections(list(dedup.values()), prefer, avoid, top_k=12)
                answer_text = _synthesize_grounded(Settings.llm, topic, nodes)

            # Step 3: grounded synthesis already ensures no hallucination
            best_context = []
            for c in nodes:
                txt = _get_content_safe(c)
                if txt:
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
                        "score": c.score,
                        "page": page,
                        "section": section,
                    })

            # Step 4: fact-check; if unsupported or insufficient, do a broader, avoid-aware search
            supported = _verify_supported(Settings.llm, topic, answer_text, [bc["context"] for bc in best_context])
            if (not supported) or (not answer_text or answer_text.strip().lower() == "insufficient evidence"):
                try:
                    vr = VectorIndexRetriever(index=vector_index, similarity_top_k=40)
                except Exception:
                    vr = retriever
                add_nodes = vr.retrieve(QueryBundle(query_str=topic))
                union = list(nodes) + list(add_nodes)
                dedup2 = {}
                for n in union:
                    try:
                        nid = getattr(n.node, 'node_id', None) or getattr(n.node, 'id_', None) or id(n.node)
                    except Exception:
                        nid = id(n)
                    if nid not in dedup2 or ((n.score or 0.0) > (dedup2[nid].score or 0.0)):
                        dedup2[nid] = n
                nodes = _filter_by_sections(list(dedup2.values()), prefer, avoid, top_k=12)
                answer_text = _synthesize_grounded(Settings.llm, topic, nodes)

            if str(possible_options).lower() != "none":
                code = _extract_code_with_options(Settings.llm, answer_text, possible_options)
            else:
                code = ""

            result = {
                "query": {"topic": topic, "possible_options": possible_options},
                "question": topic,
                "answer": answer_text,
                "code": code,
                "best_context": best_context,
                "plan": plan,
            }
            results_by_topic[topic] = {"answer": answer_text, "best_context": list(best_context)}
            return result

        # Execute in attack order
        topic_to_options = {q["topic"]: q.get("possible_options", "None") for q in QUERIES}
        results_ordered: List[Dict] = [None] * len(attack)

        def idx_of(t: str) -> int:
            try:
                return attack.index(t)
            except Exception:
                return 0

        with ThreadPoolExecutor(max_workers=min(8, max(1, os.cpu_count() or 4))) as pool:
            futs = {}
            for t in attack:
                futs[pool.submit(run_one, t, topic_to_options.get(t, "None"))] = t
            for f in as_completed(futs):
                try:
                    t = futs[f]
                    res = f.result()
                    results_ordered[idx_of(t)] = res
                except Exception:
                    continue

        # Fill any missing entries
        for i, t in enumerate(attack):
            if results_ordered[i] is None:
                results_ordered[i] = {
                    "query": {"topic": t, "possible_options": topic_to_options.get(t, "None")},
                    "question": t,
                    "answer": "ERROR: missing result",
                    "code": "",
                    "best_context": [],
                    "plan": plan,
                }

        raportGenerator.generate_partial_report(file, {}, results_ordered)

        # Optional evaluation per file
        if EVALUATION:
            try:
                if GROUND_TRUTH:
                    try:
                        dual = get_dual(file)
                        ha = dual.get("human_answer", [""] * len(results_ordered))
                        ground_truth = [ (ha[i] if i < len(ha) else "") or "" for i in range(len(QUERIES)) ]
                    except Exception:
                        ground_truth = GROUND_TRUTH_LIST.get(file, [""] * len(results_ordered))
                else:
                    ground_truth = [""] * len(results_ordered)
                evaluation = RAGEvaluator(
                    start_questions=[q["topic"] for q in QUERIES],
                    llm=Settings.llm,
                    results=results_ordered,
                    eval_model=EVAL_MODEL,
                    evaluate_ragas=RAGAS,
                    evaluate_geval=G_EVAL,
                    ground_truth=ground_truth,
                    filename=file,
                    embedding_model_name=EMBEDDING_MODEL,
                ).evaluate()
                for i, r in enumerate(results_ordered):
                    r["evaluation"] = evaluation[i] if i < len(evaluation) else {"ragas": None, "geval": None}
            except Exception:
                for r in results_ordered:
                    r["evaluation"] = {"ragas": None, "geval": None}
        else:
            for r in results_ordered:
                r["evaluation"] = None

    # Process files concurrently
    max_workers_files = max(1, int(CONCURRENCY) if str(CONCURRENCY).isdigit() else 3)
    with ThreadPoolExecutor(max_workers=max_workers_files) as pool:
        futs = {pool.submit(process_file, f): f for f in files}
        for _ in as_completed(futs):
            pass

    # Global reports
    raportGenerator.generate_main_report()
    execution_time = time.time() - start
    raportGenerator.generate_config_report(execution_time)
    print("END [REACT]")
    print(f"Execution time: {execution_time} seconds")
    return


if __name__ == "__main__":
    main()
