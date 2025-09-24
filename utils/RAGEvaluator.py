import json
import os
import time
import random
import numpy as np
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate as ragasEvaluate
from ragas.metrics import (
    Faithfulness,
    SemanticSimilarity,
    LLMContextPrecisionWithoutReference,
    AnswerCorrectness,
)
from ragas import SingleTurnSample
from ragas import EvaluationDataset as RagasEvaluationDataset

from deepeval.dataset import EvaluationDataset
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    AnswerRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase

try:
    from utils.SebaEvaluator import SebaEvaluator, parse_seba_csv
except Exception:
    SebaEvaluator = None  # type: ignore
    parse_seba_csv = None  # type: ignore


class RAGEvaluator:
    def __init__(self, start_questions, llm, results, eval_model, evaluate_ragas, evaluate_geval, ground_truth, filename, embedding_model_name, evaluate_seba: bool = False):
        self.start_questions = start_questions
        self.llm = llm
        self.results = results
        self.eval_model = eval_model
        self.evaluate_ragas = evaluate_ragas
        self.evaluate_geval = evaluate_geval
        self.evaluate_seba = evaluate_seba
        self.ground_truth = ground_truth
        self.filename = filename
        self.embedding_model_name = embedding_model_name
        self.samples = self.create_samples()
        # Preload Seba criteria once if available
        self.seba_criteria = []
        if parse_seba_csv is not None:
            try:
                defs = parse_seba_csv()
                self.seba_criteria = defs.get("criteria", []) if isinstance(defs, dict) else []
            except Exception:
                self.seba_criteria = []
        
    def create_samples(self):
        # Build samples only for codified questions (possible_options != "none")
        ragas_samples = []
        geval_samples = []
        self.codified_indices = []  # indices of results that are codified
        self.codified_q_texts = []
        self.codified_a_texts = []
        for i, r in enumerate(self.results):
            is_codified = str(r.get("query", {}).get("possible_options", "none")).lower() != "none"
            if not is_codified:
                continue
            self.codified_indices.append(i)
            answer_text = r.get("code", r.get("answer", ""))
            topic = r.get("query", {}).get("topic", "")
            contexts = r.get("contexts", [])
            gt = self.ground_truth[i] if i < len(self.ground_truth) else ""

            ragas_samples.append(
                SingleTurnSample(
                    user_input=topic,
                    retrieved_contexts=contexts,
                    response=answer_text,
                    reference=gt,
                )
            )
            geval_samples.append(
                LLMTestCase(
                    input=topic,
                    actual_output=answer_text,
                    expected_output=gt,
                    retrieval_context=contexts,
                )
            )
            self.codified_q_texts.append(topic)
            self.codified_a_texts.append(answer_text)
        return {"ragas": ragas_samples, "geval": geval_samples}

    def RAGAS(self):
        print("\nRAGAS evaluation")
        dataset = RagasEvaluationDataset(samples=self.samples["ragas"])
        try:
            from config.config_keys import OPENROUTER_API_KEY as CFG_OR_KEY, OPENAI_API_KEY as CFG_OAI_KEY
        except Exception:
            CFG_OR_KEY = None
            CFG_OAI_KEY = None
        api_key = (CFG_OR_KEY or CFG_OAI_KEY or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY"))
        evaluator_llm = LangchainLLMWrapper(
            ChatOpenAI(
                model=self.eval_model,
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
                temperature=0,
                max_tokens=256,
                # Encourage JSON-only outputs to reduce RAGAS parser errors
                response_format={"type": "json_object"},
            )
        )
        evaluator_embeddings_model = OpenAIEmbeddings(model=self.embedding_model_name, api_key=CFG_OAI_KEY or os.getenv("OPENAI_API_KEY"))
        evaluator_embeddings = LangchainEmbeddingsWrapper(evaluator_embeddings_model)
        metrics = [
            Faithfulness(llm=evaluator_llm),
            SemanticSimilarity(embeddings=evaluator_embeddings),
            LLMContextPrecisionWithoutReference(llm=evaluator_llm),
            AnswerCorrectness(llm=evaluator_llm, embeddings=evaluator_embeddings),
        ]
        # Evaluate with retries for rows producing NaNs (Gemini can return unparsable outputs sporadically)
        def evaluate_subset(idxs: list[int] | None):
            if idxs is None:
                ds = dataset
            else:
                # Build a subset dataset preserving order mapping to original indices
                subset = [self.samples["ragas"][i] for i in idxs]
                ds = RagasEvaluationDataset(samples=subset)
            # Add basic retry/backoff to tolerate transient judge failures
            last_exc = None
            for attempt in range(4):
                try:
                    res = ragasEvaluate(dataset=ds, metrics=metrics)
                    return res.to_pandas()
                except Exception as e:
                    last_exc = e
                    time.sleep(min(2.0 * (1.8 ** attempt), 15.0) + random.uniform(0.0, 0.5))
            # On persistent failure, return an empty frame with expected columns to keep pipeline moving
            import pandas as _pd
            columns = [
                'question', 'contexts', 'answer', 'reference',
                'faithfulness', 'semantic_similarity',
                'llm_context_precision_without_reference', 'answer_correctness'
            ]
            return _pd.DataFrame(columns=columns)

        dataframe = evaluate_subset(None)

        # Columns to ensure are valid (no NaNs)
        target_cols = [
            'faithfulness',
            'semantic_similarity',
            'llm_context_precision_without_reference',
            'answer_correctness',
        ]

        def find_bad_indices(df):
            bad = []
            for i in range(len(df)):
                for col in target_cols:
                    if col in df.columns:
                        v = df.at[i, col]
                        try:
                            fv = float(v)
                            if np.isnan(fv) or np.isinf(fv):
                                bad.append(i)
                                break
                        except Exception:
                            bad.append(i)
                            break
            # unique preserve order
            seen = set()
            out = []
            for i in bad:
                if i not in seen:
                    seen.add(i)
                    out.append(i)
            return out

        max_retries = 4
        attempt = 0
        bad = find_bad_indices(dataframe)
        while bad and attempt < max_retries:
            attempt += 1
            # small backoff
            time.sleep(1.0 * attempt + random.uniform(0.0, 0.5))
            # re-eval only failing rows
            sub_df = evaluate_subset(bad)
            # write back into main df
            for j, orig_idx in enumerate(bad):
                if j < len(sub_df):
                    for col in target_cols:
                        if col in sub_df.columns:
                            dataframe.at[orig_idx, col] = sub_df.at[j, col]
            bad = find_bad_indices(dataframe)

        # If still bad after retries, coerce remaining NaNs to 0.0
        if bad:
            for orig_idx in bad:
                for col in target_cols:
                    if col in dataframe.columns:
                        try:
                            fv = float(dataframe.at[orig_idx, col])
                            if np.isnan(fv) or np.isinf(fv):
                                dataframe.at[orig_idx, col] = 0.0
                        except Exception:
                            dataframe.at[orig_idx, col] = 0.0
        # No post-fallback needed; only the selected metrics are kept
        columns_to_include = dataframe.columns[4:]
        samples = [
            {col: row[col] for col in columns_to_include}
            for _, row in dataframe.iterrows()
        ]
        obj = {"samples": samples}
        with open(f"ragas_result.json", 'w', encoding='utf-8') as file:
            json.dump(obj, file, ensure_ascii=False, indent=4)
        return samples

    def DEEPEVAL(self):
        print("\nDEEPEVAL evaluation")
        metrics = [
            ContextualPrecisionMetric(threshold=0.5, model=self.eval_model, include_reason=False, verbose_mode=False),
            ContextualRecallMetric(threshold=0.5, model=self.eval_model, include_reason=False, verbose_mode=False),
            ContextualRelevancyMetric(threshold=0.5, model=self.eval_model, include_reason=False, verbose_mode=False),
            AnswerRelevancyMetric(threshold=0.5, model=self.eval_model, include_reason=False, verbose_mode=False),
            FaithfulnessMetric(threshold=0.5, model=self.eval_model, include_reason=False, verbose_mode=False),
            ]
        dataset = EvaluationDataset(test_cases=self.samples["geval"])
        dataset_result = dataset.evaluate(metrics) 
        result = []
        for test_result in dataset_result.test_results:
            scores = {}
            for metric_data in test_result.metrics_data:
                scores[metric_data.name.lower().replace(" ", "_")] = metric_data.score
            result.append(scores)
        return result

    def evaluate(self):
        geval_result = self.DEEPEVAL() if self.evaluate_geval else []
        ragas_result = self.RAGAS() if self.evaluate_ragas else []
        # SEBA (LLM-as-judge) over the same codified subset
        seba_result = []
        if self.evaluate_seba and SebaEvaluator is not None and self.seba_criteria:
            try:
                from config.config_keys import OPENROUTER_API_KEY as CFG_OR_KEY, OPENAI_API_KEY as CFG_OAI_KEY
            except Exception:
                CFG_OR_KEY = None
                CFG_OAI_KEY = None
            api_key = (CFG_OR_KEY or CFG_OAI_KEY or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY"))
            try:
                seba = SebaEvaluator(model=self.eval_model, api_key=api_key, base_url="https://openrouter.ai/api/v1", criteria=self.seba_criteria)
                for i in getattr(self, "codified_indices", []):
                    r = self.results[i]
                    topic = r.get("query", {}).get("topic", "")
                    answer_text = r.get("code", r.get("answer", ""))
                    contexts = r.get("contexts", [])
                    gt = self.ground_truth[i] if i < len(self.ground_truth) else ""
                    out = seba.evaluate_sample(query_text=topic, ground_truth=gt, ai_answer=answer_text, contexts=contexts)
                    seba_result.append(out)
            except Exception:
                seba_result = []
        # Scatter results back to original indices; non-codified => None
        total = len(self.results)
        evals = [{"ragas": None, "geval": None, "seba": None} for _ in range(total)]
        for j, i in enumerate(getattr(self, "codified_indices", [])):
            evals[i]["ragas"] = ragas_result[j] if j < len(ragas_result) else None
            evals[i]["geval"] = geval_result[j] if j < len(geval_result) else None
            if j < len(seba_result):
                evals[i]["seba"] = seba_result[j]
        return evals

