import os
import sys
from llama_index.core import Settings
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.schema import QueryBundle

from config.config import (
    API,
    INPUT_PATH,
    STORAGE_PATH,
    EXECUTION_MODEL,
    EMBEDDING_MODEL,
)
from config.config_keys import (
    OPENAI_API_KEY,
    OPENROUTER_API_KEY,
)
from utils.VectorQueryEngineCreator import VectorQueryEngineCreator


def main() -> None:
    if API != "openrouter":
        raise ValueError("This script assumes API=='openrouter'")
    if len(sys.argv) < 2:
        print('Usage: python -m utils.check_vqe "[file_basename]" [optional query]')
        sys.exit(1)
    file_basename = sys.argv[1]
    query_text = sys.argv[2] if len(sys.argv) > 2 else "What AI/ML algorithm was used?"

    # Configure LLMs
    Settings.llm = OpenAILike(
        model=EXECUTION_MODEL,
        api_base="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        is_chat_model=True,
    )
    Settings.embed_model = OpenAIEmbedding(
        model=EMBEDDING_MODEL,
        api_base="https://api.openai.com/v1",
        api_key=OPENAI_API_KEY,
    )

    creator = VectorQueryEngineCreator(
        llama_parse_api_key=os.getenv("LLAMA_CLOUD_API_KEY", ""),
        cohere_api_key=os.getenv("COHERE_API_KEY", ""),
        input_path=INPUT_PATH,
        storage_path=STORAGE_PATH,
        cohere_rerank=False,
        embedding_model_name=EMBEDDING_MODEL,
    )
    query_engine = creator.get_query_engine(file_basename)
    bundle = QueryBundle(query_str=query_text)
    response = query_engine.query(bundle)

    print("\nQuery:", query_text)
    print("Top contexts (top_section > section | page | para | score):")
    for i, nws in enumerate(response.source_nodes[:5], start=1):
        md = nws.node.metadata or {}
        top_section = md.get('top_section')
        section = md.get('section')
        page = md.get('page_label') or md.get('page')
        para = md.get('paragraph_index')
        pfx = f"{top_section} > {section}" if top_section or section else None
        print(f"{i}. {pfx} | p={page} | ¶={para} | score={nws.score:.4f}")
    print("\nSnippet:")
    print(str(response)[:600])


if __name__ == "__main__":
    main()


