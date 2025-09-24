import os
import sys
from typing import Optional

from llama_index.core import Settings


def _configure_small_llm() -> None:
    """Configure Settings.llm to use SMALL_MODEL via OpenRouter."""
    try:
        import importlib
        cfg = importlib.import_module('config.config')
        keys = importlib.import_module('config.config_keys')
        from llama_index.llms.openrouter import OpenRouter
    except Exception as e:
        raise RuntimeError(f"Missing config or OpenRouter client: {e}")

    model_name = getattr(cfg, 'SMALL_MODEL', None) or getattr(cfg, 'EXECUTION_MODEL', None) or 'qwen/qwen-2.5-7b-instruct'
    api_key = getattr(keys, 'OPENROUTER_API_KEY', '')

    Settings.llm = OpenRouter(
        api_key=api_key,
        model=model_name,
        max_tokens=512,
        context_window=4096,
    )


PROMPT = (
    "You will receive a paper's markdown content with inline page markers like `<!--PAGE:3-->`.\n"
    "Rewrite ONLY the section headings so that true sections and subsections are properly marked with markdown '#' levels.\n"
    "Rules:\n"
    "- Use '# ' for top-level sections (Abstract, Introduction, Related Work, Methods, Experiments, Results, Discussion, Conclusion, References, etc.).\n"
    "- Use '## ' for subsections under the last seen '#', and '### ' for sub-subsections when clearly indicated (e.g., numbering 2.1, 2.1.1).\n"
    "- Keep the original order of content; do NOT invent or remove body text lines.\n"
    "- Keep inline page markers `<!--PAGE:n-->` exactly where they are. They help keep pages aligned.\n"
    "- Remove spurious headings for non-sections (e.g., Table/Figure/Algorithm captions). Convert those to plain text lines (no '#').\n"
    "- Standardize titles: trim whitespace, fix casing minimally (title case for section heads), remove trailing punctuation.\n"
    "- Prefer structural numbering in titles when present (e.g., '2 Related Work', '2.1 Dataset'), but do not fabricate numbering.\n"
    "- Output MUST be valid markdown of the entire document, returning ONLY the rewritten markdown.\n\n"
    "Input markdown follows:\n\n{md}\n"
)


def normalize_markdown_headings(md_text: str) -> str:
    _configure_small_llm()
    llm = Settings.llm
    raw = f"{llm.complete(PROMPT.format(md=md_text))!s}".strip()
    return raw


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python utils/markdown_headings_normalize_llm.py /path/to/raw_markdown.md /path/to/out.md")
        sys.exit(1)
    in_path = sys.argv[1]
    out_path = sys.argv[2]
    if not os.path.exists(in_path):
        raise FileNotFoundError(in_path)
    with open(in_path, 'r', encoding='utf-8') as f:
        md = f.read()
    norm = normalize_markdown_headings(md)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(norm)
    print(out_path)


if __name__ == "__main__":
    main()


