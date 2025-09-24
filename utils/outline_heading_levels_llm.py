import os
import sys
from typing import Optional

from llama_index.core import Settings


def _configure_llm() -> None:
    """Configure Settings.llm to use SMALL_MODEL (fallback to EXECUTION_MODEL) via OpenRouter."""
    import importlib
    try:
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
    "You receive a markdown outline: each line is a heading starting with one or more '#' followed by a space,"
    " then the title, optionally ending with ' (page N)'.\n"
    "Task: ONLY adjust the heading depth to reflect section/subsection hierarchy by adding more '#' at the start of lines"
    " (e.g., '# ' stays section, '## ' for subsection, '### ' for sub-subsection).\n"
    "Rules:\n"
    "- Keep the exact line order and text (titles and '(page N)' suffix) unchanged, except for increasing the count of leading '#'.\n"
    "- Do NOT remove, rename, or reorder lines.\n"
    "- Do NOT add new lines.\n"
    "- Only change heading levels by adding '#' where appropriate.\n\n"
    "Return ONLY the adjusted markdown outline, line-for-line.\n\n"
    "Input outline:\n\n{outline}\n"
)


def adjust_outline_heading_levels(md_outline: str) -> str:
    _configure_llm()
    llm = Settings.llm
    result = f"{llm.complete(PROMPT.format(outline=md_outline))!s}".strip()
    return result


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python utils/outline_heading_levels_llm.py /path/to/section_tree.md /path/to/section_tree_adjusted.md")
        sys.exit(1)
    in_path = sys.argv[1]
    out_path = sys.argv[2]
    if not os.path.exists(in_path):
        raise FileNotFoundError(in_path)
    with open(in_path, 'r', encoding='utf-8') as f:
        outline = f.read()
    adjusted = adjust_outline_heading_levels(outline)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(adjusted)
    print(out_path)


if __name__ == '__main__':
    main()


