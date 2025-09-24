import os
import sys
import re
from typing import List

from utils.LlamaCloudJSONParser import LlamaCloudJSONParser

INPUT_DIR = os.path.join('.', 'input')


def pick_first_pdf() -> str:
    for name in sorted(os.listdir(INPUT_DIR)):
        if name.lower().endswith('.pdf'):
            return os.path.join(INPUT_DIR, name)
    raise RuntimeError('No PDFs found in input/')


def first_heading(md_text: str) -> str | None:
    for line in md_text.splitlines():
        m = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
        if m:
            return m.group(2).strip()
    return None


def main() -> None:
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        pdf_path = arg if arg.lower().endswith('.pdf') else os.path.join(INPUT_DIR, f"{arg}.pdf")
    else:
        pdf_path = pick_first_pdf()
    cloud_key = os.getenv('LLAMA_CLOUD_API_KEY')
    if not cloud_key:
        print('LLAMA_CLOUD_API_KEY not set')
        sys.exit(1)
    parser = LlamaCloudJSONParser(api_key=cloud_key)
    docs = parser.parse_to_documents(pdf_path)
    print(f"Parsed OK: {os.path.basename(pdf_path)}")
    print(f"Documents (pages): {len(docs)}")
    missing_page = [i for i, d in enumerate(docs, start=1) if 'page_label' not in (d.metadata or {})]
    print(f"Pages with page_label: {len(docs) - len(missing_page)}/{len(docs)}")
    print("Sample (first 5):")
    for i, d in enumerate(docs[:5], start=1):
        page = (d.metadata or {}).get('page_label')
        heading = first_heading(d.text) or '(no heading)'
        print(f"  p={page} | heading={heading}")


if __name__ == '__main__':
    main()


