import os
import re
import sys
import json
from typing import List, Dict


def parse_markdown_headings_markdown_only(md_text: str) -> List[Dict]:
    """
    Extract ONLY markdown '#' headings (levels 1..6) from the given text.
    - Ignores numeric or roman numeral prefixes for hierarchy
    - Optionally records the most recent page marker (<!--PAGE:n-->) seen before the heading
    """
    page = None
    headings: List[Dict] = []
    page_rx = re.compile(r'^<!--PAGE:(\d+?)-->$')
    header_rx = re.compile(r'^(#{1,6})\s+(.+?)\s*$')
    for line in md_text.splitlines():
        s = line.strip()
        mpage = page_rx.match(s)
        if mpage:
            try:
                page = int(mpage.group(1))
            except Exception:
                page = None
            continue
        mh = header_rx.match(line)
        if mh:
            level = len(mh.group(1))
            title = mh.group(2).strip()
            headings.append({'level': level, 'title': title, 'page': page})
    return headings


def build_tree_markdown_levels(headings: List[Dict]) -> Dict:
    """
    Build a tree using ONLY markdown heading levels for hierarchy.
    Level n attaches under the nearest previous heading with level < n; root otherwise.
    """
    root: Dict = {'title': 'ROOT', 'children': []}
    stack: List[Dict] = []
    for h in headings:
        node = {
            'title': h.get('title'),
            'level': h.get('level'),
            'page': h.get('page'),
            'children': [],
        }
        while stack and (stack[-1].get('level') or 1) >= (h.get('level') or 1):
            stack.pop()
        parent = stack[-1] if stack else root
        parent['children'].append(node)
        stack.append(node)
    return root


def write_json_tree(root: Dict, out_path: str) -> None:
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(root, f, ensure_ascii=False, indent=2)


def main():
    if len(sys.argv) < 3:
        print('Usage: build_section_tree_md_only.py "[57] paper_name" /full/path/to/raw_markdown.md')
        sys.exit(1)
    file_basename = sys.argv[1]
    raw_path = sys.argv[2]
    if not os.path.exists(raw_path):
        raise FileNotFoundError(raw_path)
    with open(raw_path, 'r', encoding='utf-8') as f:
        md_text = f.read()
    headings = parse_markdown_headings_markdown_only(md_text)
    tree = build_tree_markdown_levels(headings)
    # Persist next to the existing vector_index dir pattern used in the app
    out_dir = os.path.join('storage', 'openrouter', f"{file_basename}_vector_index")
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, 'section_tree_md_only.json')
    write_json_tree(tree, out_json)
    print(f"Wrote {out_json} with {len(headings)} markdown-only headings")


if __name__ == '__main__':
    main()


