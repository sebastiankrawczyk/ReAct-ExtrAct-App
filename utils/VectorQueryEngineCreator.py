import os
import re
import json
from typing import List, Dict, Tuple, Optional
from llama_index.llms.openai import OpenAI 
from llama_parse import LlamaParse 
from llama_index.core.node_parser import MarkdownElementNodeParser
try:
    # Prefer pure markdown structural parser (no LLM) for deterministic nodes
    from llama_index.core.node_parser import MarkdownNodeParser as PureMarkdownNodeParser
except Exception:
    PureMarkdownNodeParser = None
from llama_index.core import VectorStoreIndex, StorageContext, get_response_synthesizer, load_index_from_storage, Settings, Document
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import NodeWithScore, QueryBundle
try:
    from llama_index.core.schema import TextNode
except Exception:
    TextNode = None
try:
    from llama_index.core.postprocessor import BaseNodePostprocessor
except Exception:
    # Back-compat import path in some versions
    from llama_index.core.postprocessor.types import BaseNodePostprocessor
try:
    from pydantic import Field
    from pydantic import ConfigDict
except Exception:
    Field = None
    ConfigDict = None

from llama_index.core import SimpleDirectoryReader #sa

class VectorQueryEngineCreator:
    def __init__(self, llama_parse_api_key, cohere_api_key, input_path, storage_path, cohere_rerank, embedding_model_name, enable_section_reasoner=False, response_mode: str = 'tree_summarize'):
        self.llama_parse_api_key = llama_parse_api_key
        self.cohere_api_key = cohere_api_key
        self.input_path = input_path
        self.storage_path = storage_path
        self.cohere_rerank = cohere_rerank
        self.embedding_model_name = embedding_model_name
        # IMPORTANT: SectionReasonerPostprocessor uses LLM reasoning over section names.
        # It must NOT be enabled for naive or guided benchmarks.
        self.enable_section_reasoner = bool(enable_section_reasoner)
        self.response_mode = response_mode or 'tree_summarize'

        # Ensure OpenAI key is available in env for any downstream LLM usage
        try:
            if not os.getenv('OPENAI_API_KEY'):
                # Prefer .env or environment; avoid importing files with secrets unless explicitly present
                openai_key = os.getenv('OPENAI_API_KEY')
                if not openai_key:
                    try:
                        from config import config_keys as _keys  # type: ignore
                        openai_key = getattr(_keys, 'OPENAI_API_KEY', '')
                    except Exception:
                        openai_key = None
                if openai_key:
                    os.environ['OPENAI_API_KEY'] = openai_key
        except Exception:
            pass

    def parse_pdf_to_nodes(self, path_to_pdf):
        # Strict markdown path only; no JSON or simple-reader fallback
        documents = None
        node_parser = None
        nodes = None
        basename = os.path.basename(path_to_pdf)
        try:
            # Resolve API key from constructor or config/env; use only Llama Cloud key
            api_key = self.llama_parse_api_key or None
            if not api_key:
                api_key = os.getenv('LLAMA_CLOUD_API_KEY') or ''
            if not api_key:
                try:
                    from config.config_keys import LLAMA_CLOUD_API_KEY as CFG_LC_KEY
                    api_key = CFG_LC_KEY or api_key
                except Exception:
                    pass
            if not api_key:
                raise RuntimeError('Missing Llama Cloud API key for markdown parsing')
            print(f"[parse] {basename}: starting LlamaParse markdown conversion")
            documents = LlamaParse(
                api_key=api_key,
                result_type="markdown",
                page_separator="\n<!--PAGE:{pageNumber}-->\n",
            ).load_data(path_to_pdf)
            for d in documents:
                try:
                    d.metadata = d.metadata or {}
                    d.metadata.setdefault("source_file", basename)
                except Exception:
                    pass
            if PureMarkdownNodeParser is None:
                raise RuntimeError('Pure markdown parser unavailable (MarkdownNodeParser). Install compatible llama-index-core version.')
            node_parser = PureMarkdownNodeParser()
            nodes = node_parser.get_nodes_from_documents(documents)
            # Fallback: if parser produced zero nodes, create one TextNode per document
            if (not nodes) and TextNode is not None:
                try:
                    nodes = [TextNode(text=d.text, metadata=(d.metadata or {})) for d in documents if (d.text or '').strip()]
                except Exception:
                    nodes = []
            try:
                print(f"[parse] {basename}: parsed -> nodes={len(nodes) if nodes else 0}")
            except Exception:
                pass
        except Exception:
            # Fail fast strictly with no fallback
            documents = None
            node_parser = None
            nodes = None
        # Keep all content, including references
        if not documents:
            return [], None, None
        return documents, node_parser, nodes

    def _write_raw_markdown_and_outline(self, persist_dir: str, documents: List[Document]):
        try:
            os.makedirs(persist_dir, exist_ok=True)
            raw_path = os.path.join(persist_dir, 'raw_markdown.md')
            with open(raw_path, 'w', encoding='utf-8') as f:
                for d in documents:
                    # Write document text without injecting page markers and strip SEC annotations
                    txt = d.text or ''
                    try:
                        txt = re.sub(r"\s*\[\[SEC\s+id=[^\]]+\]\]\s*", "", txt)
                    except Exception:
                        pass
                    f.write(txt)
                    f.write("\n")
            outline_path = os.path.join(persist_dir, 'section_tree.md')
            header_rx = re.compile(r'^(#{1,6})\s+(.+?)\s*$')
            page_rx1 = re.compile(r'^==\s*(\d+)\s*==$')
            page_rx2 = re.compile(r'^<!--PAGE:(\d+?)-->$')
            with open(outline_path, 'w', encoding='utf-8') as f:
                f.write("# Section Tree\n\n")
                for d in documents:
                    current_page = None
                    for line in d.text.splitlines():
                        s = line.strip()
                        m1 = page_rx1.match(s)
                        m2 = page_rx2.match(s)
                        if m1 or m2:
                            try:
                                current_page = int((m1 or m2).group(1))
                            except Exception:
                                current_page = None
                            continue
                        m = header_rx.match(re.sub(r"\s*\[\[SEC\s+id=[^\]]+\]\]\s*", "", line))
                        if m:
                            level = len(m.group(1))
                            title = m.group(2).strip()
                            indent = '  ' * (level - 1)
                            line_out = f"{indent}{'#'*level} {title}"
                            if current_page is not None:
                                line_out += f" (page {current_page})"
                            f.write(line_out + "\n")
            try:
                print(f"[md] wrote raw markdown and outline → {persist_dir}")
            except Exception:
                pass
        except Exception:
            pass

    # ------------------- Tree normalization + annotation -------------------
    def _build_tree_from_raw(self, raw_md_text: str) -> Dict:
        try:
            # Use markdown-only heading parser as the primary method
            from utils.build_section_tree_md_only import (
                parse_markdown_headings_markdown_only,
                build_tree_markdown_levels,
            )
            headings = parse_markdown_headings_markdown_only(raw_md_text)
            return build_tree_markdown_levels(headings)
        except Exception:
            page = None
            headings = []
            for line in raw_md_text.splitlines():
                s = line.strip()
                mpage1 = re.match(r'^==\s*(\d+)\s*==$', s)
                mpage2 = re.match(r'^<!--PAGE:(\d+?)-->$', s)
                if mpage1 or mpage2:
                    try:
                        page = int((mpage1 or mpage2).group(1))
                    except Exception:
                        page = None
                    continue
                mh = re.match(r'^(#{1,6})\s+(.+?)\s*$', line)
                if mh:
                    level = len(mh.group(1))
                    title = mh.group(2).strip()
                    num = None
                    mnum = re.match(r'^(\d+(?:\.\d+)*)\.?\s+(.*)$', title)
                    if mnum:
                        num = mnum.group(1)
                        title = mnum.group(2).strip()
                    headings.append({'level': level, 'title': title, 'number': num, 'page': page})
            # build naive tree
            root = {'title': 'ROOT', 'children': []}
            index_by_number: Dict[str, Dict] = {}
            stack: List[Dict] = []
            for h in headings:
                node = {'title': h['title'], 'number': h['number'], 'level': h['level'], 'page': h['page'], 'children': []}
                if h['number']:
                    parts = h['number'].split('.')
                    if len(parts) == 1:
                        root['children'].append(node)
                    else:
                        parent_num = '.'.join(parts[:-1])
                        parent = index_by_number.get(parent_num) or root
                        parent['children'].append(node)
                    index_by_number[h['number']] = node
                else:
                    while stack and stack[-1]['level'] >= h['level']:
                        stack.pop()
                    parent = stack[-1] if stack else root
                    parent['children'].append(node)
                    stack.append(node)
            return root

    def _normalize_tree_with_llm(self, tree: Dict) -> Dict:
        try:
            llm = Settings.llm
            if llm is None:
                return tree
            prompt = (
                "Normalize a research paper section tree. Fix numbering continuity, deduplicate repeated headings, "
                "standardize titles, and preserve page fields when present. Return ONLY JSON for the tree root.\n\n"
                f"Input tree JSON follows:\n{json.dumps(tree, ensure_ascii=False)}\n"
            )
            raw = f"{llm.complete(prompt)!s}".strip()
            start = raw.find('{'); end = raw.rfind('}')
            json_str = raw[start:end+1] if start != -1 and end != -1 and end > start else raw
            obj = json.loads(json_str)
            if 'children' not in obj and isinstance(obj, list):
                obj = {'title': 'ROOT', 'children': obj}
            if 'title' not in obj:
                obj['title'] = 'ROOT'
            return obj
        except Exception:
            return tree

    def _flatten_tree(self, node: Dict, parent_path: List[int] = None, top_title: Optional[str] = None) -> List[Dict]:
        parent_path = parent_path or []
        flat: List[Dict] = []
        for idx, child in enumerate((node.get('children') or []), start=1):
            path = parent_path + [idx]
            synthetic_id = '.'.join(str(x) for x in path)
            sec_id = child.get('number') or synthetic_id
            child_top = top_title or child.get('title') if len(path) == 1 else top_title
            flat.append({
                'title': child.get('title'),
                'number': child.get('number'),
                'sec_id': sec_id,
                'level': child.get('level') or len(path),
                'page': child.get('page'),
                'top_title': child_top,
                'path': path,
            })
            flat.extend(self._flatten_tree(child, path, child_top))
        return flat

    def _normalize_title(self, s: str) -> str:
        t = re.sub(r'^[0-9]+(?:\.[0-9]+)*\.?\s+', '', s or '')
        t = re.sub(r'\s+', ' ', t).strip().lower()
        return re.sub(r'[^a-z0-9 ]+', '', t)

    def _find_match(self, flat: List[Dict], title: str, page: Optional[int]) -> Optional[Dict]:
        target = self._normalize_title(title)
        candidates = [e for e in flat if self._normalize_title(e.get('title')) == target]
        if not candidates:
            return None
        if page is None:
            return candidates[0]
        best = None
        best_dist = None
        for e in candidates:
            p = e.get('page')
            if p is None:
                continue
            dist = abs(int(p) - int(page))
            if best_dist is None or dist < best_dist:
                best = e
                best_dist = dist
        return best or candidates[0]

    def _annotate_raw_markdown(self, persist_dir: str, normalized_flat: List[Dict]) -> None:
        raw_path = os.path.join(persist_dir, 'raw_markdown.md')
        out_path = os.path.join(persist_dir, 'raw_markdown_annotated.md')
        if not os.path.exists(raw_path):
            return
        current_page: Optional[int] = None
        header_rx = re.compile(r'^(#{1,6})\s+(.+?)\s*$')
        with open(raw_path, 'r', encoding='utf-8') as f, open(out_path, 'w', encoding='utf-8') as out:
            for line in f:
                s = line.strip()
                mpage1 = re.match(r'^==\s*(\d+)\s*==$', s)
                mpage2 = re.match(r'^<!--PAGE:(\d+?)-->$', s)
                if mpage1 or mpage2:
                    try:
                        current_page = int((mpage1 or mpage2).group(1))
                    except Exception:
                        current_page = None
                    out.write(line)
                    continue
                mh = header_rx.match(line)
                if mh:
                    level = len(mh.group(1))
                    title = mh.group(2).strip()
                    match = self._find_match(normalized_flat, title, current_page)
                    if match:
                        tag = f" [[SEC id={match['sec_id']}|page={match.get('page') if match.get('page') is not None else (current_page if current_page is not None else '')}]]"
                        out.write(f"{'#'*level} {title}{tag}\n")
                        continue
                out.write(line)

    def _build_normalized_tree_and_annotate(self, persist_dir: str) -> None:
        try:
            raw_path = os.path.join(persist_dir, 'raw_markdown.md')
            if not os.path.exists(raw_path):
                return
            with open(raw_path, 'r', encoding='utf-8') as f:
                raw_md = f.read()
            det_tree = self._build_tree_from_raw(raw_md)
            norm_tree = self._normalize_tree_with_llm(det_tree)
            norm_json_path = os.path.join(persist_dir, 'section_tree_llm.json')
            try:
                with open(norm_json_path, 'w', encoding='utf-8') as f:
                    json.dump(norm_tree, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
            flat = self._flatten_tree(norm_tree)
            self._annotate_raw_markdown(persist_dir, flat)
        except Exception:
            pass

    def _load_normalized_flat(self, persist_dir: str) -> List[Dict]:
        try:
            norm_json_path = os.path.join(persist_dir, 'section_tree_llm.json')
            if not os.path.exists(norm_json_path):
                return []
            with open(norm_json_path, 'r', encoding='utf-8') as f:
                norm_tree = json.load(f)
            return self._flatten_tree(norm_tree)
        except Exception:
            return []

    def _parse_annotated_anchors(self, persist_dir: str, normalized_flat: List[Dict]) -> List[Dict]:
        anchors: List[Dict] = []
        annotated = os.path.join(persist_dir, 'raw_markdown_annotated.md')
        if not os.path.exists(annotated):
            return anchors
        top_by_sec_prefix: Dict[str, str] = {}
        for e in normalized_flat:
            sec_id = (e.get('sec_id') or '').split('.')
            if sec_id:
                top_by_sec_prefix[sec_id[0]] = e.get('top_title') or e.get('title')
        rx = re.compile(r'^(#{1,6})\s+(.+?)\s*\[\[SEC\s+id=([^\]|]+)\|page=([0-9]*)\]\]\s*$')
        with open(annotated, 'r', encoding='utf-8') as f:
            for ln, line in enumerate(f, start=1):
                m = rx.match(line.rstrip('\n'))
                if not m:
                    continue
                level = len(m.group(1))
                title = m.group(2).strip()
                sec_id = m.group(3).strip()
                ptxt = m.group(4).strip()
                page = int(ptxt) if ptxt.isdigit() else None
                top_title = top_by_sec_prefix.get(sec_id.split('.')[0])
                anchors.append({'line': ln, 'level': level, 'title': title, 'sec_id': sec_id, 'page': page, 'top_title': top_title})
        anchors.sort(key=lambda a: (a.get('page') if a.get('page') is not None else 10**9, a['line']))
        return anchors

    def _apply_annotated_mapping(self, persist_dir: str, nodes: List) -> None:
        try:
            normalized_flat = self._load_normalized_flat(persist_dir)
            anchors = self._parse_annotated_anchors(persist_dir, normalized_flat)
            if not anchors:
                return
            debug = bool(os.getenv('DEBUG_NODE_META'))
            for n in nodes:
                try:
                    md = getattr(n, 'metadata', None)
                    if md is None:
                        continue
                    page = md.get('page_label')
                    if not isinstance(page, (int, float)):
                        continue
                    # choose last anchor with page <= node page
                    candidates = [a for a in anchors if isinstance(a.get('page'), (int, float)) and a['page'] <= page]
                    chosen = max(candidates, key=lambda a: a['page']) if candidates else None
                    if not chosen:
                        # fallback absolute nearest
                        with_pages = [a for a in anchors if isinstance(a.get('page'), (int, float))]
                        if with_pages:
                            chosen = min(with_pages, key=lambda a: abs(a['page'] - page))
                    if not chosen:
                        continue
                    # set normalized fields, overriding weak values only
                    def is_weak(text: Optional[str]) -> bool:
                        t = self._normalize_title(text or '')
                        return not t or len(t) < 3
                    if is_weak(md.get('section')):
                        md['section'] = chosen.get('title')
                    md.setdefault('section_id', chosen.get('sec_id'))
                    if chosen.get('top_title'):
                        md.setdefault('top_section', chosen.get('top_title'))
                    if debug:
                        try:
                            first_line = (n.get_content() or '').split('\n')[0][:80]
                            print(f"[DEBUG anchor] p={page} -> sec='{md.get('section')}' id='{md.get('section_id')}' top='{md.get('top_section')}' via id={chosen.get('sec_id')} title='{chosen.get('title')}'")
                        except Exception:
                            pass
                except Exception:
                    continue
        except Exception:
            return

    def _apply_normalized_flat_sections(self, persist_dir: str, nodes: List) -> None:
        try:
            flat_path = os.path.join(persist_dir, 'section_flat_llm.json')
            if not os.path.exists(flat_path):
                return
            with open(flat_path, 'r', encoding='utf-8') as f:
                flat = json.load(f) or []
            # index by page for quick lookup
            with_pages = [e for e in flat if isinstance(e.get('page'), (int, float))]
            with_pages.sort(key=lambda e: e.get('page'))
            # collect top-level sections (path length == 1)
            tops = [e for e in with_pages if isinstance(e.get('path'), list) and len(e['path']) == 1]
            tops.sort(key=lambda e: e.get('page'))
            if not with_pages:
                return
            def choose_for_page(pg: int) -> Optional[Dict]:
                # latest section with page <= pg; fallback to nearest absolute
                chosen = None
                for e in with_pages:
                    if e['page'] <= pg:
                        chosen = e
                    else:
                        break
                if chosen is not None:
                    return chosen
                # fallback nearest
                return min(with_pages, key=lambda e: abs(e['page'] - pg))
            for n in nodes:
                try:
                    md = getattr(n, 'metadata', None)
                    if md is None:
                        continue
                    pg = md.get('page_label') or md.get('page')
                    if not isinstance(pg, (int, float)):
                        continue
                    best = choose_for_page(int(pg))
                    if not best:
                        continue
                    # Assign only the closest normalized section title
                    md['section'] = best.get('title')
                    # Remove any prior top_section to keep only one label
                    if 'top_section' in md:
                        try:
                            del md['top_section']
                        except Exception:
                            pass
                except Exception:
                    continue
        except Exception:
            return

    def _enrich_nodes_metadata(self, nodes):
        # Only normalize page label and set paragraph index per page (no section heuristics here)
        current_page = None
        paragraph_counter = 0
        for n in nodes:
            try:
                md = getattr(n, 'metadata', None)
                text = n.get_content() if hasattr(n, 'get_content') else None
                if md is None or not text:
                    continue
                debug = bool(os.getenv('DEBUG_NODE_META'))
                # standardize page label and manage per-page paragraph counter
                if 'page_label' in md:
                    page = md['page_label']
                elif 'page' in md:
                    page = md['page']
                    md['page_label'] = page
                else:
                    page = None
                if page != current_page:
                    current_page = page
                    paragraph_counter = 0
                # attach paragraph index per page only when page is known
                if page is not None:
                    paragraph_counter += 1
                    md.setdefault('paragraph_index', paragraph_counter)
                if debug:
                    try:
                        first_line = (text.splitlines()[0].strip() if text else '')[:80]
                        print(f"[DEBUG enrich] p={md.get('page_label')} ¶={md.get('paragraph_index')} | {first_line}")
                    except Exception:
                        pass
            except Exception:
                continue

    def _assign_sections_from_docs(self, documents, nodes):
        # Build heading outlines per document, track doc_id and page markers
        outlines = {}
        heading_rx = re.compile(r'^(#{1,6})\s+(.+)$')
        page_rx1 = re.compile(r'^==\s*(\d+)\s*==$')
        page_rx2 = re.compile(r'^<!--PAGE:(\d+?)-->$')
        order = 1
        for d in documents:
            heads = []
            text = d.text
            page_markers: List[Tuple[int,int]] = []  # (line_num, page)
            current_page: Optional[int] = None
            for ln, line in enumerate(text.splitlines(), start=1):
                m = heading_rx.match(line.strip())
                if m:
                    level = len(m.group(1))
                    title = m.group(2).strip()
                    heads.append((ln, level, title))
                sp = line.strip()
                m1 = page_rx1.match(sp)
                m2 = page_rx2.match(sp)
                if m1 or m2:
                    try:
                        current_page = int((m1 or m2).group(1))
                    except Exception:
                        current_page = None
                    page_markers.append((ln, current_page))
            doc_key = getattr(d, 'doc_id', getattr(d, 'id_', id(d)))
            outlines[doc_key] = {
                'text': text,
                'heads': heads,
                'order': order,
                'page_markers': page_markers,
            }
            order += 1

        def nearest_for_line(doc_key, line_num: int):
            info = outlines.get(doc_key)
            if not info:
                return None, None
            nearest = None
            top = None
            for (ln, lvl, title) in info['heads']:
                if ln <= line_num:
                    nearest = (lvl, title)
                    if lvl == 1:
                        top = title
            if top is None and info['heads']:
                top = info['heads'][0][2]
            return nearest, top

        def page_for_line(doc_key: str, line_num: int) -> Optional[int]:
            info = outlines.get(doc_key)
            if not info:
                return None
            markers: List[Tuple[int,int]] = info.get('page_markers') or []
            # choose last marker STRICTLY BEFORE line_num (assign-then-switch)
            chosen = None
            for ln, pg in markers:
                if ln < line_num:
                    chosen = pg
                else:
                    break
            return chosen

        for n in nodes:
            try:
                md = getattr(n, 'metadata', None)
                text = n.get_content() if hasattr(n, 'get_content') else None
                if md is None:
                    continue
                debug = bool(os.getenv('DEBUG_NODE_META'))
                # Prefer precise mapping via ref_doc_id and start_char_idx
                doc_key = getattr(n, 'ref_doc_id', getattr(n, 'doc_id', getattr(n, 'id_', None)))
                start_idx = getattr(n, 'start_char_idx', None)
                if doc_key in outlines and isinstance(start_idx, int) and start_idx >= 0:
                    doc_text = outlines[doc_key]['text']
                    line_num = doc_text.count('\n', 0, min(start_idx, len(doc_text))) + 1
                    nearest, top = nearest_for_line(doc_key, line_num)
                    page_label = page_for_line(doc_key, line_num)
                else:
                    # Fallback to substring search in any doc
                    best_key = None
                    best_pos = None
                    snippet = (text or '')[:200]
                    for k, info in outlines.items():
                        pos = info['text'].find(snippet) if snippet else -1
                        if pos != -1 and (best_pos is None or pos < best_pos):
                            best_pos = pos
                            best_key = k
                    if best_key is None:
                        continue
                    line_num = outlines[best_key]['text'].count('\n', 0, best_pos) + 1
                    nearest, top = nearest_for_line(best_key, line_num)
                    doc_key = best_key
                    page_label = page_for_line(doc_key, line_num)
                # If page is known, prefer the nearest heading on the same page (page-scoped)
                if page_label is not None and doc_key in outlines:
                    info = outlines[doc_key]
                    same_page_nearest = None
                    for (ln, lvl, title) in info['heads']:
                        if ln <= line_num:
                            h_page = page_for_line(doc_key, ln)
                            if h_page == page_label:
                                same_page_nearest = (lvl, title)
                    if same_page_nearest is not None:
                        nearest = same_page_nearest
                if 'section' not in md:
                    if nearest:
                        md['section'] = nearest[1]
                    elif top:
                        md['section'] = top
                if top and 'top_section' not in md:
                    md['top_section'] = top
                # attach stable doc identifier for downstream use
                if 'doc_id' not in md and doc_key is not None:
                    md['doc_id'] = doc_key
                # attach page label if detected; fallback to per-doc order
                if 'page_label' not in md:
                    if page_label is not None:
                        md['page_label'] = page_label
                    elif doc_key in outlines:
                        md['page_label'] = outlines[doc_key].get('order')
                if debug:
                    try:
                        first_line = (text or '').splitlines()[0].strip() if text else ''
                        print(f"[DEBUG assign] p={md.get('page_label')} sec='{md.get('section')}' top='{md.get('top_section')}' doc_id='{md.get('doc_id')}' | {first_line[:80]}")
                    except Exception:
                        pass
            except Exception:
                continue

        # Backfill page_label per document to avoid gaps
        last_seen_per_doc: Dict = {}
        for n in nodes:
            try:
                md = getattr(n, 'metadata', None)
                if md is None:
                    continue
                doc_key = md.get('doc_id')
                if not doc_key:
                    continue
                val = md.get('page_label')
                if isinstance(val, (int, float)):
                    last_seen_per_doc[doc_key] = int(val)
                    continue
                if doc_key in last_seen_per_doc:
                    md['page_label'] = last_seen_per_doc[doc_key]
                else:
                    md['page_label'] = 1
            except Exception:
                continue

    # ------------------- Section-aware postprocessor -------------------
    class SectionReasonerPostprocessor(BaseNodePostprocessor):
        # Pydantic v2 config to allow arbitrary attrs
        if ConfigDict is not None:
            model_config = ConfigDict(arbitrary_types_allowed=True)
        # Pydantic field for cache (no leading underscore)
        policy_cache: Dict[str, Tuple[List[str], List[str]]] = Field(default_factory=dict) if Field else {}

        @staticmethod
        def _collect_sections(nodes: List[NodeWithScore]) -> List[str]:
            sections = []
            for n in nodes:
                try:
                    sec = (n.node.metadata or {}).get('section')
                    if sec and sec not in sections:
                        sections.append(sec)
                except Exception:
                    continue
            return sections[:50]

        def _reason_policy(self, query: str, sections: List[str]) -> Tuple[List[str], List[str]]:
            key = f"{query}\n{sections}"
            cache = getattr(self, "policy_cache", {})
            if key in cache:
                return cache[key]
            # Lightweight prompt to infer preferred/avoided sections
            prompt = (
                "You act as a retrieval planner. Given a user query and available paper section names, "
                "decide which sections are most likely to contain the answer and which sections should be avoided.\n"
                "General guidance (not strict rules): content-specific questions are usually found in Methods/Approach, "
                "Experiments/Results/Evaluation, Dataset/Data, or Conclusions; generic sections like Related Work, "
                "References, Acknowledgments, Appendix often do not contain the target facts unless explicitly asked.\n\n"
                f"Query: {query}\n"
                f"Sections: {sections}\n\n"
                "Return ONLY valid JSON with two arrays, no prose:\n"
                "{\n  \"prefer\": [\"section1\", ...],\n  \"avoid\": [\"sectionA\", ...]\n}"
            )
            try:
                llm = Settings.llm
                raw = f"{llm.complete(prompt)!s}".strip()
                start = raw.find('{'); end = raw.rfind('}')
                json_str = raw[start:end+1] if start != -1 and end != -1 and end > start else raw
                import json
                obj = json.loads(json_str)
                prefer = [s for s in obj.get('prefer', []) if isinstance(s, str)]
                avoid = [s for s in obj.get('avoid', []) if isinstance(s, str)]
            except Exception:
                prefer, avoid = [], []
            cache[key] = (prefer, avoid)
            try:
                setattr(self, "policy_cache", cache)
            except Exception:
                pass
            return prefer, avoid

        def _score_boost(self, section: str, prefer: List[str], avoid: List[str]) -> float:
            if not section:
                return 1.0
            sec_lower = section.lower()
            if any(sec_lower == p.lower() for p in prefer):
                return 1.25  # boost preferred sections
            if any(sec_lower == a.lower() for a in avoid):
                return 0.6   # penalize avoided sections
            return 1.0

        # Some versions expect _postprocess_nodes; others call postprocess_nodes
        def _postprocess_nodes(self, nodes: List[NodeWithScore], query_bundle: QueryBundle) -> List[NodeWithScore]:
            if not nodes or not query_bundle:
                return nodes
            try:
                sections = self._collect_sections(nodes)
                prefer, avoid = self._reason_policy(query_bundle.query_str, sections)
                for nws in nodes:
                    try:
                        sec = (nws.node.metadata or {}).get('section')
                        factor = self._score_boost(sec, prefer, avoid)
                        nws.score = (nws.score or 0.0) * factor
                    except Exception:
                        continue
            except Exception:
                # fail open
                return nodes
            return nodes

        def postprocess_nodes(self, nodes: List[NodeWithScore], query_bundle: QueryBundle) -> List[NodeWithScore]:
            return self._postprocess_nodes(nodes, query_bundle)

    def create_vector_index(self, documents, node_parser, nodes):
        if node_parser and nodes:
            # Support both element-based and pure markdown parsers
            if hasattr(node_parser, 'get_nodes_and_objects'):
                base_nodes, objects = node_parser.get_nodes_and_objects(nodes)
            else:
                base_nodes, objects = list(nodes), []
        elif nodes:
            # Accept prebuilt nodes (e.g., TEI→custom TextNodes)
            base_nodes, objects = list(nodes), []
        else:
            raise RuntimeError("Markdown node parser produced no nodes; aborting index creation.")
        # enrich metadata on derived nodes
        self._enrich_nodes_metadata(base_nodes)
        self._enrich_nodes_metadata(objects)
        # assign sections/top_sections by inspecting original markdown docs (skip for GROBID)
        try:
            use_grobid = str(os.getenv('USE_GROBID') or '0').strip().lower() in ('1','true','yes','y','on')
            if not use_grobid and documents:
                self._assign_sections_from_docs(documents, base_nodes)
                self._assign_sections_from_docs(documents, objects)
                # apply annotated [[SEC ...]] mapping using normalized tree anchors
                persist_dir = None
                try:
                    # infer persist dir from source_file of first document
                    src = None
                    for d in documents:
                        src = (d.metadata or {}).get('source_file')
                        if src:
                            break
                    if src:
                        base = os.path.splitext(src)[0]
                        persist_dir = os.path.join(self.storage_path, f"{base}_vector_index")
                except Exception:
                    persist_dir = None
                if persist_dir and os.path.isdir(persist_dir):
                    # Prefer normalized flat mapping to set a single best section
                    self._apply_normalized_flat_sections(persist_dir, base_nodes)
                    self._apply_normalized_flat_sections(persist_dir, objects)
        except Exception:
            pass
        try:
            total_nodes = len(base_nodes) + len(objects)
            print(f"[index] building VectorStoreIndex with nodes={total_nodes}")
        except Exception:
            pass
        vector_index = VectorStoreIndex(base_nodes + objects)
        return vector_index

    def create_vector_query_engine(self, vector_index):
        retriever = VectorIndexRetriever(
            index=vector_index,
            similarity_top_k=5,
        )
        response_synthesizer = get_response_synthesizer()
        postprocessors: List = []
        if self.enable_section_reasoner:
            postprocessors.append(self.SectionReasonerPostprocessor())

        desired_mode = self.response_mode or 'tree_summarize'
        if self.cohere_rerank:
            try:
                from llama_index.postprocessor.cohere_rerank import CohereRerank
                os.environ["COHERE_API_KEY"] = self.cohere_api_key
                cohere_api_key = os.environ["COHERE_API_KEY"]
                cohere_rerank = CohereRerank(api_key=cohere_api_key, top_n=5)

                query_engine = RetrieverQueryEngine.from_args(
                    retriever=retriever,
                    response_mode=desired_mode,
                    response_synthesizer=response_synthesizer,
                    node_postprocessors=([cohere_rerank] + postprocessors) if postprocessors else [cohere_rerank],
                )
            except Exception:
                # Fallback without rerank if package/import is unavailable
                query_engine = RetrieverQueryEngine.from_args(
                    retriever=retriever,
                    response_mode=desired_mode,
                    response_synthesizer=response_synthesizer,
                    node_postprocessors=postprocessors if postprocessors else None,
                )
        else:
            query_engine = RetrieverQueryEngine.from_args(
                retriever=retriever,
                response_mode=desired_mode,
                response_synthesizer=response_synthesizer,
                node_postprocessors=postprocessors if postprocessors else None,
            )
        return query_engine

    def get_query_engine(self, file):
        vector_index_persist_path = f'{self.storage_path}/{file}_vector_index/'

        def _meta_path(p: str) -> str:
            return os.path.join(p, 'index_meta.json')

        def _write_meta(p: str) -> None:
            try:
                meta = {
                    'api': (os.getenv('API') or '').strip().lower(),
                    'embedding_api': (os.getenv('EMBEDDING_API') or '').strip().lower(),
                    'embedding_model_name': self.embedding_model_name,
                }
                with open(_meta_path(p), 'w', encoding='utf-8') as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

        def _read_meta(p: str) -> dict:
            try:
                with open(_meta_path(p), 'r', encoding='utf-8') as f:
                    return json.load(f) or {}
            except Exception:
                return {}

        def _has_index_files(p: str) -> bool:
            return os.path.exists(os.path.join(p, 'docstore.json')) and os.path.exists(os.path.join(p, 'index_store.json'))

        if os.path.exists(vector_index_persist_path) and _has_index_files(vector_index_persist_path):
            # Validate metadata; rebuild if provider/embedding changed
            try:
                cur = {
                    'api': (os.getenv('API') or '').strip().lower(),
                    'embedding_api': (os.getenv('EMBEDDING_API') or '').strip().lower(),
                    'embedding_model_name': self.embedding_model_name,
                }
                prev = _read_meta(vector_index_persist_path)
                if not prev or any((prev.get(k) or '') != (cur.get(k) or '') for k in cur.keys()):
                    # Invalidate and rebuild
                    import shutil
                    shutil.rmtree(vector_index_persist_path, ignore_errors=True)
                    os.makedirs(vector_index_persist_path, exist_ok=True)
                    raise FileNotFoundError('force rebuild due to provider/embedding change')
            except Exception:
                pass
            try:
                print(f"[index] {file}: loading existing index from {vector_index_persist_path}")
            except Exception:
                pass
            storage_context = StorageContext.from_defaults(persist_dir=vector_index_persist_path)
            vector_index = load_index_from_storage(storage_context)
        else:
            try:
                print(f"[index] {file}: no persisted index found → parsing and building")
            except Exception:
                pass
            pdf_path = os.path.join(self.input_path, f"{file}.pdf")
            documents, node_parser, nodes = self.parse_pdf_to_nodes(pdf_path)
            # First, write raw markdown/outline (for inspection)
            self._write_raw_markdown_and_outline(vector_index_persist_path, documents)
            # Create the index; metadata assignment will use '#' headings only
            vector_index = self.create_vector_index(documents, node_parser, nodes)
            vector_index.storage_context.persist(vector_index_persist_path)
            _write_meta(vector_index_persist_path)
            try:
                print(f"[index] {file}: persisted index → {vector_index_persist_path}")
            except Exception:
                pass

        query_engine = self.create_vector_query_engine(vector_index)
        try:
            print(f"[engine] {file}: query engine ready")
        except Exception:
            pass
        return query_engine

## Grobid engine moved to utils/VectorQueryEngineCreatorGrobid.py
from utils.VectorQueryEngineCreatorGrobid import VectorQueryEngineCreatorGrobid  # re-export
