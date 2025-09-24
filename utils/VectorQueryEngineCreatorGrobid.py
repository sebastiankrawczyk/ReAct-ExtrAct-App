import os
import re
from typing import List, Dict, Any

from utils.VectorQueryEngineCreator import VectorQueryEngineCreator

try:
    from llama_index.core import Document, VectorStoreIndex
    from llama_index.core.schema import TextNode
    from llama_index.core.node_parser import MarkdownNodeParser as PureMarkdownNodeParser  # type: ignore
    # JSONNodeParser not required since we'll build TextNodes directly from JSON
    from llama_index.core.node_parser import SentenceSplitter  # type: ignore
    from llama_index.core import StorageContext, load_index_from_storage
except Exception:
    Document = None  # type: ignore
    VectorStoreIndex = None  # type: ignore
    TextNode = None  # type: ignore
    PureMarkdownNodeParser = None  # type: ignore
    SentenceSplitter = None  # type: ignore
    StorageContext = None  # type: ignore
    load_index_from_storage = None  # type: ignore
    

class VectorQueryEngineCreatorGrobid(VectorQueryEngineCreator):
    # ---------- Fresh TEI section extractor (namespace-aware, minimal) ----------
    def _extract_sections_from_tei(self, tei_xml: str) -> List[Dict[str, str]]:
        try:
            import xml.etree.ElementTree as ET
        except Exception:
            return []
        try:
            root = ET.fromstring((tei_xml or '').encode('utf-8'))
        except Exception:
            return []
        NS = '{http://www.tei-c.org/ns/1.0}'
        sections: List[Dict[str, str]] = []
        def _get_text(el) -> str:
            try:
                return ''.join(el.itertext())
            except Exception:
                return ''
        # Extract figures/tables to attach later
        figures: List[Dict[str, str]] = []
        for fig in root.findall(f'.//{NS}figure'):
            cap = fig.find(f'.//{NS}figDesc')
            cap_txt = (_get_text(cap) or '').strip()
            if cap_txt:
                figures.append({'title': 'Figure', 'content': cap_txt, 'section_type': 'figure'})
        tables: List[Dict[str, str]] = []
        for tb in root.findall(f'.//{NS}table'):
            cells: List[str] = []
            for row in tb.findall(f'.//{NS}row'):
                row_text = ' | '.join([(_get_text(c) or '').strip() for c in row.findall(f'.//{NS}cell')])
                if row_text.strip():
                    cells.append(row_text)
            tbl_txt = '\n'.join(cells).strip()
            if tbl_txt:
                tables.append({'title': 'Table', 'content': tbl_txt, 'section_type': 'table'})
        def _collect(div_el):
            title_el = div_el.find(f'{NS}head')
            title = _get_text(title_el).strip() if title_el is not None else ''
            content_parts: List[str] = []
            for p in div_el.findall(f'.//{NS}p'):
                txt = _get_text(p).strip()
                if txt:
                    content_parts.append(txt)
            content = '\n\n'.join(content_parts)
            if title or content:
                sections.append({'title': title, 'content': content, 'section_type': 'body'})
            for child in list(div_el):
                if isinstance(child.tag, str) and child.tag == f'{NS}div':
                    _collect(child)
        body = root.find(f'.//{NS}text/{NS}body')
        if body is not None:
            for div in body.findall(f'{NS}div'):
                _collect(div)
        sections.extend(figures)
        sections.extend(tables)
        return sections

    def _create_hierarchical_nodes(self, sections: List[Dict[str, Any]], chunk_size: int = 512, chunk_overlap: int = 50, basename: str = '') -> List[TextNode]:
        nodes: List[TextNode] = []
        if SentenceSplitter is None or TextNode is None:
            return nodes
        # Helper: classify section type for metadata tuning
        def _classify(title: str) -> str:
            t = (title or '').lower()
            if any(k in t for k in ['introduction', 'background']):
                return 'introduction'
            if any(k in t for k in ['method', 'materials', 'procedure']):
                return 'methods'
            if any(k in t for k in ['result', 'findings']):
                return 'results'
            if any(k in t for k in ['discussion', 'limitations']):
                return 'discussion'
            if any(k in t for k in ['conclusion', 'summary']):
                return 'conclusion'
            if any(k in t for k in ['reference', 'bibliograph']):
                return 'references'
            if any(k in t for k in ['table']):
                return 'table'
            if any(k in t for k in ['figure']):
                return 'figure'
            return 'body'
        # Hygiene: normalize text
        def _clean_text(s: str) -> str:
            s = (s or '').replace('\u2013', '-').replace('\u2014', '-').replace('\u2019', "'")
            s = s.replace('\u2018', "'").replace('\u201c', '"').replace('\u201d', '"')
            s = re.sub(r"\s+", " ", s)
            return s.strip()
        for s in sections or []:
            title = (s.get('title') or '').strip()
            body = (s.get('content') or '').strip()
            if not (title or body):
                continue
            sec_type = s.get('section_type') or _classify(title)
            # Merge very short paragraphs together before splitting
            paras = [p.strip() for p in (body.split('\n\n') if body else []) if p.strip()]
            merged: List[str] = []
            buf = ''
            for p in paras:
                if len(p) < 120:
                    buf = (buf + ' ' + p).strip()
                else:
                    if buf:
                        merged.append(buf)
                        buf = ''
                    merged.append(p)
            if buf:
                merged.append(buf)
            prepped = '\n\n'.join(merged) if merged else body
            # Adaptive chunk size
            adaptive = chunk_size
            if sec_type in ('results', 'discussion'):
                adaptive = 900
            elif sec_type in ('introduction', 'methods'):
                adaptive = 700
            elif sec_type in ('conclusion', 'abstract'):
                adaptive = 400
            splitter = SentenceSplitter(chunk_size=adaptive, chunk_overlap=chunk_overlap)
            sec_text = (title + '\n\n' + _clean_text(prepped)).strip() if title else _clean_text(prepped)
            chunks = splitter.split_text(sec_text)
            for idx, ch in enumerate(chunks):
                ch = (ch or '').strip()
                if not ch:
                    continue
                meta = {
                    'source_file': basename,
                    'section_title': title,
                    'section': title,
                    'chunk_index': idx,
                    'parent_section': title,
                    'section_type': sec_type,
                }
                try:
                    nodes.append(TextNode(text=ch, metadata=meta))
                except Exception:
                    continue
        return nodes

    def _tei_to_json(self, tei_xml: str) -> Dict[str, Any]:
        try:
            from lxml import etree  # type: ignore
        except Exception:
            return {}
        try:
            root = etree.fromstring((tei_xml or '').encode('utf-8'))
        except Exception:
            return {}
        ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
        out: Dict[str, Any] = {
            'title': '',
            'authors': [],
            'abstract': '',
            'sections': [],
            'references': [],
            'paragraphs': [],
        }
        def _text(el) -> str:
            try:
                return ' '.join((''.join(el.itertext())).split())
            except Exception:
                return ''
        # Title
        t_el = root.find('.//tei:teiHeader//tei:titleStmt/tei:title', namespaces=ns)
        out['title'] = _text(t_el) if t_el is not None else ''
        # Authors (simple)
        for a in root.findall('.//tei:teiHeader//tei:titleStmt/tei:author', namespaces=ns):
            nm = _text(a)
            if nm:
                out['authors'].append({'name': nm})
        # Abstract
        abs_el = root.find('.//tei:profileDesc/tei:abstract', namespaces=ns)
        out['abstract'] = _text(abs_el) if abs_el is not None else ''
        # Structured sections (best-effort)
        body = root.find('.//tei:text/tei:body', namespaces=ns)
        try:
            print(f"[grobid-json:body] exists={'yes' if body is not None else 'no'}")
            div_count = len(root.findall('.//tei:text/tei:body//tei:div', namespaces=ns))
            p_count_ns = len(root.findall('.//tei:text/tei:body//tei:p', namespaces=ns))
            p_nodes_local = root.xpath('.//tei:text/tei:body//*[local-name()="p"]', namespaces=ns)
            p_count_local = len(p_nodes_local)
            print(f"[grobid-json:scan] divs={div_count} p(ns)={p_count_ns} p(local)={p_count_local}")
            if p_count_local > 0:
                try:
                    sample_p = ' '.join((''.join(p_nodes_local[0].itertext())).split())
                    print(f"[grobid-json:sample-p] len={len(sample_p)} text='{sample_p[:120]}'")
                except Exception:
                    pass
        except Exception:
            pass
        def _collect_sections(el, depth: int = 1) -> List[Dict[str, Any]]:
            sections: List[Dict[str, Any]] = []
            if el is None:
                return sections
            for div in el.findall('./tei:div', namespaces=ns):
                head = div.find('./tei:head', namespaces=ns)
                title = _text(head) if head is not None else ''
                paras = []
                for p in div.findall('./tei:p', namespaces=ns):
                    txt = _text(p)
                    if txt:
                        paras.append(txt)
                child_sections = _collect_sections(div, depth + 1)
                one = {'title': title, 'paragraphs': paras}
                if child_sections:
                    one['sections'] = child_sections
                sections.append(one)
            return sections
        out['sections'] = _collect_sections(body, 1)
        # Flat paragraphs (robust extraction regardless of div structure)
        try:
            # Build page mapping via preceding::pb and facsimile surfaces
            surf_map = {}
            for surf in root.findall('.//tei:facsimile/tei:surface', namespaces=ns):
                sid = surf.get('{http://www.w3.org/XML/1998/namespace}id') or surf.get('id')
                n = surf.get('n')
                if sid and n and str(n).isdigit():
                    surf_map['#' + sid] = int(n)
            # Prefer namespace search; include p, ab, s
            p_nodes = root.findall('.//tei:text//tei:p', namespaces=ns)
            ab_nodes = root.findall('.//tei:text//tei:ab', namespaces=ns)
            s_nodes = root.findall('.//tei:text//tei:s', namespaces=ns)
            if not (p_nodes or ab_nodes or s_nodes):
                # local-name() fallback
                p_nodes = root.xpath('.//tei:text//*[local-name()="p" or local-name()="ab" or local-name()="s"]', namespaces=ns)
                ab_nodes = []
                s_nodes = []
            cand_nodes = list(p_nodes) + list(ab_nodes) + list(s_nodes)
            print(f"[grobid-json:flat-scan] p={len(p_nodes)} ab={len(ab_nodes)} s={len(s_nodes)} total={len(cand_nodes)}")
            for p in cand_nodes:
                txt = _text(p)
                if not txt:
                    continue
                # Section titles from ancestor divs
                section_title = ''
                top_title = ''
                anc = p.getparent()
                closest_head = None
                top_head = None
                # walk up
                while anc is not None:
                    try:
                        from lxml import etree as _et  # type: ignore
                        lname = _et.QName(anc.tag).localname.lower()
                    except Exception:
                        lname = ''
                    if lname == 'div' and closest_head is None:
                        h = anc.find('./tei:head', namespaces=ns)
                        if h is not None:
                            closest_head = _text(h)
                    if lname == 'div':
                        h2 = anc.find('./tei:head', namespaces=ns)
                        if h2 is not None:
                            top_head = _text(h2)
                    anc = anc.getparent()
                section_title = closest_head or ''
                top_title = top_head or section_title
                # Page from nearest preceding pb or @facs
                page = None
                pb = p.xpath('preceding::tei:pb[1]', namespaces=ns)
                if pb:
                    n = pb[0].get('n')
                    if n and str(n).isdigit():
                        page = int(n)
                if page is None:
                    facs = p.get('facs')
                    if facs and facs in surf_map:
                        page = surf_map[facs]
                out['paragraphs'].append({'text': txt, 'section': section_title, 'top_section': top_title, 'page': page})
            print(f"[grobid-json:flat] paragraphs_extracted={len(out['paragraphs'])}")
        except Exception as e:
            try:
                print(f"[grobid-json:flat][error] {e}")
            except Exception:
                pass
        # Fallback: if still no paragraphs, split full text into paragraphs
        if not out['paragraphs']:
            try:
                full_text = ' '.join(''.join(root.itertext()).split())
                # Simple paragraph segmentation by sentences length threshold
                chunks: List[str] = []
                buf: List[str] = []
                for tok in full_text.split('. '):
                    buf.append(tok)
                    if sum(len(x) for x in buf) > 400:
                        chunks.append('. '.join(buf).strip())
                        buf = []
                if buf:
                    chunks.append('. '.join(buf).strip())
                for ch in chunks:
                    if len(ch) < 60:
                        continue
                    out['paragraphs'].append({'text': ch, 'section': '', 'top_section': '', 'page': None})
                print(f"[grobid-json:fallback] paragraphs_from_fulltext={len(out['paragraphs'])}")
            except Exception as e:
                try:
                    print(f"[grobid-json:fallback][error] {e}")
                except Exception:
                    pass
        # References (simple)
        for bibl in root.findall('.//tei:listBibl/tei:biblStruct', namespaces=ns):
            cit = _text(bibl)
            if cit:
                out['references'].append({'citation': cit})
        return out

    def _json_to_textnodes(self, data: Dict[str, Any], basename: str) -> List[TextNode]:
        nodes: List[TextNode] = []
        # Paragraph list takes precedence if present
        paras = data.get('paragraphs') or []
        for pr in paras:
            try:
                txt = (pr.get('text') or '').strip()
                if not txt:
                    continue
                md = {'source_file': basename}
                if pr.get('section'):
                    md['section'] = pr.get('section')
                if pr.get('top_section'):
                    md['top_section'] = pr.get('top_section')
                if isinstance(pr.get('page'), int):
                    md['page_label'] = int(pr.get('page'))
                nodes.append(TextNode(text=txt, metadata=md))
            except Exception:
                continue
        def _walk(secs: List[Dict[str, Any]], top_title: str = ''):
            for s in secs or []:
                title = (s.get('title') or '').strip()
                top = top_title or title
                for para in (s.get('paragraphs') or []):
                    txt = (para or '').strip()
                    if not txt:
                        continue
                    meta = {'source_file': basename}
                    if title:
                        meta['section'] = title
                    if top:
                        meta['top_section'] = top
                    try:
                        nodes.append(TextNode(text=txt, metadata=meta))
                    except Exception:
                        pass
                _walk(s.get('sections') or [], top)
        # Abstract first
        abs_txt = (data.get('abstract') or '').strip()
        if abs_txt:
            try:
                nodes.append(TextNode(text=abs_txt, metadata={'source_file': basename, 'section': 'Abstract', 'top_section': 'Abstract'}))
            except Exception:
                pass
        _walk(data.get('sections') or [], '')
        return nodes

    def parse_pdf_to_nodes(self, path_to_pdf):
        documents = None
        node_parser = None
        nodes = None
        basename = os.path.basename(path_to_pdf)
        try:
            from utils.grobid_client import grobid_fulltext_tei  # type: ignore
        except Exception:
            grobid_fulltext_tei = None
        strict = str(os.getenv('STRICT_GROBID') or '1').strip().lower() in ('1','true','yes','y','on')
        try:
            print(f"[parse] {basename}: using GROBID service")
            if grobid_fulltext_tei is None:
                raise RuntimeError("GROBID utilities unavailable")
            tei_xml = grobid_fulltext_tei(path_to_pdf, base_url=os.getenv('GROBID_URL') or None)
            print(f"[grobid-tei] length={len(tei_xml or '')}")
            # Extract sections, then chunk
            sections = self._extract_sections_from_tei(tei_xml)
            print(f"[grobid-sections] count={len(sections)} first_titles={[s.get('title','') for s in sections[:3]]}")
            nodes = self._create_hierarchical_nodes(sections, chunk_size=512, chunk_overlap=50, basename=basename)
            node_parser = None
            # Minimal document: concat sample for raw write
            all_text = '\n\n'.join([n.text for n in nodes[:200]]) if nodes else ''
            documents = [Document(text=all_text, metadata={'source_file': basename})] if all_text else []
            sample_secs = []
            sample_lens = []
            for n in (nodes or [])[:5]:
                sample_lens.append(len(n.text or ''))
                md = getattr(n, 'metadata', {}) or {}
                sample_secs.append((md.get('section_title') or md.get('parent_section') or '')[:40])
            print(f"[grobid-nodes] built={len(nodes) if nodes else 0} sample_len={sample_lens} sample_sections={sample_secs}")
            print(f"[parse] {basename}: parsed -> nodes={len(nodes) if nodes else 0}")
        except Exception:
            if strict:
                raise
            documents, node_parser, nodes = [], None, []
        return documents, node_parser, nodes

    def _build_nodes_from_tei(self, tei_xml: str, basename: str):
        try:
            from lxml import etree  # type: ignore
        except Exception:
            return None, None
        try:
            root = etree.fromstring((tei_xml or '').encode('utf-8'))
        except Exception:
            return None, None
        ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
        body = root.find('.//tei:text/tei:body', namespaces=ns)
        if body is None:
            return None, None
        facs_map = {}
        try:
            for surf in root.findall('.//tei:facsimile/tei:surface', namespaces=ns):
                sid = surf.get('{http://www.w3.org/XML/1998/namespace}id') or surf.get('id')
                n = surf.get('n')
                if sid and n and str(n).isdigit():
                    facs_map['#' + sid] = int(n)
        except Exception:
            facs_map = {}
        section_counters: list[int] = []
        current_section_title: str = ''
        top_section_title: str = ''
        current_sec_id: str = ''
        current_page: int | None = None
        page_lines: dict[int, list[str]] = {}
        built_nodes = []

        def _text(el) -> str:
            try:
                return ''.join(el.itertext()).strip()
            except Exception:
                return ''

        def _ensure_page(pg: int | None):
            if pg is None:
                return
            if pg not in page_lines:
                page_lines[pg] = []

        for el in body.iter():
            tag = etree.QName(el).localname.lower()
            if tag == 'pb':
                n = el.get('n')
                try:
                    current_page = int(n) if n and str(n).isdigit() else current_page
                except Exception:
                    pass
                _ensure_page(current_page)
                continue
            facs = el.get('facs')
            if facs and facs in facs_map:
                try:
                    current_page = facs_map[facs]
                except Exception:
                    pass
                _ensure_page(current_page)
            if tag == 'div':
                head = el.find('./tei:head', namespaces=ns)
                if head is not None:
                    depth = 1
                    parent = el.getparent()
                    while parent is not None and etree.QName(parent).localname.lower() == 'div':
                        depth += 1
                        parent = parent.getparent()
                    if len(section_counters) < depth:
                        section_counters.extend([0] * (depth - len(section_counters)))
                    section_counters[depth - 1] = section_counters[depth - 1] + 1
                    for i in range(depth, len(section_counters)):
                        section_counters[i] = 0
                    non_zero = [str(x) for x in section_counters if x > 0]
                    current_sec_id = '.'.join(non_zero) if non_zero else ''
                    current_section_title = _text(head)
                    if depth == 1 and current_section_title:
                        top_section_title = current_section_title
                continue
            if tag in ('p', 'item'):
                txt = _text(el)
                if not txt:
                    continue
                meta = {
                    'source_file': basename,
                    'section': current_section_title or '',
                    'top_section': top_section_title or '',
                }
                if current_sec_id:
                    meta['section_id'] = current_sec_id
                if isinstance(current_page, int):
                    meta['page_label'] = int(current_page)
                if TextNode is not None:
                    try:
                        node = TextNode(text=txt, metadata=meta)
                        built_nodes.append(node)
                    except Exception:
                        pass
                if isinstance(current_page, int):
                    _ensure_page(current_page)
                    page_lines[current_page].append(txt)
                else:
                    _ensure_page(1)
                    page_lines[1].append(txt)
                continue
            if tag in ('figdesc',):
                txt = _text(el)
                if not txt:
                    continue
                cap = f"Figure: {txt}"
                meta = {
                    'source_file': basename,
                    'section': current_section_title or '',
                    'top_section': top_section_title or '',
                }
                if current_sec_id:
                    meta['section_id'] = current_sec_id
                if isinstance(current_page, int):
                    meta['page_label'] = int(current_page)
                if TextNode is not None:
                    try:
                        node = TextNode(text=cap, metadata=meta)
                        built_nodes.append(node)
                    except Exception:
                        pass
                if isinstance(current_page, int):
                    _ensure_page(current_page)
                    page_lines[current_page].append(cap)
                else:
                    _ensure_page(1)
                    page_lines[1].append(cap)
                continue
        documents = []
        for pg in sorted(page_lines.keys()):
            txt = '\n\n'.join([ln for ln in page_lines.get(pg, []) if (ln or '').strip()])
            if not txt:
                continue
            try:
                doc = Document(text=txt, metadata={'page_label': pg, 'source_file': basename})
                documents.append(doc)
            except Exception:
                continue
        return documents, built_nodes

    def _write_raw_markdown_and_outline(self, persist_dir: str, documents: List[Document]):
        try:
            os.makedirs(persist_dir, exist_ok=True)
            raw_path = os.path.join(persist_dir, 'raw_markdown.md')
            with open(raw_path, 'w', encoding='utf-8') as f:
                for d in documents:
                    txt = d.text or ''
                    try:
                        txt = re.sub(r"\s*\[\[SEC\s+id=[^\]]+\]\]\s*", "", txt)
                    except Exception:
                        pass
                    f.write(txt)
                    f.write("\n")
            outline_path = os.path.join(persist_dir, 'section_tree.md')
            header_rx = re.compile(r'^(#{1,6})\s+(.+?)\s*$')
            with open(outline_path, 'w', encoding='utf-8') as f:
                f.write("# Section Tree\n\n")
                for d in documents:
                    for line in d.text.splitlines():
                        s = re.sub(r"\s*\[\[SEC\s+id=[^\]]+\]\]\s*", "", line or '').strip()
                        m = header_rx.match(s)
                        if not m:
                            continue
                        level = len(m.group(1))
                        title = m.group(2).strip()
                        indent = '  ' * (level - 1)
                        f.write(f"{indent}{'#'*level} {title}\n")
            try:
                print(f"[md] wrote raw markdown and outline → {persist_dir}")
            except Exception:
                pass
        except Exception:
            pass

    def create_vector_index(self, documents, node_parser, nodes):
        # Delegate to base implementation to apply markdown-based assignment
        return super().create_vector_index(documents, node_parser, nodes)

    # ---------------- Post-persist TEI enrichment ----------------
    def _normalize(self, s: str) -> str:
        import re as _re
        s = (s or '').lower()
        s = _re.sub(r"\s+", " ", s)
        s = _re.sub(r"[^a-z0-9\s]", "", s)
        return s.strip()

    def _build_tei_paragraph_entries(self, tei_xml: str) -> List[Dict[str, Any]]:
        try:
            from lxml import etree  # type: ignore
        except Exception:
            return []
        try:
            root = etree.fromstring((tei_xml or '').encode('utf-8'))
        except Exception:
            return []
        ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
        # Build facsimile id → page number map
        facs_map: Dict[str, int] = {}
        try:
            for surf in root.findall('.//tei:facsimile/tei:surface', namespaces=ns):
                sid = surf.get('{http://www.w3.org/XML/1998/namespace}id') or surf.get('id')
                n = surf.get('n')
                if sid and n and str(n).isdigit():
                    facs_map['#' + sid] = int(n)
        except Exception:
            facs_map = {}
        entries: List[Dict[str, Any]] = []
        def _text(el) -> str:
            try:
                return ' '.join(''.join(el.itertext()).split())
            except Exception:
                return ''
        p_nodes = root.findall('.//tei:text/tei:body//tei:p', namespaces=ns)
        total = len(p_nodes)
        got_page = 0
        for p in p_nodes:
            # Section labels
            section = ''
            top = ''
            anc = p.getparent()
            nearest = None
            top_h = None
            from lxml import etree as _et
            while anc is not None:
                try:
                    lname = _et.QName(anc.tag).localname.lower()
                except Exception:
                    lname = ''
                if lname == 'div' and nearest is None:
                    h = anc.find('./tei:head', namespaces=ns)
                    if h is not None:
                        nearest = _text(h)
                if lname == 'div':
                    h2 = anc.find('./tei:head', namespaces=ns)
                    if h2 is not None:
                        top_h = _text(h2)
                anc = anc.getparent()
            section = nearest or ''
            top = top_h or section
            # Page via preceding pb
            page = None
            try:
                pb = p.xpath('preceding::tei:pb[1]', namespaces=ns)
                if pb:
                    n = pb[0].get('n')
                    if n and str(n).isdigit():
                        page = int(n)
                # Also consider @facs on p or ancestors
                if page is None:
                    facs = p.get('facs')
                    if facs and facs in facs_map:
                        page = facs_map[facs]
                if page is None:
                    anc2 = p.getparent()
                    while anc2 is not None and page is None:
                        try:
                            facs_a = anc2.get('facs')
                        except Exception:
                            facs_a = None
                        if facs_a and facs_a in facs_map:
                            page = facs_map[facs_a]
                            break
                        anc2 = anc2.getparent()
            except Exception:
                pass
            txt = _text(p)
            if not txt:
                continue
            entries.append({'norm': self._normalize(txt), 'page': page, 'section': section, 'top': top})
            if isinstance(page, int):
                got_page += 1
        try:
            print(f"[enrich] tei_paragraphs={total} with_page={got_page}")
        except Exception:
            pass
        # Fallback: if almost no pages found but we have surfaces or many paragraphs, distribute pages by order
        if total > 0 and got_page <= 1:
            # derive total_pages from surfaces or heuristic (e.g., 10 pages for long docs)
            total_pages = 0
            try:
                total_pages = max(total_pages, len({v for v in facs_map.values()}))
            except Exception:
                total_pages = 0
            if total_pages <= 1:
                try:
                    # heuristic: 1 page per ~40 paragraphs
                    total_pages = max(1, min(50, total // 40))
                except Exception:
                    total_pages = 1
            if total_pages > 1:
                for idx, e in enumerate(entries):
                    # spread uniformly
                    pg = 1 + (idx * total_pages) // max(1, total)
                    e['page'] = pg
                print(f"[enrich] fallback page distribution applied: total_pages≈{total_pages}")
        return entries

    def _enrich_from_tei(self, persist_dir: str, basename: str) -> None:
        try:
            # Find TEI file
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            tei_dir = os.path.join(repo_root, 'storage', 'grobid_tei_raw')
            tei_path = os.path.join(tei_dir, f"{basename}.tei.xml")
            if not os.path.isfile(tei_path):
                print(f"[enrich] TEI not found: {tei_path}")
                return
            with open(tei_path, 'r', encoding='utf-8') as f:
                tei_xml = f.read()
            entries = self._build_tei_paragraph_entries(tei_xml)
            if not entries:
                print("[enrich] No TEI entries; skip")
                return
            try:
                from rapidfuzz.fuzz import token_set_ratio  # type: ignore
            except Exception:
                token_set_ratio = None
            # Load index/docstore
            if StorageContext is None or load_index_from_storage is None:
                return
            sc = StorageContext.from_defaults(persist_dir=persist_dir)
            index = load_index_from_storage(sc)
            docstore = sc.docstore
            updated = 0
            for doc_id, node in list(getattr(docstore, 'docs', {}).items()):
                try:
                    text = getattr(node, 'get_content', lambda: '')()
                except Exception:
                    try:
                        text = getattr(node, 'text', '')
                    except Exception:
                        text = ''
                text_norm = self._normalize(text)
                if not text_norm:
                    continue
                # Exact window match
                matched = None
                if len(text_norm) >= 40:
                    for e in entries:
                        if e['norm'] and e['norm'] in text_norm:
                            matched = e
                            break
                # Fuzzy match
                if matched is None and token_set_ratio is not None:
                    best = (-1, None)
                    for e in entries:
                        if not e['norm']:
                            continue
                        score = token_set_ratio(text_norm, e['norm'])
                        if score > best[0]:
                            best = (score, e)
                    if best[0] >= 85:
                        matched = best[1]
                if matched is None:
                    continue
                md = getattr(node, 'metadata', None) or {}
                if matched.get('page') is not None:
                    md['page_label'] = int(matched['page'])
                if matched.get('section'):
                    md['section'] = matched['section']
                    md['section_title'] = matched['section']
                if matched.get('top'):
                    md['top_section'] = matched['top']
                try:
                    setattr(node, 'metadata', md)
                    docstore.set_document(node)  # type: ignore
                    updated += 1
                except Exception:
                    continue
            # Persist changes
            try:
                sc.persist(persist_dir)
            except Exception:
                pass
            print(f"[enrich] updated_nodes={updated}")
        except Exception as e:
            try:
                print(f"[enrich][error] {e}")
            except Exception:
                pass

    # Override get_query_engine to run enrichment after persist
    def get_query_engine(self, file):
        # Delegate to base get_query_engine to keep behavior consistent
        return super().get_query_engine(file)
