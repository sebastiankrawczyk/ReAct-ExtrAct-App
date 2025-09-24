import os
import time
from typing import Optional

import requests


def grobid_fulltext_tei(pdf_path: str, base_url: Optional[str] = None, timeout: int = 180, retries: int = 2, backoff: float = 1.5) -> str:
    """
    Call GROBID /api/processFulltextDocument and return TEI XML as text.
    Immediately saves the raw TEI to storage/grobid_tei_raw/<pdf_stem>.tei.xml (or GROBID_TEI_SAVE_DIR).
    """
    url = (base_url or os.getenv('GROBID_URL') or 'http://localhost:8070').rstrip('/') + '/api/processFulltextDocument'
    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            with open(pdf_path, 'rb') as f:
                files = {'input': (os.path.basename(pdf_path), f, 'application/pdf')}
                data = {
                    'consolidateHeader': '1',
                    'consolidateCitations': '0',
                    'includeRawCitations': '0',
                    # Request granular coordinates only (no legacy flags)
                    'teiCoordinates': 'p,head,figure,table,ref,item,row',
                    'segmentSentences': '1',
                }
                resp = requests.post(url, files=files, data=data, timeout=timeout)
                resp.raise_for_status()
                tei_text = resp.text
                # Persist TEI immediately and unconditionally to a stable location for audit
                try:
                    save_dir = os.getenv('GROBID_TEI_SAVE_DIR')
                    if not save_dir:
                        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
                        save_dir = os.path.join(repo_root, 'storage', 'grobid_tei_raw')
                    os.makedirs(save_dir, exist_ok=True)
                    stem = os.path.splitext(os.path.basename(pdf_path))[0]
                    out_path = os.path.join(save_dir, f"{stem}.tei.xml")
                    with open(out_path, 'w', encoding='utf-8') as outf:
                        outf.write(tei_text or '')
                    # Quick debug: pb/surface presence
                    try:
                        pb_cnt = tei_text.count('<pb ')
                        surf_cnt = tei_text.count('<surface ')
                        print(f"[grobid-tei-debug] pb={pb_cnt} surface={surf_cnt}")
                    except Exception:
                        pass
                except Exception:
                    pass
                return tei_text
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(backoff ** attempt)
            else:
                raise
    raise RuntimeError(f"GROBID failed: {last_err}")


def tei_to_markdown(tei_xml: str) -> str:
    """
    TEI → Markdown conversion with inline page markers and ordered traversal.
    - Emits headings from nested tei:div/tei:head with appropriate levels
    - Preserves content order
    - Inserts <!--PAGE:n--> when encountering tei:pb in flow
    - Handles paragraphs, lists, simple tables, and figure captions minimally
    Returns sanitized Markdown.
    """
    try:
        from lxml import etree  # type: ignore
    except Exception:
        # Fallback: sanitize tag-stripped text
        return sanitize_markdown(extract_plain_text_from_tei(tei_xml))

    try:
        root = etree.fromstring((tei_xml or '').encode('utf-8'))
    except Exception:
        return sanitize_markdown(extract_plain_text_from_tei(tei_xml))

    ns = {'tei': 'http://www.tei-c.org/ns/1.0'}

    def _text(el) -> str:
        return ''.join(el.itertext()).strip() if el is not None else ''

    lines = []
    # Title
    title_el = root.find('.//tei:teiHeader//tei:titleStmt/tei:title', namespaces=ns)
    title_txt = _text(title_el)
    if title_txt:
        lines.append(f"# {title_txt}")

    body = root.find('.//tei:text/tei:body', namespaces=ns)
    if body is None:
        return sanitize_markdown(('\n'.join(lines) + '\n') if lines else extract_plain_text_from_tei(tei_xml))

    # Maintain hierarchical section ids via path counters
    path_stack = []  # list of ints per depth

    def render_node(el, div_depth: int = 0):
        tag = etree.QName(el).localname.lower()
        # Page break
        if tag == 'pb':
            n = el.get('n') or el.get('{http://www.w3.org/XML/1998/namespace}id') or ''
            try:
                num = int(str(n)) if str(n).isdigit() else None
            except Exception:
                num = None
            if num is not None:
                lines.append(f"<!--PAGE:{num}-->")
            return
        # Division / heading
        if tag == 'div':
            head = el.find('./tei:head', namespaces=ns)
            htxt = _text(head)
            if htxt:
                level = min(6, max(2, 1 + div_depth))
                if len(path_stack) < level:
                    path_stack.extend([0] * (level - len(path_stack)))
                path_stack[level-1] = (path_stack[level-1] if level-1 < len(path_stack) else 0) + 1
                for i in range(level, len(path_stack)):
                    path_stack[i] = 0
                non_zero = [str(x) for x in path_stack if x > 0]
                sec_id = '.'.join(non_zero) if non_zero else '1'
                page = ''
                for prev in reversed(lines):
                    if isinstance(prev, str) and prev.startswith('<!--PAGE:') and prev.endswith('-->'):
                        try:
                            page = prev[len('<!--PAGE:'):-3]
                        except Exception:
                            page = ''
                        break
                lines.append(f"{'#'*level} {htxt} [[SEC id={sec_id}|page={page}]]")
            for child in el:
                render_node(child, div_depth + 1 if etree.QName(child).localname.lower() == 'div' else div_depth)
            return
        # Paragraph
        if tag == 'p':
            txt = _text(el)
            if txt:
                lines.append(txt)
            return
        # List
        if tag == 'list':
            for it in el.findall('./tei:item', namespaces=ns):
                itxt = _text(it)
                if itxt:
                    lines.append(f"- {itxt}")
            return
        # Table (very simple)
        if tag == 'table':
            rows = []
            for row in el.findall('.//tei:row', namespaces=ns):
                cells = [_text(c) for c in row.findall('./tei:cell', namespaces=ns)]
                rows.append(cells)
            if rows:
                header = rows[0]
                if header:
                    lines.append('| ' + ' | '.join(header) + ' |')
                    lines.append('| ' + ' | '.join(['---'] * len(header)) + ' |')
                    for r in rows[1:]:
                        lines.append('| ' + ' | '.join(r) + ' |')
            return
        # Figure caption
        if tag == 'figure':
            cap = el.find('./tei:figDesc', namespaces=ns)
            ctxt = _text(cap)
            if ctxt:
                lines.append(f"_Figure:_ {ctxt}")
            return
        # Recurse into unknown containers to preserve order
        for child in el:
            render_node(child, div_depth)

    for child in body:
        render_node(child, div_depth=1)

    # Ensure blank lines between blocks
    md = []
    for ln in lines:
        if md and md[-1] and ln and not (md[-1].startswith('<!--PAGE:') or ln.startswith('<!--PAGE:')):
            md.append('')
        md.append(ln)
    out = '\n'.join(md).strip() + '\n'
    return sanitize_markdown(out)


def extract_plain_text_from_tei(tei_xml: str) -> str:
    try:
        from lxml import etree  # type: ignore
    except Exception:
        return tei_xml or ''
    try:
        root = etree.fromstring((tei_xml or '').encode('utf-8'))
        text = ' '.join(root.itertext())
        return text
    except Exception:
        return tei_xml or ''


def sanitize_markdown(md: str) -> str:
    """
    Remove any residual XML/HTML tags, namespace artifacts, and collapse whitespace.
    Convert simple HTML-ish tables, normalize quotes/dashes, and dehyphenate line breaks.
    Keeps our page markers intact.
    """
    import re
    # Convert HTML-ish <table><row><cell>... blocks to Markdown tables first
    def _convert_tables(text: str) -> str:
        tbl_rx = re.compile(r"<table>([\s\S]*?)</table>", re.IGNORECASE)
        row_rx = re.compile(r"<row>([\s\S]*?)</row>", re.IGNORECASE)
        cell_rx = re.compile(r"<cell>([\s\S]*?)</cell>", re.IGNORECASE)
        def _one_table(match: re.Match) -> str:
            inner = match.group(1)
            rows = []
            for r in row_rx.findall(inner):
                cells = [c.strip() for c in cell_rx.findall(r)] or []
                rows.append(cells)
            if not rows:
                return ''
            width = max((len(r) for r in rows), default=0)
            rows = [(r + [''] * (width - len(r))) for r in rows]
            md_lines = []
            header = rows[0]
            md_lines.append('| ' + ' | '.join(header) + ' |')
            md_lines.append('| ' + ' | '.join(['---'] * width) + ' |')
            for r in rows[1:]:
                md_lines.append('| ' + ' | '.join(r) + ' |')
            return '\n'.join(md_lines)
        return tbl_rx.sub(_one_table, text)

    md = _convert_tables(md or '')
    # Strip any tags that are not our page markers
    def _strip_tags(text: str) -> str:
        text = (text or '')
        text = text.replace('<!--PAGE:', '[[__PAGE_MARKER__:')
        text = re.sub(r'<[^>]+>', '', text)
        text = text.replace('[[__PAGE_MARKER__:', '<!--PAGE:')
        return text
    md = _strip_tags(md)
    md = re.sub(r'\bxmlns(?::\w+)?\s*=\s*"[^"]*"', '', md)
    # Remove publisher boilerplate
    boilerplate_rx = re.compile(r"^\s*(Downloaded from|See the Terms and Conditions|Wiley Online Library|Creative Commons License).*\s*$", re.IGNORECASE)
    lines_all = (md or '').splitlines()
    lines = []
    for ln in lines_all:
        if boilerplate_rx.match(ln):
            continue
        lines.append(ln)
    md = "\n".join(lines)
    # Normalize unicode quotes/dashes
    md = (md.replace('\u2013', '-')
            .replace('\u2014', '-')
            .replace('\u2019', "'")
            .replace('\u2018', "'")
            .replace('\u201c', '"')
            .replace('\u201d', '"'))
    # De-hyphenate line-break splits: word-\nword → wordword
    md = re.sub(r"(\w)-\n(\w)", r"\1\2", md)
    # Collapse intra-paragraph newlines while preserving structure
    blocks = md.split("\n\n")
    joined_blocks = []
    for b in blocks:
        b_stripped = (b or '').strip()
        if not b_stripped:
            joined_blocks.append("")
            continue
        first = b_stripped.lstrip()
        if first.startswith('#') or first.startswith('|') or first.startswith('- ') or first.startswith('<!--PAGE:') or first.startswith('```'):
            joined_blocks.append(b_stripped)
        else:
            one = re.sub(r"\s*\n\s*", " ", b_stripped)
            joined_blocks.append(one)
    md = "\n\n".join(joined_blocks)
    # Normalize line endings and collapse multiple blank lines
    out_lines = []
    for ln in (md or '').splitlines():
        if ln.strip() == '' and (out_lines and out_lines[-1] == ''):
            continue
        out_lines.append(ln.rstrip())
    return '\n'.join(out_lines).strip() + '\n'

import os
import time
from typing import Optional

import requests


def grobid_fulltext_tei(pdf_path: str, base_url: Optional[str] = None, timeout: int = 180, retries: int = 2, backoff: float = 1.5) -> str:
    """
    Call GROBID /api/processFulltextDocument and return TEI XML as text.
    """
    url = (base_url or os.getenv('GROBID_URL') or 'http://localhost:8070').rstrip('/') + '/api/processFulltextDocument'
    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            with open(pdf_path, 'rb') as f:
                files = {'input': (os.path.basename(pdf_path), f, 'application/pdf')}
                data = {
                    'consolidateHeader': '1',
                    'consolidateCitations': '0',
                    'includeRawCitations': '0',
                    'teiCoordinates': 'p,head,figure,table,ref,item,row',
                    'segmentSentences': '1',
                }
                resp = requests.post(url, files=files, data=data, timeout=timeout)
                resp.raise_for_status()
                tei_text = resp.text
                # Persist TEI immediately and unconditionally to a stable location for audit
                try:
                    # Allow override via env; otherwise save under <repo>/storage/grobid_tei_raw/
                    save_dir = os.getenv('GROBID_TEI_SAVE_DIR')
                    if not save_dir:
                        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
                        save_dir = os.path.join(repo_root, 'storage', 'grobid_tei_raw')
                    os.makedirs(save_dir, exist_ok=True)
                    stem = os.path.splitext(os.path.basename(pdf_path))[0]
                    out_path = os.path.join(save_dir, f"{stem}.tei.xml")
                    with open(out_path, 'w', encoding='utf-8') as outf:
                        outf.write(tei_text or '')
                except Exception:
                    pass
                return tei_text
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(backoff ** attempt)
            else:
                raise
    raise RuntimeError(f"GROBID failed: {last_err}")


def tei_to_markdown(tei_xml: str) -> str:
    """
    TEI → Markdown conversion with inline page markers and ordered traversal.
    - Emits headings from nested tei:div/tei:head with appropriate levels
    - Preserves content order
    - Inserts <!--PAGE:n--> when encountering tei:pb in flow
    - Handles paragraphs, lists, simple tables, and figure captions minimally
    """
    try:
        from lxml import etree  # type: ignore
    except Exception:
        # Fallback: sanitize tag-stripped text
        return sanitize_markdown(extract_plain_text_from_tei(tei_xml))

    try:
        root = etree.fromstring(tei_xml.encode('utf-8'))
    except Exception:
        return sanitize_markdown(extract_plain_text_from_tei(tei_xml))

    ns = {'tei': 'http://www.tei-c.org/ns/1.0'}

    def _text(el) -> str:
        return ''.join(el.itertext()).strip() if el is not None else ''

    lines = []
    # Title
    title_el = root.find('.//tei:teiHeader//tei:titleStmt/tei:title', namespaces=ns)
    title_txt = _text(title_el)
    if title_txt:
        lines.append(f"# {title_txt}")

    body = root.find('.//tei:text/tei:body', namespaces=ns)
    if body is None:
        return ('\n'.join(lines) + '\n') if lines else extract_plain_text_from_tei(tei_xml)

    # Maintain hierarchical section ids via path counters
    path_stack = []  # list of ints per depth

    def render_node(el, div_depth: int = 0):
        tag = etree.QName(el).localname.lower()
        # Page break
        if tag == 'pb':
            n = el.get('n') or el.get('{http://www.w3.org/XML/1998/namespace}id') or ''
            try:
                num = int(str(n)) if str(n).isdigit() else None
            except Exception:
                num = None
            if num is not None:
                lines.append(f"<!--PAGE:{num}-->")
            return
        # Division / heading
        if tag == 'div':
            head = el.find('./tei:head', namespaces=ns)
            htxt = _text(head)
            if htxt:
                level = min(6, max(2, 1 + div_depth))
                # Update path stack
                if len(path_stack) < level:
                    # new deeper level
                    path_stack.extend([0] * (level - len(path_stack)))
                # increment current level counter and reset deeper levels
                path_stack[level-1] = (path_stack[level-1] if level-1 < len(path_stack) else 0) + 1
                for i in range(level, len(path_stack)):
                    path_stack[i] = 0
                # Build sec id from non-zero prefix
                non_zero = [str(x) for x in path_stack if x > 0]
                sec_id = '.'.join(non_zero) if non_zero else '1'
                # Determine latest page from previous markers
                # Find the last page marker in lines
                page = ''
                for prev in reversed(lines):
                    if isinstance(prev, str) and prev.startswith('<!--PAGE:') and prev.endswith('-->'):
                        try:
                            page = prev[len('<!--PAGE:'):-3]
                        except Exception:
                            page = ''
                        break
                lines.append(f"{'#'*level} {htxt} [[SEC id={sec_id}|page={page}]]")
            for child in el:
                render_node(child, div_depth + 1 if etree.QName(child).localname.lower() == 'div' else div_depth)
            return
        # Paragraph
        if tag == 'p':
            txt = _text(el)
            if txt:
                lines.append(txt)
            return
        # List
        if tag == 'list':
            for it in el.findall('./tei:item', namespaces=ns):
                itxt = _text(it)
                if itxt:
                    lines.append(f"- {itxt}")
            return
        # Table (very simple)
        if tag == 'table':
            rows = []
            for row in el.findall('.//tei:row', namespaces=ns):
                cells = [ _text(c) for c in row.findall('./tei:cell', namespaces=ns) ]
                rows.append(cells)
            if rows:
                # Header + separator if possible
                header = rows[0]
                if header:
                    lines.append('| ' + ' | '.join(header) + ' |')
                    lines.append('| ' + ' | '.join(['---'] * len(header)) + ' |')
                    for r in rows[1:]:
                        lines.append('| ' + ' | '.join(r) + ' |')
            return
        # Figure caption
        if tag == 'figure':
            cap = el.find('./tei:figDesc', namespaces=ns)
            ctxt = _text(cap)
            if ctxt:
                lines.append(f"_Figure:_ {ctxt}")
            return
        # Recurse into unknown containers to preserve order
        for child in el:
            render_node(child, div_depth)

    for child in body:
        render_node(child, div_depth=1)

    # Ensure blank lines between blocks
    md = []
    for ln in lines:
        if md and md[-1] and ln and not (md[-1].startswith('<!--PAGE:') or ln.startswith('<!--PAGE:')):
            md.append('')
        md.append(ln)
    out = '\n'.join(md).strip() + '\n'
    return sanitize_markdown(out)


def extract_plain_text_from_tei(tei_xml: str) -> str:
    try:
        from lxml import etree  # type: ignore
    except Exception:
        return tei_xml


def sanitize_markdown(md: str) -> str:
    """
    Remove any residual XML/HTML tags, namespace artifacts, and collapse whitespace.
    Keeps our page markers intact.
    """
    import re
    # Convert HTML-ish <table><row><cell>... blocks to Markdown tables first
    def _convert_tables(text: str) -> str:
        tbl_rx = re.compile(r"<table>([\s\S]*?)</table>", re.IGNORECASE)
        row_rx = re.compile(r"<row>([\s\S]*?)</row>", re.IGNORECASE)
        cell_rx = re.compile(r"<cell>([\s\S]*?)</cell>", re.IGNORECASE)
        def _one_table(match: re.Match) -> str:
            inner = match.group(1)
            rows = []
            for r in row_rx.findall(inner):
                cells = [c.strip() for c in cell_rx.findall(r)] or []
                rows.append(cells)
            if not rows:
                return ''
            # Normalize column count
            width = max((len(r) for r in rows), default=0)
            rows = [ (r + [''] * (width - len(r))) for r in rows ]
            # Build Markdown
            md_lines = []
            header = rows[0]
            md_lines.append('| ' + ' | '.join(header) + ' |')
            md_lines.append('| ' + ' | '.join(['---'] * width) + ' |')
            for r in rows[1:]:
                md_lines.append('| ' + ' | '.join(r) + ' |')
            return '\n'.join(md_lines)
        return tbl_rx.sub(_one_table, text)
    md = _convert_tables(md)
    # Strip any tags that are not our page markers
    def _strip_tags(text: str) -> str:
        # Temporarily protect page markers
        text = text.replace('<!--PAGE:', '[[__PAGE_MARKER__:')
        text = re.sub(r'<[^>]+>', '', text)
        text = text.replace('[[__PAGE_MARKER__:', '<!--PAGE:')
        return text
    md = _strip_tags(md)
    # Remove xmlns-like substrings if any leaked into text
    md = re.sub(r'\bxmlns(?::\w+)?\s*=\s*"[^"]*"', '', md)
    # Normalize line endings and collapse multiple blank lines
    lines = [ln.rstrip() for ln in md.splitlines()]
    out_lines = []
    for ln in lines:
        if ln.strip() == '' and (out_lines and out_lines[-1] == ''):
            continue
        out_lines.append(ln)
    return '\n'.join(out_lines).strip() + '\n'
    try:
        root = etree.fromstring(tei_xml.encode('utf-8'))
        text = ' '.join(root.itertext())
        return text
    except Exception:
        return tei_xml


