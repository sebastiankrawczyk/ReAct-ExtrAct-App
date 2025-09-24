import os
import sys
import argparse

from utils.grobid_client import grobid_fulltext_tei, tei_to_markdown


def main() -> int:
    p = argparse.ArgumentParser(description="Convert TEI XML (or PDF via GROBID) to sanitized Markdown with page markers.")
    p.add_argument("input", help="Path to TEI XML or PDF (when --from-pdf)")
    p.add_argument("-o", "--output", help="Output .md path (default: stdout)")
    p.add_argument("--from-pdf", action="store_true", help="Treat input as PDF and fetch TEI via GROBID")
    p.add_argument("--grobid-url", default=os.getenv("GROBID_URL", "http://localhost:8070"), help="GROBID base URL (default from GROBID_URL env)")
    args = p.parse_args()

    try:
        if args.from_pdf:
            tei = grobid_fulltext_tei(args.input, base_url=args.grobid_url)
        else:
            with open(args.input, "r", encoding="utf-8") as f:
                tei = f.read()
        md = tei_to_markdown(tei)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(md)
        else:
            sys.stdout.write(md)
        return 0
    except Exception as e:
        sys.stderr.write(f"Error: {e}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

import os
import sys
import argparse

from utils.grobid_client import grobid_fulltext_tei, tei_to_markdown


def main() -> int:
    p = argparse.ArgumentParser(description="Convert TEI XML (or PDF via GROBID) to sanitized Markdown with page markers.")
    p.add_argument("input", help="Path to TEI XML or PDF (when --from-pdf)")
    p.add_argument("-o", "--output", help="Output .md path (default: stdout)")
    p.add_argument("--from-pdf", action="store_true", help="Treat input as PDF and fetch TEI via GROBID")
    p.add_argument("--grobid-url", default=os.getenv("GROBID_URL", "http://localhost:8070"), help="GROBID base URL (default from GROBID_URL env)")
    args = p.parse_args()

    try:
        if args.from_pdf:
            tei = grobid_fulltext_tei(args.input, base_url=args.grobid_url)
        else:
            with open(args.input, "r", encoding="utf-8") as f:
                tei = f.read()
        md = tei_to_markdown(tei)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(md)
        else:
            sys.stdout.write(md)
        return 0
    except Exception as e:
        sys.stderr.write(f"Error: {e}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


