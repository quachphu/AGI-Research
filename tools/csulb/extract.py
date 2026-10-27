"""
Turn the scraped pages and files into clean text and policy sections.

    .venv/bin/python tools/csulb/extract.py          # reads school_policies_data/

Writes, inside the data folder:
    text/<slug>.txt     one document each, starting with a short header
    documents.jsonl     one line per document kept (or dropped, with why)
    sections.jsonl      the documents cut at their headings, about 1,500
                        characters at most, for retrieval and for the Q&A

PDFs go through `pdftotext` (reading order, not -layout: -layout pads with
runs of spaces that the model would have to learn to copy). Word and RTF go
through LibreOffice. Web pages keep only the main content block.
"""
import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from collections import Counter
from urllib.parse import urldefrag, urljoin, urlparse

from bs4 import BeautifulSoup, NavigableString

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from minagi.ingest import why_not_text  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_corpus import clean_title  # noqa: E402

SECTION_MAX = 1500
SECTION_MIN = 120
DOC_MIN = 150
MONTHS = ("January|February|March|April|May|June|July|August|September|"
          "October|November|December")
DATE_RE = re.compile(rf"\b({MONTHS})\s+\d{{1,2}},\s+(19|20)\d\d\b")
PS_RE = re.compile(r"\b(?:Policy Statement|PS)[\s#:]*(\d{2}-\d{2})\b", re.I)
# the house style for every heading in text/*.txt
H = "## "

# ---------------------------------------------------------------- normalise
TRANS = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-",
    "\u2014": " - ", "\u2022": "-", "\u25cf": "-", "\u25aa": "-", "\uf0b7": "-",
    "\u00a0": " ", "\u2002": " ", "\u2003": " ", "\u2009": " ", "\u200b": "",
    "\ufeff": "", "\u00ad": "", "\uf0a7": "-", "\u2026": "...", "\u25e6": "-",
    "\u2212": "-", "\u00b7": "-", "\u2010": "-", "\u2011": "-",
})


def norm(s):
    s = unicodedata.normalize("NFKC", s).translate(TRANS)
    s = "".join(ch for ch in s if ch in "\n\t" or unicodedata.category(ch)[0] != "C")
    # Word's PDF export writes a hyphen as "-", soft hyphen, U+2010
    s = re.sub(r"(?<=\w)-{2,}", "-", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r" *\n *", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


# ---------------------------------------------------------------- html
BLOCK = {"p", "div", "section", "article", "ul", "ol", "li", "table", "tr",
         "blockquote", "h1", "h2", "h3", "h4", "h5", "h6", "dl", "dt", "dd",
         "br", "hr", "figure", "header", "footer", "pre"}
DROP_CLASS = re.compile(r"breadcrumb|sidebar|share|social|visually-hidden|"
                        r"sr-only|skip|pager|back-to-top|print", re.I)


def html_text(html):
    soup = BeautifulSoup(html, "html.parser")
    title = (soup.title.get_text(" ", strip=True) if soup.title else "").split("|")[0].strip()
    m = None
    for sel in ("main", "#main-content", "article", "#content", ".item-page", "[role=main]"):
        m = soup.select_one(sel)
        if m is not None:
            break
    m = m or soup.body or soup
    for t in m(["script", "style", "noscript", "form", "button", "iframe", "svg",
                "img", "nav", "aside", "select", "input"]):
        t.decompose()
    for t in m.find_all(class_=DROP_CLASS):
        t.decompose()
    # a tab strip names the panes after it: turn each name into the pane's heading
    for strip in m.select("ul.nav-tabs"):
        names = [li.get_text(" ", strip=True) for li in strip.find_all("li")]
        box = strip.find_parent(class_="tabs-wrapper") or strip.parent
        panes = box.select(".tab-pane") if box else []
        for name, pane in zip(names, panes):
            h = soup.new_tag("h2")
            h.string = name
            pane.insert(0, h)
        strip.decompose()
    for t in m.select(".accordion-button, .accordion-header, .accordion__title, summary"):
        h = soup.new_tag("h3")
        h.string = t.get_text(" ", strip=True)
        t.replace_with(h)

    out = []

    def walk(node):
        for c in node.children:
            if isinstance(c, NavigableString):
                out.append(str(c))
                continue
            name = c.name
            if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
                txt = c.get_text(" ", strip=True)
                if txt:
                    out.append(f"\n\n{H}{txt}\n\n")
                continue
            if name == "li":
                out.append("\n- ")
                walk(c)
                out.append("\n")
                continue
            if name == "tr":
                cells = [td.get_text(" ", strip=True) for td in c.find_all(["td", "th"])]
                out.append("\n" + " | ".join(x for x in cells if x) + "\n")
                continue
            if name == "br":
                out.append("\n")
                continue
            if name in BLOCK:
                out.append("\n\n")
                walk(c)
                out.append("\n\n")
            else:
                walk(c)

    walk(m)
    text = norm("".join(out))
    text = re.sub(r"\n- \n", "\n", text)
    links = len(m.find_all("a"))
    return title, text, links


# ---------------------------------------------------------------- pdf / office
PAGE_NO = re.compile(r"^\s*(page\s+)?\d{1,3}(\s+of\s+\d{1,3})?\s*$", re.I)
HEADINGISH = re.compile(r"^(\d+(\.\d+)*\.?|[IVX]+\.|[A-Z]\.|Article [IVX\d]+|"
                        r"Section \d+|ARTICLE|SECTION)\s+\S", re.I)
BULLET = re.compile(r"^(-|\*|o|\d+[.)]|[a-z][.)]|\([a-z0-9]+\))\s")


def pdf_text(path):
    try:
        raw = subprocess.run(["pdftotext", "-enc", "UTF-8", path, "-"], check=True,
                             capture_output=True, timeout=120).stdout
    except (subprocess.SubprocessError, OSError):
        return ""
    # Some exports put a lone \r after every word; splitlines() would make each
    # word a line and a lone number would then pass for a page number.
    raw = raw.decode("utf-8", errors="replace").replace("\r", " ")
    pages = raw.split("\f")
    # a line on most pages is a running header or footer
    counts = Counter()
    for p in pages:
        for line in set(x.strip() for x in p.split("\n") if x.strip()):
            counts[line] += 1
    n = len([p for p in pages if p.strip()])
    repeated = {l for l, c in counts.items() if n >= 3 and c >= max(3, n * 0.5) and len(l) < 120}
    lines = []
    for p in pages:
        for line in p.split("\n"):
            s = line.strip()
            if not s:
                lines.append("")
                continue
            if PAGE_NO.match(s) or s in repeated:
                continue
            lines.append(s)
        lines.append("")
    return join_lines(lines)


def join_lines(lines):
    """Undo the hard wrap of a PDF: a short line ending mid-sentence continues."""
    out = []
    for s in lines:
        s = re.sub(r"[ \t\u00a0]+", " ", s).strip()
        if not s:
            if out and out[-1] != "":
                out.append("")
            continue
        # double-spaced documents leave a blank line inside a sentence
        if len(out) >= 2 and out[-1] == "" and s[:1].islower() and \
                not out[-2].endswith((".", ":", ";", "?", "!")) and not is_heading_line(out[-2]):
            out.pop()
        prev = out[-1] if out else ""
        # a list marker on a line of its own belongs to the line after it
        if prev and re.fullmatch(r"(\d+(\.\d+)*\.?|[A-Za-z]\.|\([a-z0-9]+\)|[IVX]+\.)", prev):
            out[-1] = prev + " " + s
            continue
        if (prev and prev != "" and not is_heading_line(prev) and not HEADINGISH.match(s)
                and not BULLET.match(s) and not prev.endswith((":", ";"))
                and (not prev.endswith((".", "?", "!")) or s[:1].islower())):
            if prev.endswith("-") and s[:1].islower():
                out[-1] = prev[:-1] + s
            else:
                out[-1] = prev + " " + s
        else:
            out.append(s)
    return norm("\n".join(out))


def office_texts(paths, tmp):
    """LibreOffice converts a batch at once; it is slow to start."""
    if not paths or not shutil.which("soffice"):
        return {}
    env = dict(os.environ, HOME=tmp)
    for i in range(0, len(paths), 40):
        batch = paths[i:i + 40]
        subprocess.run(["soffice", "--headless", "--norestore", "--convert-to",
                        "txt:Text (encoded):UTF8", "--outdir", tmp] + batch,
                       capture_output=True, timeout=900, env=env)
    out = {}
    for p in paths:
        t = os.path.join(tmp, os.path.splitext(os.path.basename(p))[0] + ".txt")
        if os.path.exists(t):
            with open(t, encoding="utf-8", errors="replace") as f:
                out[p] = norm(f.read())
    return out


# ---------------------------------------------------------------- headings
def is_heading_line(s):
    s = s.strip()
    if s.startswith(H):
        return True
    if not s or len(s) > 90 or s.endswith((".", ",", ";")) or len(s.split()) > 12:
        return False
    letters = [c for c in s if c.isalpha()]
    if len(letters) < 3:
        return False
    upper = sum(c.isupper() for c in letters) / len(letters)
    if upper > 0.8:
        return True
    m = re.match(r"^(\d+(\.\d+)*\.?|[IVX]+\.|[A-Z]\.|Article [IVX\d]+\.?|Section \d+\.?)\s+(.+)$", s)
    if m:
        rest = m.group(3)
        words = rest.split()
        caps = sum(w[:1].isupper() for w in words if len(w) > 3)
        return len(words) <= 10 and caps >= max(1, (len([w for w in words if len(w) > 3]) + 1) // 2)
    return False


def mark_headings(text):
    lines = text.split("\n")
    out = []
    for i, s in enumerate(lines):
        isolated = (i == 0 or not lines[i - 1].strip()) or (i + 1 < len(lines) and not lines[i + 1].strip())
        if not s.startswith(H) and isolated and "|" not in s and is_heading_line(s):
            out.append(H + s.strip())
        else:
            out.append(s)
    return "\n".join(out)


# ---------------------------------------------------------------- metadata
def own_number(head, title, url):
    """The document's own number, not one of those it supersedes."""
    fname = urlparse(url or "").path.rsplit("/", 1)[-1]
    tries = [re.search(r"Policy\s+Number\s*:?\s*(\d{2}-\d{2})", head, re.I),
             re.search(r"Policy\s+Statement\s*:?\s*(\d{2}-\d{2})", head[:600], re.I),
             re.search(r"policy-statement-(\d{2}-\d{2})", url or ""),
             re.match(r"(?:ps[_ %20-]*)?(\d{2}-\d{2})", fname, re.I)]
    for m in tries:
        if m:
            return m.group(1)
    for m in list(PS_RE.finditer(title or "")) + list(PS_RE.finditer(head)):
        src = title if m.string is title else head
        before = src[max(0, m.start() - 80):m.start()].lower()
        if "supersede" not in before and "see" not in before[-10:]:
            return m.group(1)
    return None


def meta_for(text, title, url):
    head = text[:2500]
    num = own_number(head, title, url)
    date = DATE_RE.search(head)
    sup = re.search(r"supersedes?\s+([^.\n)]{3,120})", head, re.I)
    approved = re.search(r"approved by the ([A-Z][A-Za-z ]{2,40}?)(?: on| in|,|\.)", head)
    recommended = re.search(r"recommended by the ([A-Z][A-Za-z ]{2,40}?)(?: on| in|,|\.)", head)
    return dict(policy_number=f"PS {num}" if num else None,
                date=date.group(0) if date else None,
                supersedes=sup.group(1).strip() if sup else None,
                approved_by=approved.group(1).strip() if approved else None,
                recommended_by=recommended.group(1).strip() if recommended else None)


GENERIC = re.compile(r"^\W*(pdf|doc|docx|rtf|word|download|here|link|view|"
                     r"ps \d{2}-\d{2})?\W*(\[?(pdf|doc|docx)\]?)?\W*$", re.I)


def link_names(D, recs):
    """
    What each file is called on the page that links it.

    A listing names a policy once, on the link to its page, and then links the
    file as "PDF" or "DOC": the name is the link before.
    """
    names = {}
    for r in recs:
        if r.get("type") != "html":
            continue
        with open(os.path.join(D, r["file"]), encoding="utf-8", errors="replace") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
        main = soup.find("main") or soup.body or soup
        base = r.get("final_url") or r["url"]
        last = None
        for a in main.find_all("a", href=True):
            href = urldefrag(urljoin(base, a["href"].strip()))[0]
            text = " ".join(a.get_text(" ", strip=True).split())
            low = href.lower().split("?")[0]
            if low.endswith((".pdf", ".doc", ".docx", ".rtf")):
                if text and not GENERIC.match(text):
                    names.setdefault(href, text)
                elif last:
                    names.setdefault(href, last)
            elif text and not GENERIC.match(text):
                last = text
    return names


def good_title(title, text, fallback):
    t = " ".join((title or "").split())
    t = re.sub(r"\s*[(\[](PDF|DOCX?|RTF)[)\]]", "", t, flags=re.I).strip()
    if GENERIC.match(t):
        t = ""
    # the campus rules pages are titled "REGULATION XIX" and named on the next line
    if re.fullmatch(r"(?i)regulation [ivxl]+", t):
        for line in text.split("\n")[:6]:
            s = line.replace(H, "").strip()
            if s and s.lower() != t.lower() and len(s) < 100:
                name = s.title() if s.isupper() else s
                return f"Regulation {t.split()[1].upper()}: {name}"
    bad = {"", "policies", "document", "untitled", "microsoft word"}
    if t.lower() in bad or t.lower().startswith("microsoft word") or len(t) < 4:
        for line in text.split("\n")[:25]:
            s = line.replace(H, "").strip()
            if 8 <= len(s) <= 100 and sum(c.isalpha() for c in s) > 6 and \
                    not DATE_RE.search(s) and "california state university" not in s.lower():
                return s.title() if s.isupper() else s
        return fallback
    return t


# ---------------------------------------------------------------- sections
def split_sections(text):
    """[(heading, number, body)] cut at headings, long bodies cut at paragraphs."""
    blocks, heading, buf = [], None, []

    def flush():
        body = "\n".join(buf).strip()
        if body:
            blocks.append((heading, body))

    for line in text.split("\n"):
        if line.startswith(H):
            flush()
            buf = []
            heading = line[len(H):].strip()
        else:
            buf.append(line)
    flush()

    out = []
    for heading, body in blocks:
        paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        piece = ""
        for p in paras:
            while len(p) > SECTION_MAX:
                cut = p.rfind(". ", 0, SECTION_MAX)
                cut = cut + 1 if cut > SECTION_MAX // 3 else SECTION_MAX
                if piece:
                    out.append((heading, piece))
                    piece = ""
                out.append((heading, p[:cut].strip()))
                p = p[cut:].strip()
            if piece and len(piece) + len(p) + 2 > SECTION_MAX:
                out.append((heading, piece))
                piece = ""
            piece = (piece + "\n\n" + p).strip()
        if piece:
            out.append((heading, piece))
    # glue a too-short piece onto its neighbour under the same heading
    merged = []
    for h, b in out:
        if merged and merged[-1][0] == h and (len(b) < SECTION_MIN or len(merged[-1][1]) < SECTION_MIN) \
                and len(merged[-1][1]) + len(b) < SECTION_MAX * 1.3:
            merged[-1] = (h, merged[-1][1] + "\n\n" + b)
        else:
            merged.append((h, b))
    # A stub - a title block, a lone date line - carries no answer, and BM25
    # favours short texts, so it would win searches it cannot answer. It
    # joins the section after it.
    glued, carry = [], None
    for i, (h, b) in enumerate(merged):
        if carry:
            b = carry[1] + "\n\n" + b
            h = h or carry[0]
            carry = None
        if len(b) < SECTION_MIN * 2 and i + 1 < len(merged):
            carry = (h, b)
            continue
        glued.append((h, b))
    if carry:
        if glued:
            glued[-1] = (glued[-1][0], glued[-1][1] + "\n\n" + carry[1])
        else:
            glued.append(carry)
    merged = glued
    res = []
    for h, b in merged:
        num = None
        m = re.match(r"^(\d+(\.\d+)*)\.?\s", (h or "")) or re.match(r"^(\d+(\.\d+)+)\.?\s", b)
        if m:
            num = m.group(1)
        res.append((h, num, b))
    return res


# ---------------------------------------------------------------- dedupe
def shingles(text, k=6):
    w = re.findall(r"[a-z0-9]+", text.lower())
    return {hash(" ".join(w[i:i + k])) for i in range(0, max(1, len(w) - k + 1))}


def slugify(s, n=70):
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()
    return s[:n].strip("_") or "doc"


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--data", default="school_policies_data")
    a = ap.parse_args()
    D = a.data
    with open(os.path.join(D, "manifest.jsonl"), encoding="utf-8") as f:
        recs = [json.loads(l) for l in f if l.strip()]
    ok = [r for r in recs if r["status"] == "ok"]
    names = link_names(D, ok)
    page_title = {r["url"]: r.get("title") for r in ok if r.get("type") == "html"}
    by_sha = {}
    docs, dropped = [], []

    tmp = tempfile.mkdtemp(prefix="csulb_office_")
    office = office_texts([os.path.join(D, r["file"]) for r in ok
                           if r.get("type") in ("doc", "docx", "rtf")], tmp)

    for r in ok:
        if "/form/" in r["url"]:
            dropped.append(dict(url=r["url"], why="a contact form, not a policy"))
            continue
        if r["sha256"] in by_sha:
            dropped.append(dict(url=r["url"], why=f"same file as {by_sha[r['sha256']]}"))
            continue
        by_sha[r["sha256"]] = r["url"]
        path = os.path.join(D, r["file"])
        listing = False
        if r["type"] == "html":
            with open(path, encoding="utf-8", errors="replace") as f:
                title, text, links = html_text(f.read())
            # a page that is mostly links is an index of policies, not a policy
            listing = links >= 30 and len(text) / max(links, 1) < 160
        else:
            # a file on a policy's own page is that policy; on a listing it is
            # whatever the listing calls it
            title = names.get(r["url"]) or r.get("title") or ""
            if r.get("depth", 0) >= 3 and page_title.get(r.get("parent")):
                title = page_title[r["parent"]]
            text = pdf_text(path) if r["type"] == "pdf" else office.get(path, "")
        if len(text) < DOC_MIN:
            dropped.append(dict(url=r["url"], why="no_text" if r["type"] == "pdf" and len(text) < 50
                                else f"too little text ({len(text)} chars)"))
            continue
        text = mark_headings(text)
        url = r.get("final_url") or r["url"]
        name = os.path.splitext(os.path.basename(r["url"].split("?")[0]))[0]
        title = good_title(title, text, name.replace("_", " ").replace("%20", " "))
        docs.append(dict(url=url, src=r["url"], type=r["type"], section=r["section"],
                         title=title, text=text, listing=listing,
                         **meta_for(text, title, url)))
    shutil.rmtree(tmp, ignore_errors=True)

    # near-duplicates: the same policy as a page and as a PDF. Keep the page
    # (it has the address people use), else the longer copy.
    sh = [shingles(d["text"]) for d in docs]
    keep = [True] * len(docs)
    order = sorted(range(len(docs)), key=lambda i: (docs[i]["type"] != "html", -len(docs[i]["text"])))
    for ii, i in enumerate(order):
        if not keep[i]:
            continue
        for j in order[ii + 1:]:
            if not keep[j]:
                continue
            li, lj = len(docs[i]["text"]), len(docs[j]["text"])
            if min(li, lj) / max(li, lj) < 0.5:
                continue
            inter = len(sh[i] & sh[j])
            if inter / max(1, min(len(sh[i]), len(sh[j]))) > 0.8:
                keep[j] = False
                dropped.append(dict(url=docs[j]["src"], why=f"near-duplicate of {docs[i]['src']}"))

    os.makedirs(os.path.join(D, "text"), exist_ok=True)
    for old in glob.glob(os.path.join(D, "text", "*.txt")):
        os.remove(old)
    kept, sections, used = [], [], set()
    for d, k in zip(docs, keep):
        if not k:
            continue
        slug = slugify(d["title"])
        if slug in used:
            slug = f"{slug}_{hashlib.sha1(d['src'].encode()).hexdigest()[:6]}"
        used.add(slug)
        header = [f"Title: {clean_title(d['title'])}"]
        if d["policy_number"]:
            header.append(f"Policy number: {d['policy_number']}")
        if d["date"]:
            header.append(f"Date: {d['date']}")
        header += [f"Source: {d['url']}", f"Office: {d['section']}"]
        body = "\n".join(header) + "\n\n" + d["text"] + "\n"
        rel = os.path.join("text", slug + ".txt")
        with open(os.path.join(D, rel), "w", encoding="utf-8") as f:
            f.write(body)
        why = why_not_text(os.path.join(D, rel))
        if why:
            os.remove(os.path.join(D, rel))
            dropped.append(dict(url=d["src"], why=f"trainer would skip it: {why}"))
            continue
        meta = {k2: v for k2, v in d.items() if k2 != "text"}
        meta.update(id=slug, file=rel, chars=len(body))
        kept.append(meta)
        if d["listing"]:
            continue
        for n, (heading, num, btxt) in enumerate(split_sections(d["text"])):
            if len(btxt) < SECTION_MIN and not heading:
                continue
            sections.append(dict(id=f"{slug}#{n}", doc=slug, title=d["title"],
                                 policy_number=d["policy_number"], heading=heading,
                                 number=num, text=btxt, url=d["url"], office=d["section"]))

    with open(os.path.join(D, "documents.jsonl"), "w", encoding="utf-8") as f:
        for m in kept:
            f.write(json.dumps(dict(m, kept=True), ensure_ascii=False) + "\n")
        for x in dropped:
            f.write(json.dumps(dict(x, kept=False), ensure_ascii=False) + "\n")
    with open(os.path.join(D, "sections.jsonl"), "w", encoding="utf-8") as f:
        for s in sections:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    total = sum(m["chars"] for m in kept)
    print(f"{len(kept)} documents ({total / 1e6:.2f}M chars), {len(sections)} sections, "
          f"{len(dropped)} dropped")
    reasons = Counter(x["why"].split(" of ")[0].split(" as ")[0].split(":")[0] for x in dropped)
    for k, v in reasons.most_common():
        print(f"  dropped {v:4d}  {k}")


if __name__ == "__main__":
    main()
