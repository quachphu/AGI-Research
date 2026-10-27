"""
Download every publicly reachable policy linked from csulb.edu/university-policies.

    .venv/bin/python tools/csulb/scrape.py            # into school_policies_data/

The crawl is shallow on purpose: the index (depth 0), the policy sections it
links to (depth 1), and the policy pages those sections list (depth 2). Every
PDF found on any of them is kept. Word and RTF files are kept only when the
page that lists them does not already carry the policy as text, since on the
Academic Senate pages they are the same policy a third time.

Politeness: robots.txt is obeyed per host (a disallowed URL is never fetched),
and no host sees two requests closer than max(--delay, its Crawl-delay). Hosts
are interleaved, so waiting on one does not hold up the others.

Re-running is cheap: anything already in manifest.jsonl with a file on disk is
read back from disk instead of fetched.
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.robotparser
from collections import deque
from urllib.parse import urljoin, urlparse, urldefrag

import requests
from bs4 import BeautifulSoup

INDEX = "https://www.csulb.edu/university-policies"
UA = "Mozilla/5.0 (compatible; csulb-policy-study/0.1; personal research)"

DOC_EXT = {".pdf", ".docx", ".doc", ".rtf"}
OFFICE_EXT = {".docx", ".doc", ".rtf"}
# other files linked from policy pages that are not policy text
IGNORE_EXT = {".jpg", ".jpeg", ".png", ".gif", ".svg", ".mp4", ".mp3", ".zip",
              ".xlsx", ".xls", ".pptx", ".ppt", ".csv", ".ics"}
LOGIN_HOSTS = {"login.microsoftonline.com", "csulb.sharepoint.com",
               "sso.csulb.edu", "login.csulb.edu"}
NEVER_HOSTS = {"www.youtube.com", "youtube.com", "www.facebook.com", "twitter.com",
               "instagram.com", "www.instagram.com", "www.linkedin.com", "x.com"}
# footer links that are policies in their own right
FOOTER_POLICIES = ("Notice of Nondiscrimination", "Enforcement Actions on Campus",
                   "Privacy Notice", "Accessibility Statement",
                   "Annual Security Report", "Campus Safety Plan", "Title IX",
                   "Free Speech")
# a listed page with at least this much text already is the policy
TEXT_ENOUGH = 2000
MAX_BYTES = 60 << 20
# a section page linking more sub-pages than this is linking the wider site
MAX_CHILDREN = 250


def main_of(soup):
    for sel in ("main", "#main-content", "article", "#content", ".item-page",
                "[role=main]"):
        m = soup.select_one(sel)
        if m is not None:
            return m
    return soup.body or soup


def main_text_len(soup):
    m = main_of(soup)
    for t in m(["script", "style", "nav", "noscript"]):
        t.decompose()
    return len(m.get_text(" ", strip=True))


def ext_of(url):
    name = urlparse(url).path.rsplit("/", 1)[-1].lower()
    return os.path.splitext(name)[1]


def slug_for(url, ext):
    p = urlparse(url)
    base = re.sub(r"[^A-Za-z0-9]+", "_", (p.netloc + p.path).strip("/"))[-80:]
    base = base.strip("_") or "index"
    if ext and base.lower().endswith(ext.strip(".")):
        base = base[: -len(ext.strip("."))].rstrip("_")
    h = hashlib.sha1(url.encode()).hexdigest()[:8]
    return f"{base}_{h}{ext}"


class Polite:
    """robots.txt and the per-host gap between requests."""

    def __init__(self, delay, session):
        self.delay, self.s = delay, session
        self.robots, self.next_ok = {}, {}

    def _robots(self, host, scheme):
        if host not in self.robots:
            rp = urllib.robotparser.RobotFileParser()
            try:
                r = self.s.get(f"{scheme}://{host}/robots.txt", timeout=20)
                body = r.text if r.status_code == 200 else ""
                # a bot-check page served in place of robots.txt is no rules
                if "<html" in body[:500].lower():
                    body = ""
                rp.parse(body.splitlines())
            except requests.RequestException:
                rp.parse([])
            self.robots[host] = rp
        return self.robots[host]

    def allowed(self, url):
        p = urlparse(url)
        return self._robots(p.netloc, p.scheme).can_fetch(UA, url)

    def gap(self, host):
        p = self.robots.get(host)
        cd = p.crawl_delay(UA) if p else None
        return max(self.delay, float(cd or 0))

    def ready_at(self, host):
        return self.next_ok.get(host, 0.0)

    def mark(self, host):
        self.next_ok[host] = time.time() + self.gap(host)


class Crawler:
    def __init__(self, out, delay, max_depth, verbose=True):
        self.out = out
        self.max_depth = max_depth
        self.verbose = verbose
        self.s = requests.Session()
        self.s.headers["User-Agent"] = UA
        self.polite = Polite(delay, self.s)
        self.queues = {}            # host -> deque of jobs
        self.seen = set()
        self.records = {}           # url -> manifest record
        self.deferred = []          # office files waiting on their companion page
        self.page_text = {}         # url -> length of the page's main text
        self.children = {}          # page url -> how many child pages it queued
        self.mpath = os.path.join(out, "manifest.jsonl")
        self.prev = {}
        # .new is what an interrupted run had written so far
        for path in (self.mpath, self.mpath + ".new"):
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        try:
                            r = json.loads(line)
                        except ValueError:
                            continue
                        self.prev[r["url"]] = r
        if self.prev:
            with open(self.mpath, "w", encoding="utf-8") as f:
                for r in self.prev.values():
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        if os.path.exists(self.mpath + ".new"):
            os.remove(self.mpath + ".new")
        for d in ("raw/html", "raw/pdf", "raw/office"):
            os.makedirs(os.path.join(out, d), exist_ok=True)

    # ------------------------------------------------------------------ queue
    def push(self, url, depth, parent, section, kind, title=None):
        url = urldefrag(url)[0]
        if not url.startswith(("http://", "https://")) or url in self.seen:
            return
        if kind == "page" and parent:
            n = self.children.get(parent, 0)
            if n >= MAX_CHILDREN:
                return
            self.children[parent] = n + 1
        self.seen.add(url)
        host = urlparse(url).netloc.lower()
        self.queues.setdefault(host, deque()).append(
            dict(url=url, depth=depth, parent=parent, section=section, kind=kind,
                 title=title))

    def next_job(self):
        live = [h for h, q in self.queues.items() if q]
        if not live:
            return None
        host = min(live, key=self.polite.ready_at)
        return self.queues[host].popleft()

    # ------------------------------------------------------------------ io
    def log(self, *a):
        if self.verbose:
            print(*a, flush=True)

    def record(self, job, status, **kw):
        rec = dict(url=job["url"], parent=job["parent"], section=job["section"],
                   depth=job["depth"], kind=job["kind"], status=status, **kw)
        self.records[job["url"]] = rec
        with open(self.mpath + ".new", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    def cached(self, url):
        r = self.prev.get(url)
        if r and r.get("status") == "ok" and r.get("file") and \
                os.path.exists(os.path.join(self.out, r["file"])):
            return r
        return None

    def fetch(self, url):
        host = urlparse(url).netloc.lower()
        last = None
        for attempt in range(2):
            wait = self.polite.ready_at(host) - time.time()
            if wait > 0:
                time.sleep(wait)
            try:
                r = self.s.get(url, timeout=60, allow_redirects=True, stream=True)
                body = r.raw.read(MAX_BYTES + 1, decode_content=True)
                r._content = body
                self.polite.mark(host)
                if r.status_code >= 500 and attempt == 0:
                    last = r
                    continue
                return r, None
            except requests.RequestException as e:
                self.polite.mark(host)
                last = e
        if isinstance(last, requests.Response):
            return last, None
        return None, f"network error: {last}"

    # ------------------------------------------------------------------ crawl
    def run(self):
        self.push(INDEX, 0, None, "University Policies", "page")
        n = 0
        while True:
            job = self.next_job()
            if job is None:
                if not self.deferred:
                    break
                self.release_deferred()
                continue
            n += 1
            self.handle(job)
            if n % 25 == 0:
                left = sum(len(q) for q in self.queues.values())
                self.log(f"  ... {n} handled, {left} queued, {len(self.deferred)} deferred")
        os.replace(self.mpath + ".new", self.mpath)
        self.write_skipped()

    def handle(self, job):
        url = job["url"]
        host = urlparse(url).netloc.lower()
        if host in NEVER_HOSTS:
            return self.record(job, "skipped", reason="not a policy site (social/video)")
        if host in LOGIN_HOSTS:
            return self.record(job, "skipped", reason="requires CSULB / Microsoft login")
        if not self.polite.allowed(url):
            return self.record(job, "skipped", reason="robots.txt disallows crawling")

        c = self.cached(url)
        if c:
            path = os.path.join(self.out, c["file"])
            if job["kind"] == "page":
                with open(path, encoding="utf-8", errors="replace") as f:
                    self.parse_page(job, f.read(), c.get("final_url", url))
            rec = dict(c)
            rec.update(parent=job["parent"], section=job["section"], depth=job["depth"])
            self.records[url] = rec
            with open(self.mpath + ".new", "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            return rec

        r, err = self.fetch(url)
        if err:
            self.log(f"  x {url}  {err}")
            return self.record(job, "skipped", reason=err)
        final = r.url
        fhost = urlparse(final).netloc.lower()
        if fhost in LOGIN_HOSTS or "/login" in urlparse(final).path.lower():
            return self.record(job, "skipped", reason="redirects to a login page",
                               final_url=final)
        if r.status_code in (401, 403):
            self.log(f"  x {url}  {r.status_code}")
            return self.record(job, "skipped", reason=f"HTTP {r.status_code} (access blocked)",
                               http=r.status_code, final_url=final)
        if r.status_code == 202 and not (r.content or b"").strip():
            return self.record(job, "skipped", reason="empty response (bot protection)",
                               http=r.status_code, final_url=final)
        if r.status_code != 200:
            self.log(f"  x {url}  {r.status_code}")
            return self.record(job, "skipped", reason=f"HTTP {r.status_code}",
                               http=r.status_code, final_url=final)
        body = r.content or b""
        if not body.strip():
            return self.record(job, "skipped", reason="empty response (bot protection)",
                               http=r.status_code, final_url=final)
        if len(body) > MAX_BYTES:
            return self.record(job, "skipped", reason="file larger than 60 MB")

        ctype = r.headers.get("content-type", "").lower()
        is_pdf = body[:5] == b"%PDF-"
        is_html = "html" in ctype and not is_pdf
        sha = hashlib.sha256(body).hexdigest()

        if job["kind"] == "doc":
            ext = ext_of(final) or ext_of(url)
            if ext == ".pdf" and not is_pdf:
                reason = ("served a web page instead of the PDF" if is_html
                          else "not a valid PDF")
                return self.record(job, "skipped", reason=reason, final_url=final)
            sub = "raw/pdf" if is_pdf else "raw/office"
            rel = os.path.join(sub, slug_for(url, ".pdf" if is_pdf else ext))
            with open(os.path.join(self.out, rel), "wb") as f:
                f.write(body)
            self.log(f"  + {rel}  ({len(body) // 1024} KB)")
            return self.record(job, "ok", type="pdf" if is_pdf else ext.strip("."),
                               file=rel, sha256=sha, bytes=len(body), final_url=final,
                               title=job.get("title"))

        if is_pdf:      # a "page" link that turned out to be a PDF
            job["kind"] = "doc"
            rel = os.path.join("raw/pdf", slug_for(url, ".pdf"))
            with open(os.path.join(self.out, rel), "wb") as f:
                f.write(body)
            self.log(f"  + {rel}  ({len(body) // 1024} KB)")
            return self.record(job, "ok", type="pdf", file=rel, sha256=sha,
                               bytes=len(body), final_url=final)
        if not is_html:
            return self.record(job, "skipped", reason=f"unsupported content type {ctype}",
                               final_url=final)
        html = body.decode(r.encoding or "utf-8", errors="replace")
        low = html[:3000].lower()
        if "human check" in low or "just a moment" in low or "captcha" in low:
            return self.record(job, "skipped", reason="bot check page (human verification)",
                               final_url=final)
        rel = os.path.join("raw/html", slug_for(url, ".html"))
        with open(os.path.join(self.out, rel), "w", encoding="utf-8") as f:
            f.write(html)
        title, tlen = self.parse_page(job, html, final)
        self.log(f"  + {rel}  [{tlen} chars] {title[:60]}")
        return self.record(job, "ok", type="html", file=rel, sha256=sha,
                           bytes=len(body), final_url=final, title=title, text_chars=tlen)

    # ------------------------------------------------------------------ links
    def parse_page(self, job, html, final):
        soup = BeautifulSoup(html, "html.parser")
        title = (soup.title.get_text(" ", strip=True) if soup.title else "").split("|")[0].strip()
        depth = job["depth"]
        if depth == 0:
            self.seed_from_index(soup, final)
        elif depth < self.max_depth:
            self.links_from(soup, final, job, pages=True)
        else:
            self.links_from(soup, final, job, pages=False)
        tlen = main_text_len(BeautifulSoup(html, "html.parser"))
        self.page_text[job["url"]] = tlen
        self.page_text[urldefrag(final)[0]] = tlen
        return title, tlen

    def seed_from_index(self, soup, base):
        seeds = []
        q = soup.select_one("nav.field--name-field-quicklinks")
        if q:
            seeds += q.find_all("a", href=True)
        seeds += main_of(soup).find_all("a", href=True)
        for a in soup.select("div.menu--footer a[href]"):
            if any(a.get_text(" ", strip=True).startswith(t) for t in FOOTER_POLICIES):
                seeds.append(a)
        for a in seeds:
            href = urljoin(base, a["href"])
            text = a.get_text(" ", strip=True) or href
            kind = "doc" if ext_of(href) in DOC_EXT else "page"
            self.push(href, 1, base, text, kind)

    def page_allowed(self, href, base):
        p, b = urlparse(href), urlparse(base)
        if p.query or ext_of(href) in IGNORE_EXT:
            return False
        if p.netloc.lower() != b.netloc.lower():
            return False
        path = p.path.rstrip("/")
        if any(seg in path for seg in ("/user", "/search", "/node/add", "/admin")):
            return False
        if b.netloc.endswith("csulb.edu") and b.netloc.startswith("www."):
            if path.startswith(("/university-policies", "/node/")):
                return True
            first = "/" + b.path.strip("/").split("/")[0]
            return path.startswith(first + "/") and path != b.path.rstrip("/")
        return path.startswith(b.path.rstrip("/") + "/")

    def links_from(self, soup, base, job, pages):
        m = main_of(soup)
        for a in m.find_all("a", href=True):
            href = urldefrag(urljoin(base, a["href"].strip()))[0]
            if not href.startswith(("http://", "https://")):
                continue
            e = ext_of(href)
            text = a.get_text(" ", strip=True)
            if e in IGNORE_EXT:
                continue
            if e == ".pdf":
                self.push(href, job["depth"] + 1, job["url"], job["section"], "doc",
                          title=text)
            elif e in OFFICE_EXT:
                # on a listing the policy's own page is the link just before its
                # "DOC" link; on a policy page it is the page itself
                companion = job["url"]
                if pages:
                    prev = a.find_previous("a", href=True)
                    if prev is not None:
                        h = urldefrag(urljoin(base, prev["href"]))[0]
                        if ext_of(h) not in DOC_EXT and h.startswith("http"):
                            companion = h
                self.deferred.append(dict(url=href, depth=job["depth"] + 1,
                                          parent=job["url"], section=job["section"],
                                          companion=companion, title=text))
            elif pages and self.page_allowed(href, base):
                self.push(href, job["depth"] + 1, job["url"], job["section"], "page")

    def release_deferred(self):
        """Office files whose companion page turned out short (or never loaded)."""
        pending, self.deferred = self.deferred, []
        for d in pending:
            url = urldefrag(d["url"])[0]
            if url in self.seen:
                continue
            have = self.page_text.get(d["companion"], 0)
            if have >= TEXT_ENOUGH:
                self.seen.add(url)
                job = dict(url=url, depth=d["depth"], parent=d["parent"],
                           section=d["section"], kind="doc")
                self.record(job, "skipped",
                            reason="duplicate: the policy page already has this text",
                            companion=d["companion"])
                continue
            self.push(url, d["depth"], d["parent"], d["section"], "doc", title=d["title"])

    def write_skipped(self):
        rows = [r for r in self.records.values() if r["status"] != "ok"]
        ok = [r for r in self.records.values() if r["status"] == "ok"]
        by = {}
        for r in rows:
            by.setdefault(r["reason"], []).append(r)
        lines = ["# Skipped while scraping CSULB policies", "",
                 f"Fetched: {len(ok)} files "
                 f"({sum(r.get('type') == 'html' for r in ok)} pages, "
                 f"{sum(r.get('type') == 'pdf' for r in ok)} PDFs, "
                 f"{sum(r.get('type') in ('doc', 'docx', 'rtf') for r in ok)} Word/RTF). "
                 f"Skipped: {len(rows)}.", ""]
        order = sorted(by, key=lambda k: (k.startswith("duplicate"), -len(by[k])))
        for reason in order:
            rs = by[reason]
            lines += [f"## {reason} ({len(rs)})", ""]
            if reason.startswith("duplicate"):
                lines += ["(Word/RTF copies of policies whose web page already holds "
                          "the full text; not needed.)", ""]
                continue
            for r in sorted(rs, key=lambda r: r["url"]):
                lines.append(f"- [{r['section']}] {r['url']}")
            lines.append("")
        with open(os.path.join(self.out, "skipped.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--out", default="school_policies_data")
    ap.add_argument("--delay", type=float, default=10.0,
                    help="seconds between requests to one host (robots Crawl-delay "
                         "wins if larger)")
    ap.add_argument("--max-depth", type=int, default=2)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    c = Crawler(a.out, a.delay, a.max_depth, verbose=not a.quiet)
    c.run()
    ok = sum(r["status"] == "ok" for r in c.records.values())
    print(f"done in {(time.time() - t0) / 60:.1f} min: {ok} saved, "
          f"{len(c.records) - ok} skipped -> {a.out}/manifest.jsonl, {a.out}/skipped.md")


if __name__ == "__main__":
    sys.exit(main())
