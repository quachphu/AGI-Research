"""
The CSULB policy chatbot: mini-AGI's answer beside the official passage.

    .venv/bin/python tools/csulb/chat.py --weights weights_csulb --port 8765
    then open http://127.0.0.1:8765

Every question is first looked up in the policy sections with BM25 (word
matching, no model). The three best passages are shown on the right with
their links, so whatever the model writes can be checked against the real
text. With "open-book" on, the best passage is also put in front of the
question, in the format the model was trained on (train/csulb_qa_open);
with it off, the model answers from memory alone.

Read-only: the weights are never changed by chatting. Each question is
answered on its own - the model was trained on single questions, and an
earlier turn's passage would only crowd the context.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import serve  # noqa: E402
from flask import Flask, Response, jsonify, request  # noqa: E402

from bm25 import BM25  # noqa: E402
from build_corpus import clean_heading, office_of  # noqa: E402
from eval import load_docs  # noqa: E402
from qa_format import open_book_prompt  # noqa: E402

app = Flask(__name__)
IDX = {"bm25": None, "docs": None}


def sources(q, k=3):
    out = []
    for s in IDX["bm25"].search(q, k=k):
        d = IDX["docs"].get(s["doc"], {})
        out.append(dict(doc=s["doc"], title=d.get("ctitle") or s["title"],
                        number=s.get("policy_number"), heading=clean_heading(s.get("heading")),
                        office=office_of(d) if d else None, date=d.get("date"),
                        url=s["url"], text=s["text"], score=s["score"]))
    return out


@app.route("/api/sources")
def api_sources():
    return jsonify(sources(request.args.get("q", "")))


@app.route("/api/ask", methods=["POST"])
def api_ask():
    body = request.get_json(force=True)
    q = (body.get("question") or "").strip()
    open_book = bool(body.get("open_book", True))
    max_new = min(int(body.get("max_new", 500)), 1200)
    src = sources(q) if q else []
    text = q
    if open_book and src:
        top = src[0]
        text = open_book_prompt(q, IDX["docs"][top["doc"]], top["text"], top["heading"])
    model = serve.STATE["model"]
    prompt = serve.build_prompt([{"role": "user", "content": text}], int(model.cfg.block * 0.9))

    def events():
        yield "data: " + json.dumps({"sources": src, "open_book": open_book,
                                     "prompt_chars": len(prompt)}) + "\n\n"
        t0, n = time.time(), 0
        try:
            with serve.LOCK:
                for ev in serve.stream(prompt, max_new):
                    if "t" in ev:
                        n += 1
                        yield "data: " + json.dumps({"t": ev["t"]}) + "\n\n"
        except Exception as e:                                # noqa: BLE001
            yield "data: " + json.dumps({"error": str(e)}) + "\n\n"
        dt = max(time.time() - t0, 1e-6)
        yield "data: " + json.dumps({"done": True, "n": n, "cps": round(n / dt, 1)}) + "\n\n"

    return Response(events(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/info")
def api_info():
    return jsonify(weights=serve.STATE.get("weights"), docs=len(IDX["docs"] or {}),
                   sections=len(IDX["bm25"].sections) if IDX["bm25"] else 0)


@app.route("/")
def index():
    return Response(PAGE, mimetype="text/html")


PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>CSULB Policy Chatbot (mini-AGI)</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root{--gold:#f2a900;--ink:#1b1b1b;--muted:#666;--line:#e3e3e3;--bg:#fafafa}
  *{box-sizing:border-box} html,body{height:100%;margin:0}
  body{font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;color:var(--ink);background:var(--bg);display:flex;flex-direction:column}
  header{background:#000;color:#fff;padding:10px 18px;display:flex;align-items:center;gap:14px}
  header b{color:var(--gold);font-size:17px} header span{color:#bbb;font-size:13px}
  main{flex:1;display:grid;grid-template-columns:minmax(0,1.15fr) minmax(0,1fr);min-height:0}
  #chat{display:flex;flex-direction:column;border-right:1px solid var(--line);min-height:0}
  #log{flex:1;overflow:auto;padding:18px}
  .msg{max-width:92%;margin:0 0 14px;padding:10px 13px;border-radius:10px;white-space:pre-wrap;word-wrap:break-word}
  .user{background:#000;color:#fff;margin-left:auto}
  .bot{background:#fff;border:1px solid var(--line)}
  .bot .meta{display:block;color:var(--muted);font-size:12px;margin-top:6px}
  .bot a{color:#0b5cad}
  form{display:flex;gap:8px;padding:12px;border-top:1px solid var(--line);background:#fff;flex-wrap:wrap}
  textarea{flex:1;min-width:240px;resize:none;height:54px;padding:9px;font:inherit;border:1px solid #ccc;border-radius:8px}
  button{background:var(--gold);border:0;border-radius:8px;padding:0 18px;font-weight:600;cursor:pointer}
  button:disabled{opacity:.5;cursor:default}
  .opts{width:100%;display:flex;gap:16px;color:var(--muted);font-size:13px;align-items:center}
  #side{overflow:auto;padding:18px;background:#fff}
  #side h2{font-size:14px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);margin:0 0 10px}
  .src{border:1px solid var(--line);border-left:4px solid var(--gold);border-radius:8px;padding:10px 12px;margin-bottom:12px}
  .src.first{border-left-color:#000}
  .src .t{font-weight:600} .src .h{color:var(--muted);font-size:13px}
  .src .x{white-space:pre-wrap;font-size:13.5px;margin-top:6px;max-height:220px;overflow:auto}
  .src a{font-size:12.5px;word-break:break-all;color:#0b5cad}
  .tag{display:inline-block;font-size:11px;background:#000;color:#fff;border-radius:4px;padding:0 6px;margin-left:6px}
  .warn{font-size:12.5px;color:#7a5600;background:#fff6dc;border:1px solid #f3dc9b;border-radius:8px;padding:8px 10px;margin-bottom:12px}
  .empty{color:var(--muted)}
  @media (max-width:900px){main{grid-template-columns:1fr}#chat{border-right:0;border-bottom:1px solid var(--line);min-height:60vh}}
</style></head><body>
<header><b>CSULB Policy Chatbot</b><span id="info">mini-AGI</span></header>
<main>
 <section id="chat">
  <div id="log"><div class="msg bot">Ask me about a CSULB policy - for example <i>"What counts as cheating?"</i>, <i>"What is the policy on withdrawals?"</i> or <i>"Where can I find PS 21-01?"</i>. The official passage I found is shown on the right.<span class="meta">Answers come from a small model and can be wrong. The passage and link on the right are the official text.</span></div></div>
  <form id="f">
   <textarea id="q" placeholder="Ask a question about CSULB policies..." autofocus></textarea>
   <button id="go">Ask</button>
   <div class="opts">
    <label><input type="checkbox" id="ob" checked> open-book: give the model the policy passage</label>
    <span id="speed"></span>
   </div>
  </form>
 </section>
 <aside id="side"><h2>Official sources</h2><div id="srcs"><p class="empty">The matching policy passages will appear here.</p></div></aside>
</main>
<script>
const log=document.getElementById('log'), srcs=document.getElementById('srcs');
const esc=s=>s.replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const linkify=s=>esc(s).replace(/(https?:\/\/[^\s<)]+)/g,'<a href="$1" target="_blank" rel="noopener">$1</a>');
fetch('/api/info').then(r=>r.json()).then(i=>{document.getElementById('info').textContent=
  `mini-AGI weights: ${i.weights} | ${i.docs} policy documents, ${i.sections} sections`});
function showSources(list){
  if(!list.length){srcs.innerHTML='<p class="empty">No matching policy text found.</p>';return}
  srcs.innerHTML=list.map((s,i)=>`<div class="src ${i?'':'first'}">
    <div class="t">${esc(s.title)}${s.number?` <span class="tag">${esc(s.number)}</span>`:''}${i?'':' <span class="tag">given to the model</span>'}</div>
    <div class="h">${esc([s.heading,s.office,s.date].filter(Boolean).join(' - '))}</div>
    <div class="x">${esc(s.text.replace(/^## /gm,''))}</div>
    <a href="${esc(s.url)}" target="_blank" rel="noopener">${esc(s.url)}</a></div>`).join('');
  if(!document.getElementById('ob').checked){srcs.querySelectorAll('.tag').forEach(t=>{if(t.textContent==='given to the model')t.remove()})}
}
document.getElementById('q').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();document.getElementById('f').requestSubmit()}});
document.getElementById('f').addEventListener('submit',async e=>{
  e.preventDefault();
  const q=document.getElementById('q').value.trim(); if(!q)return;
  const go=document.getElementById('go'); go.disabled=true; document.getElementById('q').value='';
  log.insertAdjacentHTML('beforeend',`<div class="msg user">${esc(q)}</div>`);
  const b=document.createElement('div'); b.className='msg bot'; b.textContent='...'; log.appendChild(b); log.scrollTop=1e9;
  let text='';
  try{
    const r=await fetch('/api/ask',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({question:q,open_book:document.getElementById('ob').checked})});
    const rd=r.body.getReader(), dec=new TextDecoder(); let buf='';
    for(;;){const {value,done}=await rd.read(); if(done)break; buf+=dec.decode(value,{stream:true});
      let i; while((i=buf.indexOf('\n\n'))>=0){const line=buf.slice(0,i); buf=buf.slice(i+2);
        if(!line.startsWith('data: '))continue; const ev=JSON.parse(line.slice(6));
        if(ev.sources)showSources(ev.sources);
        if(ev.t){text+=ev.t; b.innerHTML=linkify(text); log.scrollTop=1e9}
        if(ev.error){b.innerHTML=linkify(text)+`<span class="meta">error: ${esc(ev.error)}</span>`}
        if(ev.done){b.innerHTML=linkify(text||'(no answer)')+`<span class="meta">${ev.n} characters, ${ev.cps} per second - check it against the source on the right</span>`;
          document.getElementById('speed').textContent=`${ev.cps} chars/s`}
      }}
  }catch(err){b.textContent='request failed: '+err}
  go.disabled=false; document.getElementById('q').focus();
});
</script></body></html>'''


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--weights", default="weights_csulb")
    ap.add_argument("--data", default="school_policies_data")
    ap.add_argument("--device", default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    IDX["bm25"] = BM25.load(os.path.join(a.data, "sections.jsonl"))
    IDX["docs"] = load_docs(a.data)
    serve.load(a.weights, device=a.device, learn=False)
    print(f"[csulb] {len(IDX['docs'])} documents, {len(IDX['bm25'].sections)} sections; "
          f"open http://{a.host}:{a.port}", file=sys.stderr)
    app.run(host=a.host, port=a.port, threaded=True)


if __name__ == "__main__":
    main()
