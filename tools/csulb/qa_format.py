"""
The one place the open-book question is spelled, shared by training and serving.

The model only answers from a passage the way it was trained to, so the text
around the passage must be byte-for-byte the same in build_corpus.py, eval.py
and chat.py.
"""
U0, U1, B0, B1 = "<user>", "</user>", "<bot>", "</bot>"

PASSAGE_MAX = 1500


def open_book_prompt(question, doc, passage, heading=None):
    """
    `doc` is a documents.jsonl record with `ctitle` (or `title`); its number,
    date and address go in a short header so an answer can cite them by
    copying rather than from memory.
    """
    title = doc.get("ctitle") or doc.get("title") or ""
    head = [title + (f" ({doc['policy_number']})" if doc.get("policy_number") else "")]
    if heading:
        head.append(f"Section: {heading}")
    if doc.get("date"):
        head.append(f"Date: {doc['date']}")
    head.append(f"Source: {doc.get('url', '')}")
    p = " ".join(passage.replace("## ", "").split())[:PASSAGE_MAX]
    return ("[Policy passage]\n" + "\n".join(head) + f"\n{p}\n\n"
            f"[Question]\n{question}")
