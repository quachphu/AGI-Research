# Evaluation protocol

Written before any measurement in this evaluation. The question files committed with this document are the only questions that will be scored. The code this protocol was written against is `d6aa86915ee73d650605683d237eaddc0d2872d3`. The measurement tools gain `--held-out`, `--out`, and `--force` in the following commit. Every result file records `git rev-parse HEAD` at the time it is produced. That commit is the measurement code. The API key stays in an untracked `.env` and is not committed.

## Claim

The claim under test is that continued reading put California State University, Long Beach policy text into the weights, and that the author’s frozen benchmarks show what else those weights kept or lost.

Closed-book is the headline. The model sees the question and nothing else. The gold passage is a reading check: the passage is the stored `passage` field the question was built from, not a retrieved passage. Retrieval is not part of this evaluation. `tools/csulb/eval.py` `open_prompt` is not used, because it calls BM25.

## Checkpoints

| Folder | Step | What it is |
|---|---|---|
| `weights/` | 914,508 | The author’s weights, before any CSULB reading |
| `weights_csulb_65min` | 923,915 | Labeled point only. Ten subjects, learning rate `1e-4`, plasticity reset. Not a point on the hour 2–8 line |
| `weights_csulb_h2` | 950,140 | Hour 2 of the eight-hour run |
| `weights_csulb_h4` | 965,184 | Hour 4 |
| `weights_csulb_h6` | 982,163 | Hour 6 |
| `weights_csulb` | 995,953 | Hour 8, the weights the chatbot loads |

Question answers are generated for `weights/` and `weights_csulb` only. Hours 2, 4, and 6 are not generated, and there is no hour-6 versus hour-8 comparison on the questions.

Forgetting benchmarks use all six folders for held-out loss and enwik9. PG19 uses the author’s weights, the 65-minute checkpoint, and hour 8. The author’s weights and hour 8 are measured twice on every benchmark. A gap under 0.03 nats, about 0.04 bits per byte, is noise. Hours 2 through 8 are one line. The 65-minute point is labeled separately because it was a different training condition.

Step 931,303 and the 70.7-minute checkpoint were overwritten and are not measured. Every checkpoint is checked by manifest step and `core.npz` sha256 against `RESEARCH.md` before it is measured. `config.yaml` is not edited. `data.held_out` stays `data/val`.

## Questions

`tools/csulb/draw_eval_sample.py` draws the files below. Seed `20261029`. A question is clean when its wording does not appear in `school_policies_data/train/csulb_qa`, using the same scan as `questions_in_train` in `tools/csulb/watch_8h.py`. That leaves 2,514 of 3,603. The draw takes 50 of each kind. Within one kind the script keeps at most two questions from one document, and that count starts over at the next kind. In the saved 400, fifty documents appear more than twice and one document appears six times. The sample was not redrawn. Kinds, in draw order: topic, overview, where, date, number, definition, approval, supersedes.

- Confirmatory set: `research/eval/questions_400.jsonl` (400 questions).
- Prompt development: `research/eval/questions_prompt_dev.jsonl` (20 clean questions drawn from what remained after the 400). These 20 are not in the confirmatory scores. One format adjustment is allowed on them. After that adjustment the prompt is frozen and committed.

Eleven of the 400 also appear in the original 60-question pilot. One of the 20 prompt-development questions does too. That intersection is listed in `research/eval/sample_draw.json`. It is recorded. The sample is not redrawn to remove it. The original 60 stay a separate pilot.

Each row stores the official reference in `answer`, the policy number, the document date, the URL, and `gold_user`, which is `open_book_prompt` applied to the stored passage and the document record. Generation and the Claude gold-passage request use that stored string, so the passage bytes are the ones frozen here.

## Prompts

Local closed-book. The question only, through `serve.build_prompt`, the same path as `eval.ask`. The user turn is the question string.

Local gold passage. The user turn is the stored `gold_user` string. It is not rebuilt at generation time.

Claude, no passage:

```text
Answer this question from your own knowledge in 2 to 5 sentences, then stop. Do not use a heading or a bullet list. Mention a policy number or a source only if you know it.

Question: {question}
```

Claude, gold passage:

```text
Answer this question only from the passage below, in 2 to 5 sentences, then stop. Do not use a heading or a bullet list. Mention a policy number or a source only if the passage states it.

{gold_user}
```

`{gold_user}` is the same string the local model reads. Claude never receives a local answer or a reference answer. One question per request. No conversation history.

The 20 prompt-development answers were longer than 5 sentences: closed-book ran to 8 sentences and the gold passage to 12. That was the one allowed format change. The sentences above are the frozen prompt. Those 20 questions are not in the confirmatory scores.

The Claude model id is `claude-opus-5-5`. Effort is `low`. Requests go through the Batch API. No temperature setting is sent. The output limit is high enough for the thinking trace and the answer. If that model id is rejected, this protocol is amended before any score is computed. Another model is not substituted silently.

## Generation

Greedy decoding through `serve.stream`, which is temperature 0 and the same call `eval.ask` makes, including that function’s repetition guard. Learning is off. The limit is 800 yields from `serve.stream` for every local answer in this evaluation. A yield is usually one character.

`serve.stream` stops when the reply completes `</bot>`, and it does not yield the last character of that marker. The saved answer is the yielded text, cut at the first of `</bot>`, `</bo`, `<user>`, or `<bot>`. The record says `end_marker` when the stream stopped before 800 yields, and `limit` when it reached 800 yields. Of the 360 limit stops, 291 saved answers are 800 characters and the other 69 are between 799 and 807. A new turn inside the yielded text is cut the same way and counts as `end_marker`.

## What is scored

Primary metric: cosine similarity of the answer to the official reference answer.

Primary embedding: `Qwen/Qwen3-Embedding-0.6B`. Secondary: `sentence-transformers/all-mpnet-base-v2`. Both texts are encoded the same way, with no instruction prefix. Vectors are L2-normalized. Cosine is their dot product. A ranking of checkpoints is reported only when the two embeddings agree on the order.

Before the embedding, strip a leading `According to …`, any `Source: …` line, and chat tags (`<user>`, `</user>`, `<bot>`, `</bot>`). Then cut the local answer, the Claude answer, and the reference for that question to the length of that question’s cleaned reference. This cut is per question. It is not a fixed character cap.

Exact match and word F1 use the full raw answer, after the end-of-reply cut and before that stripping. Exact match covers the policy number (the stored number without a leading `PS `, the same rule as `eval.py`), the document date when `kind` is `date`, and the full URL. A missing policy number, date, or URL on the question is not a miss.

Secondary metric: cosine of the local answer to Claude’s answer on the same question, labeled agreement. It is not the headline.

Chance: cosine between an answer and a different question’s reference, ten shuffles of the references.

Ceiling: cosine between Claude’s two answers on a 50-question re-run. The 50 are the first 50 rows of `questions_400.jsonl` in file order.

High cosine, for the error panel, is the top quarter of cosine-to-reference values for that system and that embedding, on its valid answers. The quarter is computed before the panel is read. The panel lists answers in that quarter whose policy number or date does not match.

A Claude answer is not valid if the request was refused or if it stopped because it hit the length limit. Those counts are reported. Paired comparisons use only questions where both systems have a valid answer. Bootstrap 95% intervals are computed on those paired questions.

## Frozen benchmarks

These are the author’s next-byte measurements, re-run with the measurement code. Numbers already in the training logs are the historical record. They are not the table, because held-out reading changed after those runs.

Every invocation passes `--force`, including the first measurement of `weights/`. `tools/measure_heldout.py` can otherwise reuse the author’s logged line at step 914,508. A reused log line fails this protocol. The result record says the number was computed on this run.

- Held-out on `data/val`, and held-out on `school_policies_data/heldout_mix`: all six folders.
- enwik9, frozen, 2,000 pieces of 2,048 bytes: all six folders.
- PG19, frozen, all 100 books, window 4,096: author’s weights, 65-minute checkpoint, hour 8.

The author’s weights and hour 8 are then measured a second time on each of those. Every subject the tool returns is kept. The tools overwrite shared files under `runs/`. Each record is copied into `research/eval/benchmarks/` immediately. `tools/refresh_readme.py` and `measure_benchmarks.sh` are not run.

Only output paths and the force flag were added to the author’s measurement tools. The measurement arithmetic is unchanged. With those flags omitted, each tool behaves as it did at `d6aa869`.

## Order

Frozen benchmarks and local answers both use the GPU, so local answers start after the benchmarks have released it. Claude does not use that GPU. Scoring starts after both answer sets exist.

## What this evaluation does not include

No NLI model. No hand grading. No unanswerable questions. No learning-while-reading measurement. No MMLU-style exam. No BM25. The earlier 60-question open-book scores used BM25 and are an earlier pilot. The earlier closed-book word F1 on those 60 used a 360-character limit. Of the 60 scored hour-8 closed-book answers, 31 are exactly 360 characters and 6 are 359. The other 23 stopped earlier. Eleven of the 60 contain the full URL. Those numbers are reported separately and are not recomputed here.

## Limits stated in advance

The reference answer was a training target on the other wordings of the same fact. Only the question wording is new. A high cosine means a new wording produced the trained answer. It does not mean the answer text was never trained.

Training started from the author’s checkpoint, not from empty weights. The 65-minute run reset the plasticity scale and read stories and arithmetic, which the eight-hour run did not. Claude’s answers vary from run to run, and Claude may have seen public CSULB pages. The model was not trained to say that it does not know. Cosine similarity is weak when two sentences differ by a number, a date, or a negation, which is why exact match and the error panel are reported beside it.
