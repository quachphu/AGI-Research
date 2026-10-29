# End-to-end record: CSULB policies on mini-AGI

This is the record of one experiment. It starts from the published mini-AGI checkpoint, builds a corpus of public CSULB policies, trains that same checkpoint on the corpus, and scores the answers. The model was not created from random weights. The author's checkpoint had already read 1.87 billion characters. Everything below is continued training on top of that checkpoint, plus the data and the scoring built for this project.

Only this model was trained. The questions and the reference answers were built by the project's own scripts from the scraped policies. No other language model wrote the training text. On 29 October 2026 a separate evaluation, recorded from section 19 on, also asked Claude Opus 5.5 the same questions and scored it the same way. Claude did not write or grade the training data. The API key for that call is not in the repository.

The code of the model is [volotat/mini-AGI](https://github.com/volotat/mini-AGI) by Alexey Borsky, MIT license, file `LICENSE`. The training described here ran on that code as it stood before the author's 20–24 October 2026 commits. After the 8-hour run finished, this copy was fast-forwarded to the author's commit `8f74c58` ("benchmarking scripts", 24 October 2026). The CSULB files were kept. A re-run on today's code needs `--lane-schedule equal` if it is meant to match this experiment, because the author's later default shares reading time by expected held-out drop instead of giving every subject an equal turn.

Machine: Ubuntu, Python 3.10, one NVIDIA GPU with 24,564 MiB (the run recorded 383 MiB already in use before the 8-hour launch, of 24,564 MiB). Weights directory for every CSULB training command: `weights_csulb`. The author's folder `weights/` was never the directory passed to the trainer.

## 1. What the starting checkpoint already was

The file `weights/manifest.json` at the start:

| | |
|---|---|
| Step | 914,508 |
| Characters already read | 1,872,164,864 |
| Experts | 397 |
| Plasticity scale | 0.0515 (about 1/20 of the base rate) |
| sha256 of `core.npz` | `913645b2b3942a04f301bc34935feb84a3972f21057a3ae4fb2f3d4692d38b71` |

The same snapshot is the one the author points to from the README, on Hugging Face as `Volotat/mini-AGI-undertrained`. It was copied to `weights_csulb` and that copy is what training wrote. `weights/` itself was not modified. Its manifest timestamp was checked again after the pilot and after the 8-hour run.

What the trainer printed about this checkpoint, and what stayed true for every later session:

- 397 experts on disk, 32 resident on the GPU. About 100.7M parameters in VRAM out of 1,248.9M total.
- Compute in bf16. Weights stay fp32. Adam moments are stored as bf16.
- Context is 4,096 characters. The gradient reaches back 24 rows.
- Each character ranks all 397 experts and asks for its top 8. A forward uses the 32 experts that the current text asks for most. Experts that are not on the card are paged in from disk.
- The trunk (about 8.1M parameters) learns at 0.1 times the pool's rate. The pool is about 100.9M parameters in VRAM, the rest of the experts live on disk.
- One subject is read for 32,768 characters (8 windows of 4,096), then the next subject. Over a long run each subject receives the same number of characters. This is the equal schedule this experiment used.
- The learning rate is a base rate times a plasticity scale. The scale is moved by held-out loss. It can ease by at most 0.4% at a check, and it does not go below 0.01 of the base rate. `--lr-reset` sets the scale back to 1.0. Leaving it off keeps whatever scale is already in the checkpoint.
- A sample is written every `--sample-every` minutes. The unit is minutes.
- SIGTERM and SIGINT mean: finish the current chunk, write the checkpoint, exit. The print line says `weights/` even when `--weights-dir` is `weights_csulb`.

Chat text, in training and at serving time, is byte for byte:

```
<user>
question
</user>
<bot>
answer
</bot>
```

## 2. Decisions that fixed the shape of the experiment

- Train and score with mini-AGI only.
- The chatbot is open-book: the official passage and its URL are shown, and the model is asked to answer from that passage. Closed-book answers (question only, no passage) are scored as well, so memory and reading can be told apart.
- If a page cannot be fetched, skip it and keep going. Login walls, robots rules, and missing files are listed, not worked around.
- Keep the author's `weights/` folder untouched.
- The first long run was a pilot of about two hours. The later run was eight hours of training with a checkpoint record every two hours, and no early stop.

## 3. General text that shared the training time

mini-AGI reads several subjects and gives each subject an equal share of characters. If CSULB text were the only subject, the model would spend the whole run repeating a few tens of megabytes. The other subjects are hard links under `school_policies_data/train_general/`, taken from the general corpus already on disk (`data/train`). Hard links share the bytes. They are not a second download.

| Subject | About |
|---|---|
| `chat` | 443 MB |
| `reasoning` | 144 MB |
| `chess` | 126 MB |
| `code` | 108 MB |
| `wikipedia` | 55 MB |

`stories` and `arithmetic` exist in the author's corpus as very small local files. They were included in the first 65 minutes and then removed, because those two lanes started repeating (section 10). They are still inside the held-out mix, so the held-out number always includes them even when training does not.

The held-out mix is `school_policies_data/heldout_mix/`. It is the author's `data/val` subjects (arithmetic, chat, chess, code, reasoning, stories, wikipedia) plus `csulb_qa` and `csulb_qa_open`. `train.py` scores every subject in that folder. The headline held-out loss is the average across those subjects, each counted once, so a drop in CSULB loss is diluted by stories, arithmetic, and Wikipedia.

## 4. Scrape

Command:

```bash
.venv/bin/python tools/csulb/scrape.py
```

Code: `tools/csulb/scrape.py`. Start URL: `https://www.csulb.edu/university-policies`.

What it does:

1. Read `robots.txt` for each host. A disallowed URL is not fetched. `www.csulb.edu` publishes Crawl-delay 10, and the crawler waits at least that long between requests to one host. Hosts are interleaved, so one slow host does not block the others.
2. Crawl three levels: the index, the section pages it links, and the policy pages those sections list. PDFs linked from those pages are kept.
3. Word and RTF files are kept only when the listing page does not already contain the policy as text (at least 2,000 characters). On the Academic Senate pages the office file is the same policy a third time.
4. Skip image, spreadsheet, slide, zip, and calendar links. Skip YouTube and social hosts. Skip Microsoft and CSULB login hosts.
5. Also follow a short list of footer policies (nondiscrimination, privacy, accessibility, Title IX, the annual security report, and a few others).
6. User-Agent: `csulb-policy-study/0.1`. A file already recorded in `manifest.jsonl` is read from disk on a re-run.
7. Cap a download at 60 MB. A section page with more than 250 child links is treated as a link to the wider site and is not expanded that far.

Result, from `school_policies_data/skipped.md`:

| | |
|---|---|
| Files fetched | 466 (188 HTML pages, 278 PDFs, 0 Word/RTF kept) |
| Skipped | 175 |

Why pages were skipped:

| Reason | Count |
|---|---|
| Duplicate office file; the web page already has the text | 158 |
| HTTP 404 | 5 |
| `robots.txt` disallows the host (PolicyStat) | 4 |
| HTTP 403 | 2 |
| SharePoint / Microsoft login | 2 |
| URL returned a web page instead of the PDF | 2 |
| Empty catalog response | 1 |
| YouTube link | 1 |

The two 403 pages were the LGBTQIA campus climate committee page and the Research Foundation policies page. The four PolicyStat URLs were refused because their robots file disallows crawling. The log is `runs/csulb_scrape.log` and `runs/csulb_scrape_part1.log`.

## 5. Extract

Command:

```bash
.venv/bin/python tools/csulb/extract.py
```

Code: `tools/csulb/extract.py`.

What it does:

1. HTML: keep the main content block (`main`, `#main-content`, `article`, or the body). Drop scripts, styles, and navigation.
2. PDF: `pdftotext -enc UTF-8` in reading order. The `-layout` mode was not used, because it pads lines with spaces the model would then try to copy. Carriage returns and hyphenation across line breaks are cleaned.
3. Word and RTF, if any had been kept, would go through LibreOffice. None were kept.
4. A document under 150 characters is dropped. Sections are cut on headings. A section is at least 120 characters and at most 1,500. Each section carries the document title, policy number, heading, URL, and office.
5. Near-duplicates: 6-word shingles. If two documents overlap more than 0.8 of the shorter one's shingles, and their lengths are within a factor of two, keep the HTML page (it has the public address), otherwise the longer copy.
6. Policy numbers are read from the text, the title, and the URL. Each kept document is written to `text/<slug>.txt` with a header: policy number, date, supersedes, source URL, office.

Result, counted from `documents.jsonl` and `sections.jsonl`:

| | |
|---|---|
| Document records from the scrape | 466 |
| Documents kept | 312 |
| Characters in the kept documents | 4,042,325 |
| Sections | 3,996 |

Outputs: `school_policies_data/text/`, `documents.jsonl`, `sections.jsonl`.

## 6. Build the questions

Command:

```bash
.venv/bin/python tools/csulb/build_corpus.py
```

Code: `tools/csulb/build_corpus.py`. Seed 0. Three training wordings per fact. One further wording is held out. The eval file is 60 of those held-out wordings.

A fact is one thing a policy states. The builder makes several questions for the same answer. Repeating one sentence teaches the model to recite. Several wordings of one fact are what it can learn. The answer is always the same shape, so the model can copy the citation:

```
According to <title or "CSULB Policy Statement NN-NN">, section "...": <excerpt>
Source: <url>
```

Fact kinds:

| Kind | What the question asks |
|---|---|
| `topic` | What a named section says |
| `definition` | A term the section defines ("X is defined as ...", "X means ...") |
| `overview` | What the whole policy is about, from its first real section |
| `number` | Which policy-statement number it is |
| `date` | The date on the policy |
| `supersedes` | Which older policy it replaces |
| `approval` | Who approved it |
| `where` | The office and the URL |

For each fact the questions are shuffled. Up to three go to training. One is held out for `val/` and, if it was one of the 60, for the scored test. The open-book copy wraps the question with the passage, in `tools/csulb/qa_format.py`. Training, evaluation, and the chatbot call that same function, so the bytes match.

The passage header is:

```
[Policy passage]
<title> (<policy number, if any>)
Section: <heading>
Date: <date>
Source: <url>
<passage, spaces collapsed, at most 1,500 characters>

[Question]
<question>
```

Counted from the files that command wrote:

| Lane | Role | Size |
|---|---|---|
| `train/csulb_policies` | The 312 cleaned documents | 4.04 MB |
| `train/csulb_qa` | Closed-book questions | 6.30 MB |
| `train/csulb_qa_open` | The same questions with the passage in front | 18.54 MB |
| `val/csulb_qa` | Held-out wording, no passage | 2.12 MB |
| `val/csulb_qa_open` | Held-out wording, with passage | 6.28 MB |

3,603 facts received a held-out wording (`eval/questions_all.jsonl`). The scored subset is 60 questions (`eval/questions.jsonl`), spread across kinds and with at most two questions from one document:

| Kind | Questions in the 60 |
|---|---|
| topic | 21 |
| definition | 12 |
| overview | 12 |
| number | 5 |
| where | 4 |
| date | 3 |
| supersedes | 2 |
| approval | 1 |

One extra question lives in `eval/my_questions.jsonl` ("What happens if I am caught cheating on an exam?"). It has no reference answer, so it is printed in the report and left out of the F1 average.

A limit of this split: holding out one wording does not hold out the question string. The same words can be generated for two different facts (a policy and a memo that share a number, for example) with two different answers. Of the 60 scored questions, 15 use a wording that also appears in `train/csulb_qa` paired with a different document. The tables below report all 60 and the other 45 separately. The 45 are called the clean set.

At about 4,800 characters per second across eight subjects, one subject receives about 600 characters per second. A first pass of the open-book lane (18.5 MB) is on the order of 8 to 9 hours. The policy documents (4.0 MB) are a couple of hours. The closed-book questions (6.3 MB) are about three hours. That is why an 8-hour run finishes about one pass of the open-book lane and several passes of the smaller lanes.

## 7. Retrieval, before any CSULB training

Command:

```bash
.venv/bin/python tools/csulb/bm25.py --check
```

Code: `tools/csulb/bm25.py`. This is word matching. It is not a model and it does not learn. Okapi BM25 with k1 = 1.2, b = 0.6. The section title is counted twice, the heading four times. A policy number in the question (`21-01` or `PS 21-01`) adds 10 to every section of that policy. Stop-words include "csulb", "policy", and "university", so those words do not dominate.

On all 3,603 held-out questions, measured again from the files in this folder:

| | Right section first | Right section in the top 3 | Right document first |
|---|---|---|---|
| All 3,603 | 49.4% | 74.0% | 86.0% |

The 60-question score used later asks only whether the right document was first, and that figure is 80% both before and after training. Training does not change it. The chatbot shows the top 3 passages to the person, and puts only the top 1 into the model's prompt. Open-book F1 is therefore retrieval and reading added together. A wrong first passage puts the right words out of the model's reach.

## 8. Score the author's weights before any CSULB training

Command:

```bash
.venv/bin/python tools/csulb/eval.py --weights weights --tag before
```

Code: `tools/csulb/eval.py`. Learning is off. Decoding is greedy, so the same weights give the same text. Each question is asked twice. Closed-book is the question alone. Open-book is the top BM25 passage in the training format. New characters are capped at 360.

Four scores, each 0 to 100%:

| Score | Rule |
|---|---|
| Word F1 | Shared content words between the reply and the reference answer |
| Policy number | The number without the `PS ` prefix occurs in the reply |
| URL | The full source URL occurs in the reply |
| Form | The reply starts with "According to" or contains `Source: http` |

Right-document-first is scored from the retrieval hit, not from the model's words. The report writer (`--report`) reads only the tags `before` and `after` and writes `runs/csulb_eval.md`. Later checkpoints were not written into that file. Their JSON is `runs/csulb_eval_h2.json` through `h8`.

The author's answers, before any CSULB training, are in the "before" column of section 10.

## 9. First training attempt, stopped

`weights/` was copied to `weights_csulb`. The first launch kept the author's plasticity scale, so the printed rate was `x0.052`. It then crashed on the first step:

```
FileNotFoundError: weights_csulb/experts/e00723.npz
```

Log: `runs/csulb_pilot_aborted_lowlr.log`. No checkpoint from this attempt was kept. The folder was copied again from `weights/`, and the next launch passed `--lr-reset`.

## 10. Pilot, part 1: 65 minutes, ten subjects, learning rate 1e-4

Fresh copy of the author's weights. `--lr-reset` so the scale starts at 1.0. Base rate `1e-4`. Ten subjects: arithmetic, chat, chess, code, csulb_policies, csulb_qa, csulb_qa_open, reasoning, stories, wikipedia.

The log header (`runs/csulb_pilot_part1.log`):

- Optimiser restored at step 914,509 (the author's history).
- 1,872.2M characters already read.
- Held-out at the start: 0.7070 ± 0.0144 on `school_policies_data/heldout_mix`.
- Line present: `plasticity RESET: full learning rate restored`, then `rate x1.000`.

It ran 65.6 minutes, read 19.27M characters (about 4,900 characters per second), and stopped with a checkpoint. Mean loss over the files 0.5681. Held-out after: 0.6945 ± 0.0183 (change −0.0125). Step 923,915. That folder was copied aside as `weights_csulb_65min` and was not trained further.

At 64 minutes the sample in `runs/csulb_samples.txt` shows why this mix was changed. Stories held-out loss had risen from 0.484 at the start to 1.169. Arithmetic had risen from 0.584 to 0.783. Those two lanes are tiny, so an equal share means the model reads them over and over. CSULB itself was moving the right way in that same sample: `csulb_qa` 0.412 and `csulb_qa_open` 0.436, down from 0.822 and 0.765 at the start. The sample's learning-rate line is `lr 1.00e-04` and `rate x1.000`.

## 11. Pilot, part 2: 55 minutes, eight subjects, learning rate 5e-5

Continued from step 923,915. `--lr-reset` again, so the scale is 1.0 even though part 1 had just finished. Base rate lowered to `5e-5`. Stories and arithmetic removed from training. Eight subjects: chat, chess, code, csulb_policies, csulb_qa, csulb_qa_open, reasoning, wikipedia.

Header of `runs/csulb_pilot_part2.log`:

- 9,992 files, 905.14M characters in the training roots.
- Optimiser restored at step 923,916.
- 1,891.4M characters already read.
- Held-out at this start: 0.6947 ± 0.0184.
- `plasticity RESET`, `rate x1.000`.

It reached the 55-minute stop. Read 15.13M characters in 55.1 minutes. Mean file loss 0.5535. Held-out after: 0.6292 ± 0.0188 (change −0.0656). Step 931,303. This is the checkpoint the pilot evaluation calls "after", and it is the checkpoint the 8-hour run starts from. The sample line shows `lr 5.00e-05`.

Together the two pilot parts are about two hours of training that actually updated the weights (65.6 + 55.1 minutes), after the crashed attempt.

## 12. Pilot scores: author weights against the two-hour checkpoint

`runs/csulb_eval.md`. Same 60 questions. "Before" is `weights` (the author). "After" is `weights_csulb` at the end of part 2. Before took 9.1 minutes to score, after took 10.8.

| Measure | Author weights | After the pilot |
|---|---|---|
| Closed-book word F1 | 13.4% | 31.6% |
| Closed-book policy number | 3.0% | 18.2% |
| Closed-book URL | 0.0% | 8.3% |
| Closed-book answer form | 0.0% | 86.7% |
| Open-book word F1 | 17.7% | 51.6% |
| Open-book policy number | 15.2% | 78.8% |
| Open-book URL | 0.0% | 25.0% |
| Open-book answer form | 0.0% | 91.7% |
| Right policy document first | 80.0% | 80.0% |

The form score is the clearest change. After the pilot, most replies open with "According to" or carry a `Source:` line, which the author's checkpoint never did. The URL score stays low because the score requires the entire URL string. A shortened or rewritten link counts as a miss. Retrieval did not move, because it is BM25.

The full replies for 15 of the 60 questions, plus the cheating question, are in `runs/csulb_eval.md`. Two patterns show up there. When the first passage is the right section, the open-book reply often copies the policy's own sentence and the URL (the date question on Policy Statement 73-03 is word for word). When the first passage is a different policy that happens to share a word such as "exceptions" or "distribution", the model answers that other policy, fluently, in the trained form. Closed-book replies often name a real CSULB policy and a real URL, and just as often name the wrong one. Policy Statement 14-04 is a frequent wrong citation.

JSON: `runs/csulb_eval_before.json`, `runs/csulb_eval_after.json`.

## 13. The chatbot

Command:

```bash
.venv/bin/python -u tools/csulb/chat.py --weights weights_csulb --port 8765
```

Code: `tools/csulb/chat.py`. Page: `http://127.0.0.1:8765`.

For each question it searches the 3,996 sections, shows the top 3 passages with titles and URLs, and, when open-book is on, feeds only the top passage through `open_book_prompt`. Closed-book sends the question alone. `serve.load(..., learn=False)` so a chat never writes the weights. Each question is a new prompt. Earlier turns are not carried in, because an earlier passage would crowd the 4,096-character window, and the model was trained on single questions. The chatbot was stopped before the 8-hour run so it would not hold the GPU, and started again on the final weights afterward. `/api/info` then reported 312 documents, 3,996 sections, and weights `weights_csulb`.

## 14. Eight-hour run

The plan was one continuous 480 minutes, a record every two hours, and no early stop. The trainer is not paused for the record. A separate watcher (`tools/csulb/watch_8h.py`) never signals the trainer. It polls every 20 seconds. When the plan clock crosses 120, 240, or 360 minutes, it waits until the log contains `checkpointed`, then hard-links the folder with `cp -al`. Hard links freeze the files that already exist. Later training rewrites experts in `weights_csulb` as new files, so the snapshot stays at the step it was linked. The author folder and `weights_csulb_65min` are checked by manifest timestamp and must not change.

### 14.1 First launch, 70.7 minutes, then a stop signal

Started 20 October 2026, process 34523. The command was the 480-minute command below. Preflight from `runs/csulb_8h_review.md`:

- Scale in `weights_csulb` was 1.0, step 931,303, 1,906.6M characters already read.
- `--lr-reset` was not passed.
- Eight subjects, no others.
- GPU: 383 MiB of 24,564 MiB in use. Free disk at that time: 447 GB.
- Held-out at the start of this session: 0.6292 ± 0.0188.
- The log says `rate x1.000` and does not say `plasticity RESET`.

The process received a stop signal at 70.7 minutes, before the two-hour mark. It finished the chunk and checkpointed. It did not print `weights/ updated`, because the signal arrived during the closing held-out pass. The checkpoint is consistent: step 940,860, 19.57M characters in that session, 1,926.1M over the model's life, 397 experts, scale still 1.0. `weights/` and `weights_csulb_65min` were not written. No hour-2 snapshot exists from this attempt. The watcher died with the trainer and wrote nothing.

Held-out during those 70 minutes, from `runs/csulb_samples.txt`:

| Minutes | Train loss | Held-out | csulb_qa | csulb_qa_open | chat | code | reasoning |
|---|---|---|---|---|---|---|---|
| 0 | | 0.6292 | 0.268 | 0.306 | 0.458 | 0.616 | 0.603 |
| 16 | 0.5161 | 0.6239 | 0.272 | 0.302 | 0.447 | 0.616 | 0.595 |
| 32 | 0.5067 | 0.6145 | 0.245 | 0.237 | 0.442 | 0.618 | 0.601 |
| 48 | 0.4962 | 0.6141 | 0.232 | 0.239 | 0.428 | 0.611 | 0.597 |
| 64 | 0.5191 | 0.6071 | 0.232 | 0.213 | 0.427 | 0.614 | 0.593 |

### 14.2 Resume for the remaining 410 minutes

The stop had already consumed 70.7 minutes of the 480. Starting another 480 would have trained about 9.2 hours and would have reset nothing useful, but it would not have been the plan. The resume is `--minutes 410`, from the checkpoint, still without `--lr-reset`.

Started 22 October 2026 at 07:54, process 32177. Watcher process 33392, told that 70.7 minutes were already done and that new sample text starts at byte 63,380 of `runs/csulb_samples.txt`.

Startup check:

- Eight subjects, exactly: chat, chess, code, csulb_policies, csulb_qa, csulb_qa_open, reasoning, wikipedia.
- Optimiser restored at step 940,861. Earlier reading 1,926.1M.
- `rate x1.000`, no plasticity reset.
- Held-out at the resume: 0.6055 ± 0.0203.
- Author manifest and the 65-minute manifest still had their old timestamps.

Command (the first launch used `--minutes 480`; the resume used `--minutes 410` and appended to the same log):

```bash
.venv/bin/python -u train.py read \
  school_policies_data/train_general school_policies_data/train \
  --weights-dir weights_csulb \
  --held-out school_policies_data/heldout_mix \
  --minutes 410 --save --lr 5e-5 \
  --sample-every 15 \
  --sample-log runs/csulb_samples.txt \
  --history runs/csulb_expert_history.jsonl
```

`--sample-every 15` writes a sample every 15 minutes. The history file is the CSULB expert log, not the author's `runs/expert_history.jsonl`.

### 14.3 Hour 2, hour 4, hour 6

The "step" in the review is the sample step. The hard-linked folder is the next checkpoint, a few hundred steps later. Both are listed. Scale and character counts below are from the sample. The manifest step is the one in `research/checkpoints/`.

| Plan clock | When | Sample step | Snapshot step | Snapshot folder | Session minutes | Train loss | Held-out | Scale | csulb_qa | csulb_qa_open |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 h (134.7 min) | 22 Oct 2026, 09:04 | 949,412 | 950,140 | `weights_csulb_h2` (401 files) | 64.0 | 0.4772 | 0.5920 ± 0.0374 | 1.000 | 0.164 | 0.265 |
| 4 h (245.7 min) | 10:55 | 964,456 | 965,184 | `weights_csulb_h4` (401 files) | 175.0 | 0.4465 | 0.5833 ± 0.0404 | 0.997 | 0.097 | 0.193 |
| 6 h (372.7 min) | 13:03 | 981,462 | 982,163 | `weights_csulb_h6` (402 files) | 302.0 | 0.4460 | 0.5596 ± 0.0399 | 0.992 | 0.082 | 0.189 |

Other subjects at those same samples stayed near where they started. At hour 2: arithmetic 0.994, chat 0.422, chess 0.413, code 0.602, reasoning 0.589, stories 0.802, wikipedia 1.076. At hour 6: arithmetic 0.816, chat 0.407, chess 0.412, code 0.599, reasoning 0.591, stories 0.881, wikipedia 1.061. Chat, code, and reasoning did not climb the way stories had climbed in part 1. Stories is in the held-out mix only, so its number is a measurement, not a training lane.

Reading speed on these samples was about 4,800–4,900 characters per second. The gradient norm sat near 2 against a clip of 1, so gradients were clipped, which the trainer already does at that setting.

### 14.4 Hour 8, the run finished

Recorded 22 October 2026, 14:46:55. The trainer had exited. The log ends with:

```
reached 410 minutes
weights/ checkpointed at step 995,953 (112.8M characters)
read 112.83M characters in 410.1m (4.6k char/s over the whole session)
mean loss over the files: 0.4623
after:  0.5743 +/-0.0219   -0.0312
weights/ updated - the model has read this and kept it
```

The change −0.0312 is against the resume start of 0.6055, not against the author's 0.707.

| | |
|---|---|
| Step | 995,953 |
| This session | 112.83M characters, 410.1 minutes |
| Plan clock | 480.8 minutes (70.7 + 410.1) |
| Life characters | 2,039.0M |
| Train loss | 0.4273 |
| Held-out | 0.5743 ± 0.0219 |
| Scale | 0.9896 |
| Base rate at the sample | 4.95e-05 |
| Experts | 397 |

Per subject at the end: arithmetic 0.990, chat 0.380, chess 0.411, code 0.593, csulb_qa 0.090, csulb_qa_open 0.141, reasoning 0.577, stories 0.919, wikipedia 1.066.

`csulb_qa` went from 0.268 at the start of the 8-hour plan to 0.090. `csulb_qa_open` went from 0.306 to 0.141. Chat went from 0.458 to 0.380. Code and reasoning stayed in the same band. Wikipedia stayed near 1.07. The author's folder and `weights_csulb_65min` were unchanged.

The two plots the trainer rewrites at the end of a run, `runs/dashboard.png` and `runs/training_progress.png`, and the rate trace `runs/expert_rates.jsonl`, were written at 14:46 on 22 October 2026. They sit on the training machine. `runs/*` is gitignored, so those three files are not in the git commit. The CSULB sample log and the CSULB expert history were force-added and are in the repository: `runs/csulb_samples.txt`, `runs/csulb_expert_history.jsonl`.

## 15. Question scores at hours 2, 4, 6, and 8

Scored after the run, one checkpoint at a time, same 60 questions, same greedy decode, learning off. Open-book still shows the model only the top search hit. "Clean" means the 45 questions whose wording is absent from `train/csulb_qa`. The other 15 are the leaked wordings.

| Checkpoint | Set | n | Closed F1 | Closed number | Closed URL | Closed form | Open F1 | Open number | Open URL | Open form |
|---|---|---|---|---|---|---|---|---|---|---|
| h2 | all 60 | 60 | 36.8% | 36.4% | 11.7% | 88.3% | 56.6% | 81.8% | 28.3% | 93.3% |
| h2 | clean | 45 | 40.2% | 25.0% | 13.3% | 86.7% | 62.2% | 79.2% | 35.6% | 91.1% |
| h4 | all 60 | 60 | 38.6% | 42.4% | 18.3% | 85.0% | 56.4% | 78.8% | 30.0% | 95.0% |
| h4 | clean | 45 | 42.0% | 41.7% | 22.2% | 82.2% | 63.6% | 79.2% | 40.0% | 93.3% |
| h6 | all 60 | 60 | 36.3% | 42.4% | 16.7% | 81.7% | 60.5% | 84.8% | 36.7% | 93.3% |
| h6 | clean | 45 | 39.2% | 37.5% | 17.8% | 77.8% | 65.8% | 83.3% | 44.4% | 91.1% |
| h8 | all 60 | 60 | 41.9% | 48.5% | 18.3% | 88.3% | 57.0% | 81.8% | 35.0% | 93.3% |
| h8 | clean | 45 | 45.9% | 41.7% | 22.2% | 86.7% | 61.4% | 79.2% | 42.2% | 91.1% |

Read against the pilot, on all 60:

| | Author | After the pilot | Hour 8 |
|---|---|---|---|
| Closed-book word F1 | 13.4% | 31.6% | 41.9% |
| Open-book word F1 | 17.7% | 51.6% | 57.0% |
| Open-book policy number | 15.2% | 78.8% | 81.8% |
| Open-book URL | 0.0% | 25.0% | 35.0% |
| Open-book form | 0.0% | 91.7% | 93.3% |

Hour 6 (`weights_csulb_h6`, snapshot step 982,163) is the best open-book word F1: 60.5% on all 60, 65.8% on the clean 45. Hour 8 (`weights_csulb`, step 995,953) is the best closed-book word F1, 41.9% on all 60 and 45.9% on the clean 45, and it is the folder the chatbot loads. Open-book F1 did not rise in a straight line from hour 6 to hour 8. The extra two hours improved closed-book and the policy-number score, and the open-book word overlap came back down from the hour-6 peak.

JSON: `runs/csulb_eval_h2.json`, `runs/csulb_eval_h4.json`, `runs/csulb_eval_h6.json`, `runs/csulb_eval_h8.json`. The scoring log is `runs/csulb_eval_8h.log`.

## 16. Where each folder stands

These folders are on the machine that trained them. They are not in git. Each `optim.npz` is 436 MB, and GitHub rejects a file over 100 MB. A copy of every `manifest.json` is in `research/checkpoints/`. sha256 is of `core.npz`.

| Folder | What it is | Step | Scale | Characters read | sha256 of `core.npz` |
|---|---|---|---|---|---|
| `weights/` | Author snapshot. Never trained by this experiment | 914,508 | 0.0515 | 1,872,164,864 | `913645b2b3942a04f301bc34935feb84a3972f21057a3ae4fb2f3d4692d38b71` |
| `weights_csulb_65min` | End of pilot part 1. Kept and not trained further | 923,915 | 1.0 | 1,891,430,400 | `fb55119ba39ae2e1240424e1e1022a59e6c8273b4767ea3f263fbc5b60e0041d` |
| `weights_csulb_h2` | Plan hour 2 | 950,140 | 1.0 | 1,945,139,200 | `688ce486fbe563ca2d839d515fba15fa8235bccbaa4c36c7de9d1e8714932c15` |
| `weights_csulb_h4` | Plan hour 4 | 965,184 | 0.9965 | 1,975,949,312 | `6cf3fa918f4f81a22d459ccfff2dad922338c492dfbcdcd878bacb96825c36bc` |
| `weights_csulb_h6` | Plan hour 6. Best open-book score | 982,163 | 0.9921 | 2,010,722,304 | `f6b9dd9d5801d71e55b5959f92486a325ff1b60c863902a74a17348e1e4ea6d6` |
| `weights_csulb` | End of hour 8. What the chatbot serves | 995,953 | 0.9896 | 2,038,964,224 | `ca52ded040b088fdf6f64099807cbabac6f43db1cadb41b99670479c0a4b7032` |

`weights_mine` is a separate early trial: step 1,747, 66 experts, 3,579,904 characters. It is not a CSULB checkpoint and it was not used for any score above.

Full hashes are in [RESEARCH.md](../RESEARCH.md). Serving can load a folder with no `optim.npz`, because that file is the trunk's Adam state and serving does not learn. Continuing training wants that file. Each expert file already holds that expert's own moments.

## 17. What changed in the author's repository after this training

After the scores were written, this working copy was fast-forwarded to the author's `main`. The new commits are `aa9d53f` (20 October 2026, optimisation and a cropped-evaluation fix) and `8f74c58` (24 October 2026, benchmarking scripts). The CSULB tools, the corpus, the logs, and the weight folders were still on disk after that update.

Two changes matter if this experiment is continued:

- The default lane schedule is now `auto` (`minagi/lanes.py`). It gives more reading time to the subject whose held-out loss is expected to drop most, with a floor of one quarter of an equal share and a ceiling of three times. This run used the older equal rotation. Pass `--lane-schedule equal` to repeat that.
- `serve.load(..., lr=None)` now follows the run's own rate. The chatbot calls `serve.load(weights, learn=False)`, which does not train, so the chatbot behaviour is unchanged.

The author's benchmarks are next-character prediction on held-out text, in nats per character or bits per byte. They are not MMLU, HellaSwag, or ARC. The sets are his own held-out mix, a 50 MB slice of enwik9, and PG19. Those numbers in the README are for his weights, not for `weights_csulb`.

## 18. What this experiment shows, in one place

From the author's checkpoint to the end of hour 8, on the same 60 public-policy questions:

- The reply learned the citation shape. Answer-form went from 0% to about 90% closed-book and open-book.
- With the passage in front, word overlap with the reference went from 17.7% to 57.0% (61.4% on the 45 clean questions at hour 8, 65.8% at hour 6).
- With no passage, word overlap went from 13.4% to 41.9%. The model does remember policy-shaped sentences. It often attaches them to the wrong policy. The closed-book URL score is still 18%.
- Finding the page is a separate system. BM25 puts the right document first for 80% of the 60 questions and 86% of all 3,603 held-out questions, and the right section first for 49% of those 3,603. The model never sees the second and third hits.
- The general subjects that stayed in the mix (chat, code, reasoning, chess, wikipedia) did not show the blow-up that stories and arithmetic showed in the first 65 minutes.
- The plasticity scale ended at 0.990, essentially still at full rate. The run was not trained into the floor.

Files that hold the raw record:

| File | What it is |
|---|---|
| `runs/csulb_scrape.log` | The crawl |
| `school_policies_data/skipped.md` | Every URL that was not kept, and why |
| `runs/csulb_pilot_aborted_lowlr.log` | The crashed first launch |
| `runs/csulb_pilot_part1.log` | 65 minutes, ten subjects, 1e-4 |
| `runs/csulb_pilot_part2.log` | 55 minutes, eight subjects, 5e-5 |
| `runs/csulb_eval.md` | Author versus pilot, with example answers |
| `runs/csulb_8h.log` | Both pieces of the 8-hour run |
| `runs/csulb_8h_review.md` | The hour 2, 4, 6, and 8 record and the score table |
| `runs/csulb_samples.txt` | Samples every 15 minutes |
| `runs/csulb_eval_h2.json` … `h8.json` | Every scored answer at those checkpoints |

The 60-question numbers above are the earlier pilot. They used a 360-character cap, and the open-book condition used BM25. They are not the 29 October evaluation. That evaluation is section 19.

## 19. Why a second evaluation was run

The 60-question scores answer a different question from the one the later protocol tests. Open-book F1 there mixes retrieval with reading, because the passage was the top BM25 hit. The generation cap was 360 characters. Of the 60 scored hour-8 closed-book answers, 31 are exactly 360 characters and 6 are 359. The other 23 stopped earlier, so the cap did not cut every reply. Eleven of those 60 contain the full URL, which is the 18.3% in the table above. The file has 61 rows because it also stores the cheating question, and that row is left out of the average. The 60 reference answers have a median length of 485 characters and a maximum of 764, and 56 of them end with a `Source:` line.

The claim under test on 29 October 2026 is narrower. Continued reading put CSULB policy text into the weights, and the author's frozen next-byte benchmarks show what else those weights kept or lost. Closed-book is the headline: the model sees the question and nothing else. Where a passage is given, it is the stored `passage` field the question was built from, passed through `open_book_prompt` in `tools/csulb/qa_format.py`. `eval.py`'s `open_prompt` is not used, because that function calls BM25.

The reference answer was a training target on the other wordings of the same fact. Only the question wording is new. A high cosine means a new wording produced the trained answer. It does not mean the answer text was never trained.

The written protocol is `research/eval_protocol.md`, committed as `8085b11` before any measurement in this evaluation. The code it was written against is `d6aa869`. The measurement tools then gained `--held-out`, `--out`, and `--force` in `2eea41f`. With those flags omitted, each tool follows the path it had at `d6aa869`. The measurement arithmetic was not changed. `config.yaml` was not edited. `data.held_out` stays `data/val`. `tools/refresh_readme.py` and `measure_benchmarks.sh` were not run.

## 20. The 400 questions

A question is clean when its exact wording does not appear in `school_policies_data/train/csulb_qa`, using the same scan as `questions_in_train` in `tools/csulb/watch_8h.py`. That leaves 2,514 of 3,603. The other 1,089 collide with a training wording.

`tools/csulb/draw_eval_sample.py` draws the sample. Seed `20261029`. Fifty of each kind. Within one kind, at most two questions come from one document. The count starts over at the next kind, so a document can appear more than twice in the full 400. Fifty documents do. The most is six, from Misconduct in Research & Creative Activity. Kinds, in draw order: topic, overview, where, date, number, definition, approval, supersedes. The files are `research/eval/questions_400.jsonl` and `research/eval/questions_prompt_dev.jsonl`. The draw notes are `research/eval/sample_draw.json`.

| | |
|---|---|
| Confirmatory questions | 400 |
| Prompt-development questions, drawn from what remained | 20 |
| Overlap between the 400 and the original 60 | 11 |
| Prompt-development questions that also sit in the original 60 | 1 |

The overlap is recorded. The sample was not redrawn to remove it. The 20 development questions are not in the confirmatory scores. Each row stores the official reference, the policy number, the document date, the URL, and `gold_user`, which is `open_book_prompt` applied once at draw time. Generation uses that stored string, so a later edit to `qa_format.py` does not change the passage bytes.

Of the 400, 279 have a policy number (all of those start with `PS `) and 121 do not. All 50 date questions have a document date. All 400 have a URL.

## 21. Frozen next-byte benchmarks

The main negative result is in these measurements. Hour 8 is worse than the author's weights on enwik9 (1.356 to 1.736 bits/byte), PG19 (2.058 to 2.214 bits/byte), stories (0.476 to 0.905 nats), and arithmetic (0.657 to 0.970 nats). Chat and chess improved. Code, reasoning, and the project's own Wikipedia held-out stayed within 0.03 nats. Most of the damage shows up early. enwik9 is worst at hour 2 (1.779). Stories is already 0.992 at the 65-minute checkpoint. That points to the pilot's learning-rate reset and to reading the tiny stories and arithmetic files over and over. Arithmetic keeps rising after that checkpoint, from 0.793 at 65 minutes to 0.970 at hour 8. The project's Wikipedia held-out barely moves, from 1.077 to 1.067. enwik9, a different Wikipedia slice, gets much worse. This record states that split and does not explain it.

These are the author's measurements, re-run on this code. Numbers already in the training logs are the historical record. They are not this table. Held-out reading changed after those runs, and the first call on `weights/` can otherwise reuse the author's logged line at step 914,508. Every invocation passed `--force`, including the first measurement of `weights/`. A reused log line would have failed the protocol. All 29 records say `fresh: true`. The wrapper is `tools/csulb/bench_frozen.py`, driven by `tools/csulb/run_frozen_suite.py`. Each record was copied into `research/eval/benchmarks/` before the next run overwrote the shared file under `runs/`.

Before the runs, each of the six folders was checked by manifest step and `core.npz` sha256 against `RESEARCH.md`. All six matched. `weights/manifest.json` was last written at 2026-10-12 15:55:31 -0700. No `train.py` and no `chat.py` was running. The GPU was an RTX 4090 with 24,564 MiB. The manifest timestamp was the same after the suite.

The enwik9 slice is the last 50,000,000 bytes of enwik9, sha256 `fa616cef732f5622ff8bd6706fec08638491beb6b7934786bcf95bf69c8a86cc`. PG19 is the test split, 100 books, 41,289,101 bytes on disk and 41,289,001 bytes scored, window 4,096.

The author's weights and hour 8 were measured twice. The largest repeat gap was under 0.0001 nats. The author's own noise line is about 0.03 nats. The repeats are inside that line.

Held-out loss is nats per character. `data/val` is the author's seven subjects. The CSULB mix adds `csulb_qa` and `csulb_qa_open`, so its average is a different quantity. Hours 2 through 8 are one series. The 65-minute checkpoint is a different run (ten subjects, learning rate `1e-4`, plasticity reset) and is labeled separately. The chart is `research/eval/benchmarks/heldout_val.svg`.

`data/val`, mean of the seven subjects. The second number is the repeat.

| Checkpoint | Step | Loss |
|---|---:|---:|
| Author | 914,508 | 0.6945, repeat 0.6945 |
| 65 minutes | 923,915 | 0.7465 |
| Hour 2 | 950,140 | 0.7033 |
| Hour 4 | 965,184 | 0.6796 |
| Hour 6 | 982,163 | 0.6843 |
| Hour 8 | 995,953 | 0.7028, repeat 0.7028 |

Per subject on `data/val`:

| Subject | Author | 65 min | Hour 2 | Hour 4 | Hour 6 | Hour 8 |
|---|---:|---:|---:|---:|---:|---:|
| Arithmetic | 0.657 | 0.793 | 0.812 | 0.839 | 0.915 | 0.970 |
| Chat | 0.525 | 0.504 | 0.452 | 0.411 | 0.386 | 0.380 |
| Chess | 0.960 | 0.446 | 0.414 | 0.415 | 0.413 | 0.413 |
| Code | 0.596 | 0.672 | 0.630 | 0.604 | 0.602 | 0.601 |
| Reasoning | 0.571 | 0.660 | 0.612 | 0.588 | 0.578 | 0.583 |
| Stories | 0.476 | 0.992 | 0.894 | 0.827 | 0.823 | 0.905 |
| Wikipedia | 1.077 | 1.158 | 1.108 | 1.075 | 1.073 | 1.067 |

Arithmetic rose by 0.314 nats and stories by 0.430. Both gaps are larger than 0.03. Chat fell by 0.145 and chess by 0.547. Code, reasoning, and the project's own Wikipedia held-out moved by less than 0.03.

On the CSULB mix the other seven subjects stay within 0.001 nats of that table. The two added subjects:

| Subject | Author | 65 min | Hour 2 | Hour 4 | Hour 6 | Hour 8 |
|---|---:|---:|---:|---:|---:|---:|
| csulb_qa | 0.802 | 0.389 | 0.143 | 0.099 | 0.082 | 0.081 |
| csulb_qa_open | 0.795 | 0.434 | 0.246 | 0.184 | 0.160 | 0.139 |

The mix mean goes from 0.7177 (repeat 0.7176) to 0.5710 (repeat 0.5710). That drop is those two subjects.

enwik9, frozen, 2,000 pieces of 2,048 bytes, bits per byte:

| Checkpoint | Bits/byte |
|---|---:|
| Author | 1.3556, repeat 1.3556 |
| 65 minutes | 1.6284 |
| Hour 2 | 1.7790 |
| Hour 4 | 1.6980 |
| Hour 6 | 1.6984 |
| Hour 8 | 1.7359, repeat 1.7358 |

PG19, frozen, all 100 books, window 4,096. Hours 2, 4, and 6 were not run. Bits per byte, and nats per byte:

| Checkpoint | Bits/byte | Nats/byte |
|---|---:|---:|
| Author | 2.0578, repeat 2.0577 | 1.4263 |
| 65 minutes | 2.2864 | 1.5848 |
| Hour 8 | 2.2141, repeat 2.2141 | 1.5347 |

Hour 8 is worse than the author's weights on enwik9 (1.356 to 1.736 bits/byte) and on PG19 (2.058 to 2.214 bits/byte). Both gaps are larger than 0.03 nats. Stories goes from 0.476 to 0.905 and arithmetic from 0.657 to 0.970. Chat and chess improve. Code, reasoning, and the project's own Wikipedia held-out stay within 0.03 nats. enwik9 is worst at hour 2 (1.779), and stories is already 0.992 at the 65-minute checkpoint. The Wikipedia held-out goes from 1.077 to 1.067 while enwik9 gets much worse. That split is left as it stands. These fresh numbers are not comparable to the README figures for a later author checkpoint.

## 22. Local answers on the 400

`tools/csulb/generate_answers.py`. Checkpoints are the author's weights and hour 8 only. Hours 2, 4, and 6 were not generated. There is no hour-6 versus hour-8 comparison on these questions.

Decoding is `serve.stream`, temperature 0, the same call `eval.ask` makes, including that function's repetition guard. Learning is off. The limit is 800 yields from `serve.stream`. A yield is usually one character. The saved answer is cut at `</bot>`, at the incomplete marker `</bo` that `serve.stream` leaves when it stops, or at a new `<user>` or `<bot>` turn, whichever comes first. `end_marker` means the stream finished before 800 yields. `limit` means it used all 800.

1,600 answers were written, 400 questions by 2 modes by 2 checkpoints, to `research/eval/local/answers.jsonl`. None is empty. None still contains a chat marker. A limit stop is 800 yields from `serve.stream`. A yield is usually one character, and the saved text is stripped. Of the 360 limit stops, 291 saved answers are 800 characters. The other 69 are between 799 and 807.

| | Closed, end | Closed, limit | Gold, end | Gold, limit |
|---|---:|---:|---:|---:|
| Author | 204 | 196 | 266 | 134 |
| Hour 8 | 382 | 18 | 388 | 12 |

## 23. Claude

Model id `claude-opus-5-5`. A one-question probe returned that id, stop reason `end_turn`. Effort is `low`, sent as `output_config.effort`. No temperature, `top_p`, or `top_k` is sent. One question per request. No history. Claude never receives a local answer or a reference answer. The output limit is 16,000 tokens, which covers the thinking trace and the answer.

The 20 development questions were asked one at a time, both conditions, 40 replies. All 40 stopped on `end_turn`. Closed-book length ran from 479 to 1,000 characters (median 668.5), and from 3 to 8 sentences. Gold-passage length ran from 385 to 923 characters (median 622), and from 3 to 12 sentences. That was the one allowed format change. Both prompts now say to stop after 5 sentences and not to use a heading or a bullet list. The frozen text is `research/eval/claude/prompt.json`, commit `57c3bc7`. Those 20 are not scored.

The 400 questions then went through the Batch API, both conditions, 800 requests. A second batch repeated the first 50 questions in file order, both conditions, 100 requests, for the run-to-run ceiling. Results do not return in order. Each result was matched by question id. All 900 replies are model `claude-opus-5-5`, stop reason `end_turn`, and valid. Refusals: 0. Length-limit stops: 0. Those files are version 1 and were not overwritten.

Version 2, recorded in the 29 October 2026 amendment, adds one sentence to the start of the closed-book prompt: the questions are about policies of California State University, Long Beach. Nothing else in that prompt changes. The gold-passage prompt was not re-run. The re-run is 400 closed-book questions plus the first 50, closed-book only. All 450 replies are model `claude-opus-5-5`, stop reason `end_turn`, and valid. Refusals: 0. Length-limit stops: 0. The new files are `research/eval/claude/prompt_v2.json` and `answers_v2.jsonl`.

## 24. Scores on the 400

`tools/csulb/score_similarity.py`. Primary metric: cosine of the answer to the official reference.

Primary embedding: `Qwen/Qwen3-Embedding-0.6B`. Secondary: `sentence-transformers/all-mpnet-base-v2`. Both texts are encoded with no instruction. Qwen's default prompt name is unset, so the query instruction is not applied to either side. Vectors are L2-normalized. Cosine is their dot product. A ranking is reported only when the two embeddings agree, and they do, on both conditions.

Before the embedding, a leading `According to …` (through the first colon), any `Source:` line, and chat tags are removed. The local answer, the Claude answer, and the reference are then cut to the length of that question's cleaned reference. Exact match and word F1 use the raw answer after the end-of-reply cut and before that stripping. Word F1 is the project's function in `tools/csulb/eval.py`. The policy number is the stored number without the `PS ` prefix. Date is scored only for `kind == date`. A missing number, date, or URL is not a miss.

Chance is the cosine to a different question's reference, 10 shuffles, seed `20261029`, with no answer left on its own reference. The ceiling is the cosine between Claude's two runs on the 50. High cosine, for the error panel, is the top quarter of cosine-to-reference for that system and that embedding. A tie at the cutoff stays in. Bootstrap intervals are 10,000 resamples on the paired questions. Every system has a valid answer on all 400, so the paired comparisons use all 400.

### Closed-book

Claude in this table is version 2. The last column is version 1, the same closed-book prompt without the campus sentence.

| | Author | Hour 8 | Claude | Without CSULB context |
|---|---:|---:|---:|---:|
| Qwen cosine | 0.543 | 0.693 | 0.642 | 0.601 |
| mpnet cosine | 0.461 | 0.645 | 0.585 | 0.559 |
| Word F1 | 0.117 | 0.562 | 0.194 | 0.174 |
| Policy number | 22 / 279 | 112 / 279 | 32 / 279 | 32 / 279 |
| Date | 2 / 50 | 12 / 50 | 4 / 50 | 4 / 50 |
| Full URL | 0 / 400 | 174 / 400 | 0 / 400 | 0 / 400 |

Naming the campus raises Claude's closed-book cosine by 0.041 on Qwen and 0.026 on mpnet, and word F1 by 0.020. The policy-number, date, and URL counts do not move.

Chance, Qwen then mpnet: author 0.305 and 0.206, hour 8 0.374 and 0.259, Claude version 2 0.401 and 0.282, version 1 0.366 and 0.259.

| Contrast | Embedding | Mean gap | 95% interval |
|---|---|---:|---:|
| Hour 8 minus author | Qwen | 0.150 | 0.126 to 0.174 |
| Hour 8 minus author | mpnet | 0.184 | 0.157 to 0.212 |
| Hour 8 minus Claude, version 2 | Qwen | 0.051 | 0.025 to 0.076 |
| Hour 8 minus Claude, version 2 | mpnet | 0.060 | 0.029 to 0.092 |

Both embeddings rank hour 8, then Claude, then the author. With version 1 the hour-8 gap on Qwen was 0.092 (0.065 to 0.119). Agreement between hour 8 and Claude version 2 is 0.577 (Qwen) and 0.487 (mpnet). Hour 8 is closer to the official policy wording (0.693) than to Claude. Hour 8 was trained on these reference answers, on other wordings of the same facts. Claude paraphrases. The version-2 closed-book ceiling on the 50 is 0.916 (Qwen) and 0.895 (mpnet). The version-1 ceiling was 0.896 and 0.887.

### Gold passage

Claude in this table is version 1. The gold-passage prompt was not re-run.

| | Author | Hour 8 | Claude |
|---|---:|---:|---:|
| Qwen cosine | 0.634 | 0.918 | 0.792 |
| mpnet cosine | 0.577 | 0.909 | 0.760 |
| Word F1 | 0.201 | 0.867 | 0.445 |
| Policy number | 25 / 279 | 236 / 279 | 278 / 279 |
| Date | 4 / 50 | 44 / 50 | 50 / 50 |
| Full URL | 0 / 400 | 353 / 400 | 79 / 400 |

Claude's gold-passage URL count is 79 of 400. Its prompt said to mention a source only if the passage states it, and to answer in 2 to 5 sentences. Hour 8 was trained to end every answer with `Source:` and the URL.

| Contrast | Embedding | Mean gap | 95% interval |
|---|---|---:|---:|
| Hour 8 minus author | Qwen | 0.284 | 0.266 to 0.303 |
| Hour 8 minus author | mpnet | 0.332 | 0.310 to 0.353 |
| Hour 8 minus Claude | Qwen | 0.127 | 0.110 to 0.143 |
| Hour 8 minus Claude | mpnet | 0.149 | 0.128 to 0.170 |

Both embeddings rank hour 8, then Claude, then the author. The gold ceiling is 0.954 (Qwen) and 0.938 (mpnet). Agreement between hour 8 and Claude is 0.765 (Qwen) and 0.726 (mpnet).

### By kind, closed-book Qwen cosine

Fifty questions of each kind.

Claude in this table is version 2.

| Kind | Author | Hour 8 | Claude | Without CSULB context | Hour 8 word F1 |
|---|---:|---:|---:|---:|---:|
| Topic | 0.407 | 0.458 | 0.583 | 0.563 | 0.345 |
| Overview | 0.480 | 0.587 | 0.661 | 0.615 | 0.387 |
| Where | 0.560 | 0.856 | 0.792 | 0.672 | 0.711 |
| Date | 0.483 | 0.629 | 0.632 | 0.534 | 0.527 |
| Number | 0.676 | 0.926 | 0.774 | 0.710 | 0.813 |
| Definition | 0.448 | 0.387 | 0.529 | 0.548 | 0.265 |
| Approval | 0.660 | 0.886 | 0.633 | 0.636 | 0.784 |
| Supersedes | 0.630 | 0.813 | 0.532 | 0.532 | 0.663 |

Claude's cosine is higher than hour 8's on topic (0.583 against 0.458), overview (0.661 against 0.587), and definition (0.529 against 0.387). On definition, hour 8 is below the author's weights as well. mpnet agrees that hour 8 is below the author on definition (0.368 to 0.293) and above the author on the other seven kinds. On definition, both embeddings also place Claude above hour 8 (mpnet 0.521 against 0.293). The overall order is still hour 8, then Claude, then the author.

### Error panel and the demo

Of the 36 hour-8 closed-book answers in the Qwen top quarter whose policy number or date is missing, 19 omit a fact the reference itself contains. The other 17 are flagged because the document has a number and the reference sentence does not use it. The hour-8 gold panel has 142 rows, because the 100th cosine is already 1.0 and ties stay in. All 21 flagged rows in that panel are the second kind. Claude version 2's closed-book panel flags 78 of 100, and 66 of those omit a fact the reference contains. The other 12 are flagged because the reference sentence does not use a number the document has. Version 1 flagged 81 of 100. Claude's gold panel, version 1, flags none of its top 100.

The closed-book example is `f02638`, under the rule in the 29 October 2026 amendment. The question names CSULB, the kind is overview, and the hour-8 Qwen cosine is 1.0, the highest among topic, overview, and definition. The stored policy text confirms policy number 19-06, the date 5/16/2019, the superseded statements, the March 21, 2019 recommendation, the April 18, 2019 approval, and the Academic Senate URL. Hour 8 repeats that reference. The author's weights repeat the question as a numbered list. Claude version 2 describes the salary-range procedure. Its cosine to the adoption-paragraph reference is 0.526.

`f01658` is not the example. That question does not name CSULB. Version 1 of Claude answered it as possibly the University of New Mexico or the University of Southern California.

Sorted by closed-book Qwen cosine, the two middle values of the 400 are 0.734. They are `f00208` (0.734086) and `f01589` (0.734422). The backup is `f00208`. The reference says the complainant should contact the Human Resources Manager. Hour 8 says the Executive Director. The cosine is high and the office is wrong.

The same tables, the chart, and this demo are in `research/eval/results.md`. The per-question cosines are in `research/eval/scores.json`.

## 25. What is and is not in git

The weight folders stay on the training machine. Each `optim.npz` is about 436 MB, GitHub rejects a file over 100 MB, and the six folders together are about 61 GB. `data/` (the general corpus, the enwik9 slice, and PG19) is gitignored and rebuilt by `python3 -m corpora`. `.env` holds the Claude key and is gitignored. The evaluation text, the 400 questions, the benchmark JSON, the local answers, the Claude answers, and the score file are in the repository. They do not contain the key.
