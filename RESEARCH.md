# CSULB policy research

The full step-by-step record is [research/end-to-end.md](research/end-to-end.md): the author's starting weights, the scrape, the question build, the pilot, the 8-hour run, and the 29 October 2026 evaluation, with the commands and the scores. The evaluation tables are also in [research/eval/results.md](research/eval/results.md).

This repository is the thesis record for a continual-learning experiment on public California State University, Long Beach (CSULB) policies. It is a copy of the model code with the experiment added on top. Nothing here replaces the original project.

## Where the model comes from

The model, the training program, and the commits up to `8f74c58` (24 October 2026, “benchmarking scripts”) are from:

- Repository: https://github.com/volotat/mini-AGI
- Author: Alexey Borsky
- License: MIT, kept in full in `LICENSE` (Copyright (c) 2026 Alexey Borsky)

The original remote is still recorded locally as `origin`. This thesis remote is `research` (https://github.com/quachphu/AGI-Research). New work is the CSULB tools, the scraped corpus, the training logs, and the evaluation files. The author’s code was not rewritten.

The author’s published weights are the separate `weights/` folder on the training machine (step 914,508). That folder was never the target of the CSULB trainer. The same snapshot is the one the author documents in this README. It is not uploaded here, because it is the author’s checkpoint and it contains a 436 MB `optim.npz`.

## What was trained

Only this model was trained. No other language model wrote or scored the answers. The run reads eight subjects with an equal share: chat, chess, code, reasoning, wikipedia, `csulb_policies`, `csulb_qa`, and `csulb_qa_open`. The learning rate is `5e-5`. `--lr-reset` is not used on the continuation. The weights directory is `weights_csulb`.

The chatbot shows the official passage and its URL, then asks the model to answer from that passage. The page is `tools/csulb/chat.py` on port 8765, pointed at `weights_csulb`.

| Stage | What it is |
|---|---|
| 2-hour pilot | `runs/csulb_eval.md`, `runs/csulb_eval_before.json`, `runs/csulb_eval_after.json` |
| 8-hour continuation | `runs/csulb_8h.log`, `runs/csulb_8h_review.md`, `runs/csulb_samples.txt` |
| Question scores | `runs/csulb_eval_h2.json`, `h4`, `h6`, `h8` |

On the 60 held-out questions after the 8-hour run, closed-book word F1 at the final checkpoint is 41.9% and open-book word F1 is 57.0%. On the 45 questions whose wording is not already in the training file, those figures are 45.9% and 61.4%. Hour 6 (`weights_csulb_h6`) is the strongest open-book checkpoint (60.5% on all 60, 65.8% on the clean 45). The full table is in `runs/csulb_8h_review.md`.

## Where the trained weights are

The weight folders are on the machine that trained them, at `/home/pquach/Downloads/mini-AGI/`. They are not in this git repository. GitHub refuses any single file over 100 MB, and each `optim.npz` is 436 MB. Together the folders are about 61 GB, which is also past what a normal git push will accept. The chat and the evaluation can reload a folder without `optim.npz` (that file is the trunk’s Adam state, used when training continues). The expert files still hold each expert’s own moments.

| Folder | Role | Step | Plasticity scale | Characters read | sha256 of `core.npz` |
|---|---|---|---|---|---|
| `weights/` | Author snapshot, not trained by this experiment | 914,508 | 0.0515 | 1,872,164,864 | `913645b2b3942a04f301bc34935feb84a3972f21057a3ae4fb2f3d4692d38b71` |
| `weights_csulb_65min` | Kept snapshot from the pilot | 923,915 | 1.0 | 1,891,430,400 | `fb55119ba39ae2e1240424e1e1022a59e6c8273b4767ea3f263fbc5b60e0041d` |
| `weights_csulb_h2` | Plan hour 2 | 950,140 | 1.0 | 1,945,139,200 | `688ce486fbe563ca2d839d515fba15fa8235bccbaa4c36c7de9d1e8714932c15` |
| `weights_csulb_h4` | Plan hour 4 | 965,184 | 0.9965 | 1,975,949,312 | `6cf3fa918f4f81a22d459ccfff2dad922338c492dfbcdcd878bacb96825c36bc` |
| `weights_csulb_h6` | Plan hour 6, best open-book score | 982,163 | 0.9921 | 2,010,722,304 | `f6b9dd9d5801d71e55b5959f92486a325ff1b60c863902a74a17348e1e4ea6d6` |
| `weights_csulb` | Final checkpoint, the one the chatbot serves | 995,953 | 0.9896 | 2,038,964,224 | `ca52ded040b088fdf6f64099807cbabac6f43db1cadb41b99670479c0a4b7032` |
| `weights_mine` | Early local trial, 66 experts, not the CSULB model | 1,747 | 1.0 | 3,579,904 | `90628038a5223d6816b98d7041f19fcab128c8e2675e4262358e4bc7f65f0750` |

A copy of each `manifest.json` is in `research/checkpoints/`. Each final folder has 397 experts except `weights_mine`.

## What else is in the repository

- `tools/csulb/` — scrape, extract, corpus build, question format, BM25 retrieval, evaluation, chatbot, and the 8-hour watcher.
- `school_policies_data/` — pages that could be fetched, extracted text, the training and held-out files, and the general-subject lanes used so the CSULB text was not the only thing the model read.
- `explaination/` — notes and diagrams written while reading the model.
- `runs/csulb_*` — training logs, sample log, and evaluation JSON.

Pages the crawl could not fetch (robots rules, login walls, and missing pages) are listed in `school_policies_data/skipped.md`. Those pages were skipped.
