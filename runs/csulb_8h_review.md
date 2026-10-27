# CSULB 8-hour training review

One uninterrupted continuation from `weights_csulb`. The trainer is not paused at 2, 4, or 6 hours. Snapshots are hardlinks taken after the next checkpoint once session time passes those marks. Question scores are added after the run.

## Preflight

- Recorded: 2026-10-20, before launch
- Plasticity scale in `weights_csulb/manifest.json`: 1.0 (step 931,303, 1906.6M characters already read)
- `--lr-reset` is not used
- Weights directory: `weights_csulb`. `weights/` and `weights_csulb_65min/` are not passed to the trainer
- Training subjects: chat, chess, code, reasoning, wikipedia, csulb_policies, csulb_qa, csulb_qa_open
- Held-out: `school_policies_data/heldout_mix` (arithmetic, chat, chess, code, csulb_qa, csulb_qa_open, reasoning, stories, wikipedia)
- Chatbot on port 8765 was stopped so it would not hold the GPU
- GPU before launch: 383 MiB of 24564 MiB in use
- Free disk: 447 GB

## Startup check

- Trainer pid: 34523
- 8 subjects, and no others: chat, chess, code, csulb_policies, csulb_qa, csulb_qa_open, reasoning, wikipedia
- Plasticity line says `rate x1.000` and does not say `plasticity RESET`
- Held-out at the start of this session: 0.6292 +/-0.0188 on `school_policies_data/heldout_mix`
- `weights/manifest.json` and `weights_csulb_65min/manifest.json` were not rewritten during startup
- Command line is `--weights-dir weights_csulb` with `--minutes 480` and no `--lr-reset`

## Interruption at 71 minutes

The first process (pid 34523) received a stop signal at 70.7 minutes, before the 2-hour mark. It checkpointed and kept the weights. There is no `weights/ updated` line because the final score after the stop was cut off. The saved checkpoint is consistent: step 940,860, 1,926.1M characters over the model's life (19.57M of them in that session), 397 experts, plasticity scale still 1.0. `weights/` and `weights_csulb_65min/` were not written.

Held-out loss during that partial hour, from `runs/csulb_samples.txt`:

| minutes | train | held-out | csulb_qa | csulb_qa_open | chat | code | reasoning |
|---|---|---|---|---|---|---|---|
| 0 | | 0.6292 | 0.268 | 0.306 | 0.458 | 0.616 | 0.603 |
| 16 | 0.5161 | 0.6239 | 0.272 | 0.302 | 0.447 | 0.616 | 0.595 |
| 32 | 0.5067 | 0.6145 | 0.245 | 0.237 | 0.442 | 0.618 | 0.601 |
| 48 | 0.4962 | 0.6141 | 0.232 | 0.239 | 0.428 | 0.611 | 0.597 |
| 64 | 0.5191 | 0.6071 | 0.232 | 0.213 | 0.427 | 0.614 | 0.593 |

The continuation starts from that checkpoint, without `--lr-reset`, for the remaining 410 minutes. Snapshots at plan hours 2, 4, and 6 are 49, 169, and 289 minutes into this continuation.

## Continuation startup

- Trainer pid: 32177, started 2026-10-22 07:54
- 8 subjects, no others: chat, chess, code, csulb_policies, csulb_qa, csulb_qa_open, reasoning, wikipedia
- Optimiser restored from step 940,861. Earlier reading: 1926.1M characters
- Plasticity line says `rate x1.000` and does not say `plasticity RESET`
- Held-out at this resume: 0.6055 +/-0.0203
- `--minutes 410`, `--weights-dir weights_csulb`, no `--lr-reset`
- `weights/manifest.json` and `weights_csulb_65min/manifest.json` still have their original timestamps

Command:

```bash
.venv/bin/python -u train.py read \
  school_policies_data/train_general school_policies_data/train \
  --weights-dir weights_csulb \
  --held-out school_policies_data/heldout_mix \
  --minutes 480 --save --lr 5e-5 \
  --sample-every 15 \
  --sample-log runs/csulb_samples.txt \
  --history runs/csulb_expert_history.jsonl
```

## Hour 2 (h2)

- Recorded: 2026-10-22 09:04:17
- Trainer still running: yes (pid 32177)
- Step: 949412
- Characters read this session: 17.5M
- Characters read over the model's life: 1943.6M
- Minutes into this session: 64.0
- Minutes of the 8-hour plan: 134.7
- Train loss: 0.4772
- Held-out loss: 0.592 +/-0.0374
- Learning-rate scale: 1.0
- Base learning rate: 5.00e-05
- Per subject: arithmetic 0.994   chat 0.422   chess 0.413   code 0.602   csulb_qa 0.164   csulb_qa_open 0.265   reasoning 0.589   stories 0.802   wikipedia 1.076

```
step 949,412   1943.6M of 905M characters (214.73%)   64 min   397 experts
context 4,096 characters of 4,096   reading 4,909 char/s   writing 59.6 char/s   still gaining +0.0221 deep into it
grad norm 1.89 against a clip of 1   clipping
train loss 0.4772   lr 5.00e-05   evidence t +1.97 over 65.7 (effect +0.0438)   rate x1.000
held-out loss 0.5920 +/-0.0374 nats   0.8541 bits/char   perplexity 1.81   gap +0.1148   level +0.0053
  arithmetic 0.994   chat 0.422   chess 0.413   code 0.602   csulb_qa 0.164   csulb_qa_open 0.265   reasoning 0.589   stories 0.802   wikipedia 1.076
repeats 16% of 8-grams, greedy with no guard
```
- Snapshot: hardlinked 401 files to weights_csulb_h2
- Author weights: unchanged: weights/manifest.json and weights_csulb_65min/manifest.json

## Hour 4 (h4)

- Recorded: 2026-10-22 10:55:43
- Trainer still running: yes (pid 32177)
- Step: 964456
- Characters read this session: 48.4M
- Characters read over the model's life: 1974.5M
- Minutes into this session: 175.0
- Minutes of the 8-hour plan: 245.7
- Train loss: 0.4465
- Held-out loss: 0.5833 +/-0.0404
- Learning-rate scale: 0.997
- Base learning rate: 4.99e-05
- Per subject: arithmetic 1.035   chat 0.400   chess 0.410   code 0.599   csulb_qa 0.097   csulb_qa_open 0.193   reasoning 0.589   stories 0.855   wikipedia 1.073

```
step 964,456   1974.5M of 905M characters (218.14%)   175 min   397 experts
context 4,096 characters of 4,096   reading 4,869 char/s   writing 59.1 char/s   still gaining +0.0188 deep into it
grad norm 2.08 against a clip of 1   clipping
train loss 0.4465   lr 4.99e-05   evidence t +1.22 over 65.7 (effect +0.0270)   rate x0.997
held-out loss 0.5833 +/-0.0404 nats   0.8416 bits/char   perplexity 1.79   gap +0.1369   level +0.0053
  arithmetic 1.035   chat 0.400   chess 0.410   code 0.599   csulb_qa 0.097   csulb_qa_open 0.193   reasoning 0.589   stories 0.855   wikipedia 1.073
repeats 19% of 8-grams, greedy with no guard
```
- Snapshot: hardlinked 401 files to weights_csulb_h4
- Author weights: unchanged: weights/manifest.json and weights_csulb_65min/manifest.json

## Hour 6 (h6)

- Recorded: 2026-10-22 13:03:09
- Trainer still running: yes (pid 32177)
- Step: 981462
- Characters read this session: 83.2M
- Characters read over the model's life: 2009.3M
- Minutes into this session: 302.0
- Minutes of the 8-hour plan: 372.7
- Train loss: 0.446
- Held-out loss: 0.5596 +/-0.0399
- Learning-rate scale: 0.992
- Base learning rate: 4.96e-05
- Per subject: arithmetic 0.816   chat 0.407   chess 0.412   code 0.599   csulb_qa 0.082   csulb_qa_open 0.189   reasoning 0.591   stories 0.881   wikipedia 1.061

```
step 981,462   2009.3M of 905M characters (221.99%)   302 min   397 experts
context 4,096 characters of 4,096   reading 4,809 char/s   writing 60.0 char/s   still gaining +0.0207 deep into it
grad norm 2.09 against a clip of 1   clipping
train loss 0.4460   lr 4.96e-05   evidence t +1.39 over 65.7 (effect +0.0308)   rate x0.992
held-out loss 0.5596 +/-0.0399 nats   0.8074 bits/char   perplexity 1.75   gap +0.1136   level +0.0053
  arithmetic 0.816   chat 0.407   chess 0.412   code 0.599   csulb_qa 0.082   csulb_qa_open 0.189   reasoning 0.591   stories 0.881   wikipedia 1.061
repeats 31% of 8-grams, greedy with no guard
```
- Snapshot: hardlinked 402 files to weights_csulb_h6
- Author weights: unchanged: weights/manifest.json and weights_csulb_65min/manifest.json

## Hour 8 (final save)

- Recorded: 2026-10-22 14:46:55
- Trainer still running: no, the trainer has exited
- Step: 995953
- Characters read this session: 112.9M
- Characters read over the model's life: 2039.0M
- Minutes into this session: 411.0
- Minutes of the 8-hour plan: 481.7
- Train loss: 0.4273
- Held-out loss: 0.5743 +/-0.0219
- Learning-rate scale: 0.9895755939831209
- Base learning rate: 4.95e-05
- Per subject: arithmetic 0.990   chat 0.380   chess 0.411   code 0.593   csulb_qa 0.090   csulb_qa_open 0.141   reasoning 0.577   stories 0.919   wikipedia 1.066

```
step 995,953   2039.0M of 905M characters (225.26%)   411 min   397 experts
context 4,096 characters of 4,096   reading 4,698 char/s   writing 59.8 char/s   still gaining +0.0199 deep into it
grad norm 1.90 against a clip of 1   clipping
train loss 0.4273   lr 4.95e-05
held-out loss 0.5743 +/-0.0219 nats   0.8285 bits/char   perplexity 1.78   gap +0.1470
  arithmetic 0.990   chat 0.380   chess 0.411   code 0.593   csulb_qa 0.090   csulb_qa_open 0.141   reasoning 0.577   stories 0.919   wikipedia 1.066
repeats 25% of 8-grams, greedy with no guard
```
- Author weights: unchanged: weights/manifest.json and weights_csulb_65min/manifest.json

Last log lines:

```
    weights/ checkpointed
    weights/ checkpointed at step 994,943 (110.8M characters)
    weights/ checkpointed at step 995,633 (112.2M characters)
  reached 410 minutes
    weights/ checkpointed at step 995,953 (112.8M characters)

read 112.83M characters in 410.1m (4.6k char/s over the whole session, checks and samples included)
  mean loss over the files: 0.4623
  after:  0.5743 +/-0.0219   -0.0312
  final samples in runs/csulb_samples.txt
  weights/: 397 experts + 3 bundles, 10461.6 MB (paged - experts were already on disk)
  weights/ updated - the model has read this and kept it
```

## Question scores after the run

Scored after training finished. Open-book shows the model the top search hit. The clean set is the test questions whose wording does not appear in `train/csulb_qa`. The other test questions use a wording that training pairs with a different document.

| checkpoint | set | n | closed F1 | closed number | closed URL | closed form | open F1 | open number | open URL | open form |
|---|---|---|---|---|---|---|---|---|---|---|
| h2 | all (leaked 15) | 60 | 36.8% | 36.4% | 11.7% | 88.3% | 56.6% | 81.8% | 28.3% | 93.3% |
| h2 | clean wording | 45 | 40.2% | 25.0% | 13.3% | 86.7% | 62.2% | 79.2% | 35.6% | 91.1% |
| h4 | all (leaked 15) | 60 | 38.6% | 42.4% | 18.3% | 85.0% | 56.4% | 78.8% | 30.0% | 95.0% |
| h4 | clean wording | 45 | 42.0% | 41.7% | 22.2% | 82.2% | 63.6% | 79.2% | 40.0% | 93.3% |
| h6 | all (leaked 15) | 60 | 36.3% | 42.4% | 16.7% | 81.7% | 60.5% | 84.8% | 36.7% | 93.3% |
| h6 | clean wording | 45 | 39.2% | 37.5% | 17.8% | 77.8% | 65.8% | 83.3% | 44.4% | 91.1% |
| h8 | all (leaked 15) | 60 | 41.9% | 48.5% | 18.3% | 88.3% | 57.0% | 81.8% | 35.0% | 93.3% |
| h8 | clean wording | 45 | 45.9% | 41.7% | 22.2% | 86.7% | 61.4% | 79.2% | 42.2% | 91.1% |
