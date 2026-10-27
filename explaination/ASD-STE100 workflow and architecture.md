# mini-AGI: Workflow and Architecture
## Written in Simplified Technical English (ASD-STE100 style)

| Item | Data |
|------|------|
| Subject | mini-AGI, repository commit `60f7ae3` |
| Date | 12 Oct 2026 |
| Computer used for the tests | Ubuntu, Python 3.10, GPU RTX 4090 (24 GB) |
| Related files | `detail explaination.md`, folder `diagram/` |

---

## 0. About this document

**Note on the writing style.**
This document follows the main writing rules of ASD-STE100 Simplified Technical English.
The rules used are:

- Sentences in a procedure have 20 words or fewer.
- Other sentences have 25 words or fewer.
- Each sentence has one topic. Each instruction has one action.
- The verbs are in the active voice and in the simple tenses.
- Each technical name has one meaning. The name does not change in the text.

This document is not an official ASD-STE100 document. It was not checked with the official
STE checker or the official dictionary. Use the name "STE-style" if you need to be exact.

---

## 1. Technical names

Each name has one meaning in this document.

| Name | Meaning |
|------|---------|
| **character** | One unit of text for the model. It is one byte or one marker. |
| **byte** | A number from 0 to 255. |
| **marker** | A special character, for example `<user>` or `</bot>`. There are 9 markers. |
| **vocabulary** | The 265 characters that the model knows: 256 bytes and 9 markers. |
| **model** | The program data that reads and writes characters. |
| **parameter** | One number that the model learns. |
| **loss** | A number that shows how wrong the model is. A low loss is good. |
| **nat** | The unit of loss. One nat is 1.443 bits. |
| **held-out text** | Text that the model does not learn from. It is used to test the model. |
| **lane** | One subject of training text. One lane is one folder. |
| **prelude block** | One of the 2 blocks that process each character one time. |
| **recurrent block** | The one block that the model uses again and again. |
| **row** | One use of the recurrent block for one character. A character can use up to 24 rows. |
| **state** | The vector of 512 numbers that the model keeps for each character. |
| **halting head** | The part of the model that decides when a character stops. |
| **expert** | A small network of 3.15 million parameters. It is stored in one file. |
| **pool** | The set of all experts. The snapshot has 397 experts. |
| **router** | The part of the model that gives a score to each expert. |
| **slot** | A place in GPU memory for one expert. The model has 32 slots. |
| **admit** | To put an expert in a slot for one forward pass. |
| **vote** | The sum of the requests of all characters for each expert. |
| **forward pass** | One run of the model on a block of characters. |
| **trunk** | The part of the model that all characters use. It is not the pool. |
| **checkpoint** | A copy of the model that the program saves on disk. |
| **GPU memory** | The memory of the graphics card. It is also named VRAM. |
| **RAM cache** | A store of 96 experts in computer memory. |

---

## 2. Cautions

> **CAUTION**
> Do not use `--weights-dir weights` for your own training tests.
> The folder `weights` holds the snapshot of the author. Training changes the files in the folder.

> **CAUTION**
> Do not use the command `pkill -f` with a long pattern in the terminal.
> It can stop your own shell.

> **CAUTION**
> The chat server changes the model files if you do not use `--no-learn`.
> Make a copy of the folder before you start the server without this option.

> **CAUTION**
> Do not stop a training run with `kill -9`.
> Use `kill -INT <pid>`. The program saves a checkpoint, then it stops.

> **NOTE**
> Port 8080 was in use on the test computer. The tests used port 8765.

---

## 3. System description

### 3.1 Purpose

mini-AGI is a small language model. It reads text one character at a time.
It learns from the text while it reads. Reading and learning are the same action.

The model has no tokenizer. It does not stop learning. It is a research toy and not a real AGI.

### 3.2 Main idea

The design uses four main ideas.

1. The model uses bytes as characters. It can read any file.
2. The model uses one block again and again. Each character chooses how many rows to use.
3. The model keeps most parameters on disk. Only 32 experts are in GPU memory at one time.
4. The trunk learns at one tenth of the speed of the pool. This protects old knowledge.

### 3.3 The three levels of storage

The experts are in three places.

| Level | Content | Speed |
|-------|---------|-------|
| Disk | All experts. One file for each expert, 25 MB each. | Slow |
| RAM cache | The last 96 experts that the program used. | Medium |
| GPU memory | The 32 experts in the slots. | Fast |

### 3.4 Main parts

| Part | File | Function |
|------|------|----------|
| Settings | `config.yaml` | Holds the model size and the training settings. |
| Training program | `train.py` | Reads text and trains the model. |
| Chat program | `serve.py` | Shows a chat page and writes replies. |
| Corpus builder | `corpora/` | Makes the training text. |
| Model | `minagi/recur.py`, `minagi/model.py` | Calculates the next character. |
| Pool | `minagi/pool.py` | Chooses the experts for each character. |
| Paging | `minagi/paged.py` | Moves experts between disk, RAM and GPU memory. |
| Learning-rate controller | `minagi/plasticity.py` | Changes the learning rate. |
| Weights store | `minagi/store.py` | Reads and writes the folder `weights/`. |

---

## 4. Data

### 4.1 Lanes

The builder makes eight lanes. The training program gives each lane the same reading time.

| Lane | Source | Content |
|------|--------|---------|
| `wikipedia` | Hugging Face | Wikipedia text in MediaWiki form |
| `stories` | TinyStories | Short stories for children |
| `chat` | OpenHermes and generated text | Messages between `<user>` and `<bot>` |
| `reasoning` | OpenThoughts | Reasoning in `<think>` markers |
| `arithmetic` | Generator | Sums with the steps shown |
| `code` | Python files on your computer | Python code and docstrings |
| `chess` | Lichess | Chess games in `<g>` markers |
| `self-knowledge` | `corpora/self_knowledge.yaml` | Questions and answers about the model |

### 4.2 Build steps

1. The command `python3 -m corpora all` starts the builder.
2. The builder starts one process for each lane.
3. A lane that has files is skipped. The option `--force` builds it again.
4. The builder keeps some records for the folder `data/val`.
5. The builder writes the rest in files of about 200,000 characters.
6. The builder prints the size of each lane.

### 4.3 Known faults of the build

- The lane `self-knowledge` makes no files. The code writes its output in `data/train/chat`.
- The option `--limit 5000` makes lanes of unequal size. The lane `stories` has 16 KB of training text.

---

## 5. Characters

1. The program changes each byte of the text to a number from 0 to 255.
2. The program changes each marker to a number from 256 to 264.
3. The vocabulary has 265 characters.
4. No unknown character exists.
5. A file of 1 KB gives about 1,000 characters.

The program skips binary files. It tests a sample of the file and does not use the file name.

---

## 6. Model

### 6.1 Order of operations for one character

1. The embedding changes the character to a vector of 512 numbers.
2. Prelude block 1 processes the vector.
3. Prelude block 2 processes the vector. The result is `x`.
4. The state `h` starts at zero.
5. The adapter joins `h` and `x`. The adapter adds `x` again in each row.
6. The recurrent block does attention. Attention uses the earlier characters.
7. The recurrent block uses 8 experts from the router.
8. The model makes a new state `h`.
9. The model calculates the scores for the next character.
10. The halting head calculates the probability to stop.
11. If the character does not stop, the model goes to step 5. The limit is 24 rows.

### 6.2 Size of the model

| Item | Value |
|------|-------|
| Vocabulary | 265 |
| State size | 512 |
| Attention heads | 8, with 64 numbers each |
| Prelude blocks | 2 |
| Recurrent blocks | 1 |
| Maximum rows | 24 |
| Position method | RoPE. The model has no learned position numbers. |
| Context length | 2,048 at the start. It grows to 4,096. |
| Parameters in one expert | 3,145,728 |
| Parameters, total | 1,257 million |
| Parameters in GPU memory | about 100 million |
| Active parameters for one character | about 345 million |

### 6.3 Halting

The halting head gives a probability to stop in each row.
The model uses these probabilities in two ways.

- **In training:** The loss is the sum of the loss of each row. Each row has a weight. The weight is the probability that the character stops in this row.
- **In use:** The character stops when the sum of the probabilities reaches 0.9.

The training run also chooses the number of rows at random. The mean is 13.8 rows.
A penalty keeps the mean number of stopped rows near 13.9.

### 6.4 Frozen characters

When a character stops, it is frozen.

1. The state of the character does not change.
2. The character does not ask for experts.
3. Later characters read its final state in the deeper rows.

This rule applies in training and in writing.

---

## 7. Experts and paging

### 7.1 Expert

Each expert has three weight matrices: `w1`, `w3` and `w2`.
The size of the expert is 512 to 2048 to 512.
No person gives a subject to an expert. The specialization is automatic.

### 7.2 Choice of experts

The forward pass uses these steps.

1. Each character gives a score to all experts in the pool.
2. Each character asks for its 8 best experts.
3. The program adds all requests. The sum is the vote.
4. The program admits the 32 experts with the most votes.
5. The program loads each missing expert. It uses an empty slot first.
6. If no slot is empty, the program uses the slot of the oldest admitted expert.
7. Each character then uses only the admitted experts.
8. A character that has no admitted choice uses its best admitted expert.

The text makes the choice. The earlier content of the slots does not change the choice.

### 7.3 Rules for the optimizer

1. The Adam moments belong to the expert. They do not belong to the slot.
2. An expert that is in a slot stays in that slot.
3. The program saves an expert only if the optimizer changed it.

### 7.4 Balance term

The loss includes a small balance term. The term pushes the router to use all experts.
The term trains only the router. Its weight is 0.000035.

---

## 8. Training

### 8.1 Command

```bash
python3 train.py read data/train --save --weights-dir weights_mine \
    --held-out data/val --sample-every 10
```

### 8.2 Start of the run

1. The program finds the training files.
2. If the folder has no `core.npz`, the program makes a new model from `config.yaml`.
3. The program loads the trunk and the router.
4. The program makes the optimizer. The trunk uses 0.1 times the learning rate.
5. The program measures the held-out loss. This value is the "before" value.
6. The program loads the saved state, if it exists.
7. The program makes one lane for each folder in `data/train`.

### 8.3 One training step

1. The program chooses the next lane. It chooses a random file and a random position.
2. The program reads a chunk of 2,048 characters.
3. The program takes the last 4,096 characters at most. This is the window.
4. The model runs a forward pass on the window.
5. The program calculates the loss.
6. The program calculates the gradients.
7. The program limits the gradient size to 1.0.
8. The program puts the moments of each admitted expert in its slot.
9. The optimizer changes the parameters.
10. The program repeats from step 2. After 16 chunks, it goes to the next lane.

### 8.4 Loss

The loss has four parts.

- The expected loss over the rows (weighted by the halting probabilities)
- A penalty of weight 0.01 for the difference from the prior of the stopping rows
- A router term of weight 0.01
- The balance term

### 8.5 Settings

| Setting | Value |
|---------|-------|
| Optimizer | AdamW, betas 0.9 and 0.95 |
| Learning rate | 0.0003 |
| Trunk multiplier | 0.1 |
| Weight decay | 0.1 |
| Gradient limit | 1.0 |
| Number format | bf16 (weights in fp32) |
| Chunk | 2,048 characters |
| Passage for each visit | 32,768 characters |
| Checkpoint interval | 5 minutes |

### 8.6 Tasks during the run

| Event | Action |
|-------|--------|
| Each `--sample-every` minutes | The program measures the held-out loss, writes 9 samples and saves a checkpoint. |
| Each 5 minutes | The program saves a checkpoint. |
| Each 100,000 characters | The context grows by 1 character, if the gain is more than 0.015. |
| Each 2 million characters | The program can add 1 expert. It can delete old experts. |
| Ctrl-C | The program saves a checkpoint and stops. |

### 8.7 Resume

1. Run the same command again.
2. The program reads `manifest.json`.
3. The program continues from the saved step.
4. The random choice uses the number of characters read. The program does not read the same text again.

---

## 9. Learning-rate controller

The controller changes the learning rate after each held-out test.
There is no planned end of training, so the controller does not use a fixed plan.

1. The controller makes two fits of the held-out loss. One fit is slow and one fit is fast.
2. The controller calculates the evidence that the loss goes down.
3. If the evidence is low, the controller lowers the rate a small amount.
4. If the loss clearly goes down, the controller holds the rate or raises it a small amount.
5. The rate cannot go below 0.01 times the learning rate. It cannot go above 1.0.
6. If the loss jumps up and stays up, the controller multiplies the rate by 2. It also resets the fits.

The controller ignores a single reading that is very far from the fit.
It also re-levels its fits for the first five readings after a restart.

---

## 10. Protection of old knowledge

The author tested a model on a new subject (PG19 novels).
The model read 1,048,576 characters of this subject only.

| Test | Change on new subject | Change on 8 old subjects |
|------|----------------------|-------------------------|
| Experts frozen, trunk rate equal | −0.291 | +1.266 |
| Experts swapped, trunk rate equal | −0.311 | +1.263 |
| **Experts swapped, trunk rate 0.1** | **−0.422** | **+0.130** |
| New subject as a ninth lane | −0.312 | −0.0065 |

A negative number is a gain. A positive number on the old subjects is forgetting.
The slow trunk rate gives about ten times less forgetting.
Only 44 of 174 experts received a gradient in the test.

---

## 11. Growth and pruning

### 11.1 Growth

The program tests for growth each 2 million characters.
The program adds 1 expert only if all five conditions are true.

| Condition | Rule |
|-----------|------|
| ROOM | The disk and the GPU memory have free space. The disk limit is 10 GB. |
| USED | Less than 35 percent of the experts are dying. |
| EARNING | At least 35 percent of the new experts still get requests. |
| FITS | No more than 50 experts are in their trial. |
| HONEST | The held-out loss minus the training loss is less than 0.4. |

The new expert takes each hidden unit from a random one of 16 parent experts.
The router row of the new expert is the mean of the parent rows. The start gate is 0.001.

### 11.2 Pruning

The program deletes an expert if no forward pass admits it for 100 million characters.
A new expert has the same trial time. The program does not use the gate value for this test.

---

## 12. Weights folder

| File | Content |
|------|---------|
| `manifest.json` | Settings, step, counters, learning-rate state |
| `core.npz` | Embedding, attention, norms, adapter, halting head |
| `routers.npz` | Router rows, depth embedding, gates |
| `optim.npz` | Adam moments of the trunk and the router |
| `experts/eNNNNN.npz` | One expert, with its Adam moments |

The program writes each file to a `.tmp` name first. Then the program renames the file.
A crash cannot leave a half file. The weights are in fp32. The Adam moments are in bf16.

---

## 13. Chat server

### 13.1 Reply cycle

1. The user sends a message.
2. The program makes the prompt: `<user>`, the message, `</user>`, `<bot>`.
3. The program keeps the end of the prompt if the prompt is too long. The limit is 90 percent of the context.
4. The program reads the prompt in chunks of 512 characters. It fills the cache.
5. The program makes one forward pass for each new character.
6. The program chooses the character with the highest score. It uses the adaptation trace.
7. The program sends each character to the browser.
8. The program stops at `</bot>` or at the limit of 400 characters.

### 13.2 Adaptation trace

The program lowers the score of each character that it used lately.
The penalty is 2.5 for one use. The penalty falls by the factor 0.88 for each new character.
The trace stops loops. The program uses no random numbers.

### 13.3 Live learning

If you start the server without `--no-learn`, the model learns from the chat.
The program saves the model after 8 learning steps.

---

## 14. Procedures

### 14.1 Prepare the computer

1. Install `uv`.
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   export PATH="$HOME/.local/bin:$PATH"
   ```
2. Go to the project folder.
   ```bash
   cd ~/Downloads/mini-AGI
   ```
3. Make the virtual environment.
   ```bash
   uv venv --python 3.10 .venv
   ```
4. Install the packages.
   ```bash
   uv pip install --python .venv/bin/python torch numpy pyyaml matplotlib flask scipy chess zstandard datasets huggingface_hub
   ```
5. Make sure that the GPU is free.
6. Run the device check.
   ```bash
   .venv/bin/python tools/device_check.py
   ```

> **NOTE**
> The check stops if another program uses most of the GPU memory. Use `--force` to ignore this.

### 14.2 Update the code

1. Get the new code.
   ```bash
   git pull --ff-only
   ```

### 14.3 Build the training text

1. Build a small test set.
   ```bash
   .venv/bin/python -m corpora all --limit 5000
   ```
2. Wait for the end of the build. The test build takes some minutes.
3. Make sure that the folders `data/train` and `data/val` have files.
4. For a full training run, use `python3 -m corpora all` without `--limit`.

### 14.4 Get the author's weights

1. Download the snapshot (403 files, 9.8 GiB).
   ```bash
   .venv/bin/python - <<'EOF'
   from huggingface_hub import snapshot_download
   snapshot_download('Volotat/mini-AGI-undertrained', local_dir='hf_snapshot', max_workers=8)
   EOF
   ```
2. Move the folder.
   ```bash
   mv hf_snapshot/weights weights
   ```
3. Look at the table of parameters.
   ```bash
   .venv/bin/python -m minagi.store weights
   ```
4. Make sure that the table shows 397 experts.

### 14.5 Start the chat server

1. Start the server without learning.
   ```bash
   .venv/bin/python serve.py --weights weights --no-learn --prime-chars 0 --port 8765
   ```
2. Wait for the message `Running on http://127.0.0.1:8765`.
3. Open this address in a browser.
4. Type a question and press Enter.
5. To stop the server, press Ctrl-C in the terminal.

### 14.6 Train your own model

1. Make sure that the GPU has free memory. The test run used about 4 GB for the pool.
2. Start the run.
   ```bash
   .venv/bin/python train.py read data/train --save --weights-dir weights_mine \
       --held-out data/val --sample-every 10 --sample-log runs/my_samples.txt
   ```
3. Read the line `before:`. This is the first held-out loss.
4. Wait. The program prints a block after each sample interval.
5. To stop, press Ctrl-C one time.
6. Wait for the line `weights/ checkpointed`.

To run in the background, use `nohup` and write the output to a log file.

### 14.7 Monitor a run

1. Look at the last lines of the log.
2. Find the line `held-out loss`. A lower value is better.
3. Find the line `reading ... char/s`. This is the speed.
4. Open the images `runs/training_progress.png` and `runs/dashboard.png`.
5. If the held-out loss does not go down for a long time, wait. The controller lowers the learning rate.

### 14.8 Teach the model your own files

1. Do a dry run. The program does not save.
   ```bash
   .venv/bin/python train.py read ~/notes
   ```
2. Read the held-out loss before and after.
3. Train and save with a low learning rate.
   ```bash
   .venv/bin/python train.py read ~/notes --passes 3 --save --weights-dir weights_mine --lr 5e-5
   ```

---

## 15. Results of the tests

| Test | Result |
|------|--------|
| Environment | Python 3.10.12, torch 2.14.1 with CUDA 13.0, RTX 4090 |
| Weights download | 403 files in 17 minutes 54 seconds |
| Snapshot | 397 experts, step 914,508, 1,872,164,864 characters read |
| Chat test | The model gave correct answers about itself. |
| Own training test | The held-out loss went from 5.578 to 2.580 nats in 6 minutes |
| Characters read in the test | 3.58 million, at 9.9 thousand characters each second |
| Experts after the test | 66 (64 at the start) |
| Samples after 6 minutes | Not useful. The text is mostly repeated word parts. |

---

## 16. Fault finding

| Fault | Probable cause | Remedy |
|-------|----------------|--------|
| `.venv/bin/activate` is missing | The virtual environment is not complete | Delete `.venv`. Make it again with `uv venv`. |
| `pip` is not found | The computer has no `pip` | Use `uv pip install --python .venv/bin/python ...`. |
| `PyExtensionType` error in `datasets` | `datasets` is too old for the `pyarrow` version | Run `uv pip install -U datasets`. |
| Device check says the card is busy | Another program uses the GPU | Stop the other program. Or use `--force`. |
| Port 8080 is in use | Another program uses the port | Start the server with `--port 8765`. |
| `self-knowledge` makes no files | The code writes to `data/train/chat` | Use `--prime-chars 0`. Ignore the lane. |
| Out of GPU memory | The card is not free | Stop other programs. Reduce `chunk` in `config.yaml`. |
| The program stops with a `lr_horizon` error | The old keys are in `config.yaml` | Delete `lr_horizon`, `lr_warmup` and `lr_floor` from `training:`. |
| Chat replies are bad | The model is not trained enough | Use the author's snapshot. Or train for a longer time. |
| Samples repeat the same text | The model is at an early stage | Continue to train. Look at the `adapted` samples. |

---

## 17. Command summary

```bash
.venv/bin/python -m corpora all --limit 5000
.venv/bin/python -m minagi.store weights
.venv/bin/python serve.py --weights weights --no-learn --prime-chars 0 --port 8765
.venv/bin/python train.py read data/train --save --weights-dir weights_mine --held-out data/val --sample-every 10
kill -INT <pid>
```

**END OF DOCUMENT**
