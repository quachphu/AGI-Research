# mini-AGI — the full explanation, from scratch to end

> Written on 12 Oct 2026, for repository commit `60f7ae3` (author: Alexey Borsky,
> <https://github.com/volotat/mini-AGI>). Everything here was checked against the
> code in this folder and against a real run on your machine (RTX 4090).
>
> Companion files in this folder:
> - `ASD-STE100 workflow and architecture.md` – the same story in Simplified Technical English
> - `diagram/` – Excalidraw drawing, Mermaid diagrams (`diagrams.md` + ready-made PNGs in `diagram/png/`) and an **interactive HTML** (`diagram/architecture-interactive.html`, just double-click it)

---

## Table of contents

0. [How to read this](#0-how-to-read-this)
1. [The big idea in plain words](#1-the-big-idea-in-plain-words)
2. [A tiny dictionary](#2-a-tiny-dictionary)
3. [Map of the project](#3-map-of-the-project)
4. [Step 0 – Set up the machine](#4-step-0--set-up-the-machine)
5. [Step 1 – Build the data](#5-step-1--build-the-data)
6. [Step 2 – Turn text into numbers (bytes)](#6-step-2--turn-text-into-numbers-bytes)
7. [Step 3 – The model, one character at a time](#7-step-3--the-model-one-character-at-a-time)
8. [Step 4 – Experts and paging (the clever part)](#8-step-4--experts-and-paging-the-clever-part)
9. [Step 5 – The training loop, line by line](#9-step-5--the-training-loop-line-by-line)
10. [Step 6 – The learning-rate controller](#10-step-6--the-learning-rate-controller)
11. [Step 7 – Not forgetting (continual learning)](#11-step-7--not-forgetting-continual-learning)
12. [Step 8 – Growing and pruning experts](#12-step-8--growing-and-pruning-experts)
13. [Step 9 – The weights folder is the model](#13-step-9--the-weights-folder-is-the-model)
14. [Step 10 – Talking to the model (serve.py)](#14-step-10--talking-to-the-model-servepy)
15. [Step 11 – Teaching it your own files](#15-step-11--teaching-it-your-own-files)
16. [What happened when we ran it](#16-what-happened-when-we-ran-it)
17. [How to read the training log](#17-how-to-read-the-training-log)
18. [Config reference (`config.yaml`)](#18-config-reference-configyaml)
19. [Known quirks and honest limits](#19-known-quirks-and-honest-limits)
20. [Command cheat sheet](#20-command-cheat-sheet)
21. [FAQ](#21-faq)

---

## 0. How to read this

Each section answers three questions:

- **What** is this part?
- **Why** does the author do it this way?
- **Where** is it in the code?

If a word is new, look in [section 2](#2-a-tiny-dictionary). Numbers such as "397 experts" or
"32 slots" are real values from the model that the author published on 12 Oct 2026 and that
we downloaded.

---

## 1. The big idea in plain words

### What it is

**mini-AGI is a small language model that never stops learning.** You give it text. It reads
the text one character at a time. After every small chunk of text it adjusts itself a little.
Reading and learning are the *same* action.

### What makes it unusual

A normal chat model (GPT, Llama, …) is trained once, then **frozen**. If you train it more on
new data, it tends to **forget** old skills. People call this *catastrophic forgetting*.

mini-AGI tries to avoid that, and to do it on **one ordinary gaming GPU (8 GB)**. To make that
possible, it uses four ideas:

| # | Idea | In one sentence |
|---|------|-----------------|
| 1 | **Bytes, not words** | The "alphabet" is just the 256 byte values, so it can read any file. |
| 2 | **A loop with adaptive depth** | One block of layers is reused up to 24 times; easy characters stop early, hard ones think longer. |
| 3 | **A big pool of small experts stored on disk** | The model has about 1.25 billion parameters, but only about 100 million sit on the GPU at once. |
| 4 | **A slow, careful learning rate for the shared part** | The part every character uses learns at 0.1× speed, which protects old knowledge. |

### What it is NOT

- It is **not** a real AGI. The name is a half-joke (the README says so).
- It is **not** as smart as ChatGPT. It is a "toy-level" model. Its answers about itself are good
  (it was trained on text about itself), but code, arithmetic and Wikipedia samples are weak.
- It is **not finished**. The author is still training it and updating the repo every day or two.

### Analogy: a library, a shelf and a desk

Imagine a huge library (**disk**) with 397 small notebooks (the experts).
You cannot carry all of them to your desk. So you keep:

- the **warehouse** = disk: all 397 notebooks, always safe,
- a **shelf** nearby = RAM: the 96 notebooks you used lately,
- your **desk** = GPU memory (VRAM): only 32 notebooks, the ones you need *right now*.

Before you read a page, you check which notebooks the page needs most, and you fetch those 32
to your desk. That is *paging*. Almost every design choice in the repo supports this trick.

---

## 2. A tiny dictionary

| Word | Meaning here |
|------|--------------|
| **Byte** | A number from 0 to 255. One English letter is one byte. Some symbols (é, 你) are 2–4 bytes. |
| **Token** | What the model reads in one step. Here: one byte, or one of 9 special markers. |
| **Parameter** | One learned number inside the model. This model has about 1,257 million. |
| **Layer / block** | A small calculation unit: attention followed by a feed-forward network. |
| **Attention** | Lets each character look back at earlier characters. |
| **RoPE (rotary position)** | A way to tell attention *where* each character is, with no learned numbers. |
| **SwiGLU** | A common feed-forward network shape (`w1`, `w3`, `w2` matrices). |
| **MoE (mixture of experts)** | Many small feed-forward networks (experts); a *router* picks a few for each character. |
| **Expert** | One SwiGLU network: 512 → 2048 → 512, about 3.15 million parameters, stored in one file. |
| **Router** | A small linear layer that scores every expert for each character. |
| **Top-k** | Take the k best-scoring items. Here k = 8 experts per character per row. |
| **Resident** | "Currently on the GPU." There are 32 resident slots. |
| **VRAM / RAM / disk** | GPU memory (small, fast) / computer memory / hard drive (big, slow). |
| **Gradient** | The direction that makes the loss smaller. Computed by *backward*. |
| **Loss** | How wrong the model is. Lower is better. Measured in *nats* per character. |
| **Nat / bit** | Units of information. 1 nat = 1.443 bits. Bits per byte = nats × 1.443. |
| **Perplexity** | `e^loss`. "On average the model is as unsure as choosing between N characters." |
| **Held-out set** | Text the model never trains on. Used to measure real skill. |
| **Adam / AdamW** | The optimizer. It remembers two numbers per parameter ("moments") to take smart steps. |
| **Learning rate (lr)** | How big each step is. Too big = unstable; too small = slow. |
| **Gradient clipping** | Cap the size of a step (here, at 1.0) so it cannot explode. |
| **bf16** | A 16-bit number format. Faster and smaller than 32-bit (fp32). Native on your 4090. |
| **KV cache** | Stored "keys and values" of past characters so attention need not recompute them. |
| **Halting (PonderNet)** | A mechanism that lets each character decide *when to stop thinking*. |
| **Catastrophic forgetting** | New learning wipes out old knowledge. |
| **Continual learning** | Learning forever from a stream without forgetting. |
| **Checkpoint** | A saved copy of the model on disk. |
| **Greedy decoding** | Always pick the most likely next character (no randomness). |

---

## 3. Map of the project

```
mini-AGI/
├── config.yaml            ← the settings worth changing (model shape, pool, training, growth…)
├── train.py               ← the main program: `read` (continual training), `stream`, `ponder-probe`
├── serve.py               ← a small chat web page (Flask)
├── minagi/                ← the model, as a library (no command lines)
│   ├── config.py          ← reads config.yaml
│   ├── precision.py       ← bf16/fp32 choice, how Adam moments are stored
│   ├── tokenizer.py       ← bytes in, bytes out (+9 markers)
│   ├── model.py           ← RMSNorm, RoPE, attention (flash / blocked), SwiGLU, Block
│   ├── decode.py          ← picking the next character (greedy + anti-loop trace)
│   ├── ingest.py          ← finding text files and turning them into streams
│   ├── pool.py            ← the expert pool: routing, dispatch, AutoGrow brakes
│   ├── paged.py           ← disk / RAM / VRAM tiers, admission, growth, pruning
│   ├── recur.py           ← the whole model: recurrence + adaptive depth + generate()
│   ├── stream.py          ← FileReader, held-out evaluators
│   ├── store.py           ← the weights/ folder format (save / load)
│   ├── optim.py           ← a gradient "signal meter" (only reports)
│   ├── plasticity.py      ← the learning-rate controller
│   ├── live.py            ← learning while you chat
│   ├── report.py          ← statistics from the weights
│   └── create.py          ← builds a fresh weights/ folder from config.yaml
├── corpora/               ← `python3 -m corpora all` builds the training text
├── replication/           ← the forgetting experiment and its figures
├── tools/                 ← device_check.py, plotting scripts
├── runs/samples.txt       ← the author's training history (very large)
└── README.md
```

Folders created by *running* things (none are in git):

| Folder | Created by | Contents |
|--------|-----------|----------|
| `data/train`, `data/val` | `corpora` | text files, one sub-folder per subject ("lane") |
| `data_char`, `data_chat_char`, … | `corpora` | intermediate packed `.bin` files |
| `weights/` | you (download) or `train.py` | the model |
| `runs/` | `train.py` | sample logs and graphs |

---

## 4. Step 0 – Set up the machine

**Goal:** get Python, PyTorch and the other packages working on your GPU.

### What we found on your machine

- Python 3.10.12, no `pip`, no `ensurepip` (the package `python3-venv` is not installed).
- RTX 4090 with 24 GB, driver 580.178.04, 62 GB RAM, 500 GB free disk.
- The `.venv` folder from before had only symlinks and no `activate` script – it was broken.

### What we did

1. Installed **uv** (a fast Python package manager that does not need `pip`).
2. Deleted the broken `.venv` and created a new one: `uv venv --python 3.10 .venv`.
3. Installed: `torch numpy pyyaml matplotlib flask scipy chess zstandard datasets huggingface_hub`.
4. Hit one bug: `datasets 2.14.4` did not work with the new `pyarrow 25`. Upgrading `datasets`
   (to 5.1.0) fixed it.
5. Ran `tools/device_check.py`.

### What we confirmed

```
torch 2.14.1+cu130 (CUDA 13.0), device: NVIDIA GeForce RTX 4090, driver 580.178.04
```

`tools/device_check.py` refuses to run when the card is busy. Another program on your machine
was using about 21 GB of the 24 GB when we checked, so it printed
`the card has 0.3 of 23.5 GiB free - something else is using it` (add `--force` to override).
We did not need it: the real training test (section 16.4) ran fine and reported
`vram 2067 MB` and `vram 3826 MB` for the pool.

---

## 5. Step 1 – Build the data

**Command:** `python3 -m corpora all` (we used `--limit 5000` for a small test).

### Why eight "lanes"?

The model reads *subjects* in turns, like a student switching between school subjects. Mixing
subjects is the main protection against forgetting. Each subject lives in its own folder
(a **lane**) and gets an **equal share** of reading time, no matter how big it is.

| Lane | Where the text comes from | Looks like |
|------|--------------------------|-----------|
| `wikipedia` | Hugging Face `wikimedia/wikipedia` (20231101.en) | raw MediaWiki markup, `[[links]]` |
| `stories` | `roneneldan/TinyStories` | simple children's stories |
| `chat` | `teknium/OpenHermes-2.5` + generated chat | `<user>…</user><bot>…</bot>` |
| `reasoning` | `open-thoughts/OpenThoughts-114k` | long `<think>…</think>` traces |
| `arithmetic` | synthetic generator | `add 417 + 1 = 418`, with scratch-pad steps |
| `code` | **Python files on your own computer** (stdlib + site-packages) | docstring/implementation pairs |
| `chess` | Lichess database (streamed) | `<g>1700 1-0 1. e4 e5 …</g>` |
| `self-knowledge` | generated from `corpora/self_knowledge.yaml` | the model's own Q&A about itself |

### How the build works

`corpora/build.py` runs each lane **in its own subprocess**, so one failure does not stop the
others. A lane that already has files is skipped (use `--force` to rebuild).

- **Downloaded lanes** (`fetch.py`, `reasoning.py`) stream a dataset from Hugging Face, hold some
  records back for the validation set, and write text shards.
- **Generated lanes** (`arithmetic.py`, `code.py`, `chess_games.py`, `chat.py`) write packed
  `uint16` `.bin` files into `data_*` folders. Then `expand.py` converts them to text shards
  (~200,000 characters per file) under `data/train/<lane>/` and `data/val/<lane>/`.

Why many small files? Training opens one *random file at a random offset* per visit. One giant
file would be one long window with no natural boundaries.

### Special markers

Text such as `<user>`, `</user>`, `<bot>`, `<think>`, `<g>` is converted to **one special token**
instead of being spelled out in letters (see Step 2). `expand.py` decodes `.bin` back to text
*through the tokenizer* so those markers survive the round trip.

### What we got (small test build)

Training text under `data/train/` (measured with `du` after the build):

```
chat         2,215 files   425 MB   (generated chat)
reasoning       10 files   147 MB
wikipedia        1 file     56 MB
chess          633 files   122 MB
code           541 files   104 MB
arithmetic       1 file    188 KB
stories          1 file     16 KB   (quirk, see section 19)
self-knowledge   –          nothing was written here (quirk)
```

The build summary printed `TOTAL 434 MB` for the lanes it counted and the line
`lanes that produced nothing: self-knowledge`. The held-out text under `data/val/` is much
smaller than the training text for most lanes (for example `stories`: 4.7 MB of validation text
against 16 KB of training text).

---

## 6. Step 2 – Turn text into numbers (bytes)

File: `minagi/tokenizer.py` and `minagi/ingest.py`.

A computer cannot read letters, only numbers. Most models use a *tokenizer* that chops text
into word pieces (a vocabulary of 32,000+ pieces) and must be **trained first**.

mini-AGI skips that:

- **Ids 0–255** = the raw UTF-8 bytes. The letter `A` is id 65. There is no "unknown" character.
- **Ids 256–264** = nine **markers**: `<think>`, `</think>`, `<user>`, `</user>`, `<bot>`, `</bot>`,
  `<g>`, `</g>`, `<|endoftext|>`.
- Vocabulary size = **265**.

`ingest.collect()` walks folders, skips binaries (images, zips, …) by *sampling the content*
rather than trusting the extension, and caches its answer in `runs/corpus_index.json` because
opening half a million files takes minutes. `ingest.as_stream(path)` turns one file into a NumPy
`uint16` array of ids.

**Trade-off:** one byte is a tiny piece of text, so sequences are long (a 4,096 window is only
~4 KB of text). That is why the model needs a *lot* of compute per character.

---

## 7. Step 3 – The model, one character at a time

Files: `minagi/recur.py` (the model), `minagi/model.py` (the building blocks).

### 7.1 The picture

```
bytes ──► Embedding ──► Prelude 1 ──► Prelude 2 ──► x  (the text, encoded)
                                                    │
        h = 0 (latent state) ───────────────┐       │
                                            ▼       ▼
                                  Adapter: Linear([h ‖ x])
                                            │
                                            ▼
              ┌──────────►  Recurrent block (attention + MoE)  ──► new h
              │                                                    │
              │                                  RMSNorm → heads:  │
              │                         next-byte logits  +  halt probability λ
              │                                                    │
              └──────────── not halted? repeat (up to 24 rows) ◄───┘
```

### 7.2 Embedding

`nn.Embedding(265, 512)`. Each id becomes a vector of 512 numbers. The same table is used at the
end to turn a vector back into scores for the 265 possible next tokens ("tied embeddings"),
which saves parameters.

### 7.3 The two prelude blocks

Each block is the classic transformer recipe:

```
x = x + Attention(RMSNorm(x))
x = x + SwiGLU(RMSNorm(x))
```

- **RMSNorm** keeps numbers at a stable size.
- **Attention** has 8 heads of 64 numbers. Positions use **RoPE**, so there are *no learned
  position parameters*. That is why the context window can grow later "for free".
- **SwiGLU** (dense) has hidden size 1408.
- The two prelude blocks run **once** per character.

The output is called `x`: "what the text says".

### 7.4 The adapter and the latent state `h`

`h` is a hidden "thought" vector for each character, starting at **zero**.

The **adapter** is a `Linear(1024 → 512)` that takes `[h ‖ x]` (both side-by-side) and produces
the input to the recurrent block. It starts as `[I | I]`, meaning "output = h + x". Effect:
**the original text is injected again on every row**, so the loop cannot drift away from what is
being read. The latent state is *never decoded into a character between rows*.

### 7.5 The recurrent block (one block, reused)

This is a single block that is **applied again and again** with the same weights. One application
is called a **row**. A row is:

1. attention (with the KV cache of earlier characters),
2. a **mixture of experts** instead of a dense feed-forward (see Step 4):
   the router scores all experts for this character, the model takes the **top 8** of the
   32 that are on the GPU, mixes their outputs with weights, and adds the result back.

Depth in numbers: 2 prelude + up to 24 recurrent rows = **up to 26 block applications** per
character, while only **3 distinct blocks** exist. The model is "deep" without storing 26 sets of
weights.

### 7.6 Halting – each character decides when to stop (PonderNet)

After every row the model computes:

- `logits_n` – scores for the next byte, using the current state, and
- `λ_n = sigmoid(halt(state))` – the probability to **stop now**.

The chance to stop exactly at row *n* is `p_n = (probability not stopped before) × λ_n`.
At the last row `λ` is forced to 1. At the end:

- **Training loss** = `Σ p_n × CrossEntropy(logits_n)` – the expected loss under the stopping
  distribution. Because `p_n` depends on `λ`, the halting head learns by gradient.
- **KL penalty** pulls the stopping distribution toward a geometric prior with mean
  ≈ `1/0.072 ≈ 13.9` rows, weight `ponder_beta = 0.01`, so the model does not always run to 24.
- **Inference** stops at the first row whose cumulative stop mass reaches `halt_thresh = 0.9`.

While **training**, the number of rows run per step is also *sampled*: `Poisson(12.8) + 1`,
clipped to 1…24 (average ≈ 13.8 rows). This saves compute and still shows the model deep rows.

### 7.7 Freeze and carry (`halt_freeze: true`)

Once a character has halted, it is **finished**:

- its state stops changing,
- its experts stop running (it asks for no more experts, so it costs nothing),
- **later characters still read its final state** at the deeper rows (through its stored keys and
  values).

This is used in *both* training and writing so that what the model learns is what it writes with.
It requires `n_recur = 1` and `n_coda = 0`.

### 7.8 Gradient reach

Only the last `bptt_window = 24` rows carry gradient back through the recurrence (with
`max_steps = 24` that is *all* rows). Setting it lower would save memory by detaching early rows.

### 7.9 Counting the parameters (check it yourself)

```
one expert   = 3 × 512 × 2048        = 3,145,728  (≈ 3.15 M)
397 experts  = 397 × 3,145,728       ≈ 1,248.9 M
core         (embedding, attention, norms, adapter, halting head) = 8.27 M
routers      (one row per expert at the one recurrent site, etc.) = 0.20 M
TOTAL                                 ≈ 1,257 M

On the GPU at once: 32 experts × 3.15 M = 100.7 M  (plus the small core)
```

**Cost per character:** about 345 M "active" parameters (README's 6ND estimate) – like a dense
345 M model, not a 1.25 B one.

### 7.10 Attention details that matter

- With a KV cache, a new chunk of T queries sits at positions P…P+T-1 against P+T keys.
  The causal mask must be shifted by P (`causal_lower_right`). The code takes care of this.
- Where no fused attention kernel exists (some AMD cards), `blocked_attention` computes the same
  result in blocks with a custom backward so memory stays small.
- On your 4090 the fast **flash / efficient** kernels are used.

---

## 8. Step 4 – Experts and paging (the clever part)

Files: `minagi/pool.py` (routing, dispatch), `minagi/paged.py` (storage tiers, admission).

### 8.1 What is an expert?

A small network: `y = w2( silu(w1 x) ⊙ (w3 x) )` with shapes 512 → 2048 → 512. No labels say
"this is the chess expert". Specialisation emerges (or not) by itself.

### 8.2 The three storage tiers

| Tier | Key in config | Holds | Speed |
|------|---------------|-------|-------|
| Disk | – | **all** experts, one `.npz` file each (~25 MB) | slowest |
| RAM | `ram_cache: 96` | recently used experts, **LRU** (least recently used is dropped) | medium |
| VRAM | `resident: 32` | the experts the current forward may use; fixed tensors `w1,w3,w2` of shape `[32, …]` | fastest |

Why 32? Everything a *training* step uses must stay on the card until its backward pass and the
optimizer step finish, including gradients and Adam moments (about 4× the weights).

### 8.3 "The text chooses" – the selection rule

Every **forward pass** (a training window, a held-out chunk, a prompt, one character of a reply)
does this:

1. **Rank everything.** On the first pass, *every character* scores **all** experts with the router
   (the router always has one row per expert, whether or not it is on the GPU).
2. **Ask.** Each character asks for its top 8, each request carrying the router probability.
3. **Vote.** The requests of all characters are added to the **vote of the text** – counted over
   everything read since position 0.
4. **Admit.** The forward is admitted the **32 most-voted experts** (`PagedPool.admit`).
5. **Place.** Missing ones are loaded: an empty slot first, then the slot of the expert that was
   admitted longest ago.
6. **Route.** Now every character routes **only among the admitted 32**. If an expert it asked for
   was not admitted, it takes its best admitted expert instead.

*While writing a reply*, each new character is its own tiny forward that adds its requests to the
same vote – so the card changes **only when the vote changes**. The README measured that in 15
minutes of training a step loaded on average only 2.4 new experts.

**Important property:** *what was on the card before changes the cost, never the choice.* The
same text always selects the same experts.

### 8.4 Three rules that keep training correct

1. **Adam moments belong to the expert, not the slot.** Before every optimizer step, a hook
   (`_own_moments`) copies each resident expert's *own* moments from RAM or disk into the slot.
   Otherwise a newly loaded expert would be pushed by the *previous* occupant's momentum.
2. **An expert already on the card stays in its slot.** Admission is a set; no needless moves.
3. **An expert that nothing trained is not written back.** On eviction, only experts that an
   optimizer step touched are copied to RAM/disk. Writing a reply loads many experts but writes
   nothing.

### 8.5 How the expert maths runs (dispatch)

For each row and each recurrent call-site (`PooledMLP._route`):

1. compute router probabilities over the 32 admitted experts,
2. `topk` → 8 experts + weights per character, renormalised, multiplied by the expert's **gate**,
3. sort the assignments by expert; run each expert on *only* its own characters
   (`run_exact` – exact-size matmuls, no padding),
4. `index_add_` the results back to the characters.

During **training** this is wrapped in gradient checkpointing (recomputed in backward) to save
memory; during **writing** a faster path (`_few`) runs only the experts actually selected.

`capacity_factor` (once 1.5, now **0**) used to drop assignments past an expert's fair share;
the author turned that off on 10–12 Oct.

### 8.6 The balance term – use the whole pool

Left alone, routers like to use the same few experts. So the loss gets an extra term:

```
balance_loss = balance × ( E × Σ_e share_e × P_e  −  1 )
```

- `P_e` = router's mean probability for expert e over the whole pool,
- `share_e` = expert e's share of **recent training forwards that admitted it** (about the last
  1,000),
- `balance = 3.5e-5` (weakened from 2.5e-4 on 10–12 Oct).

It is zero when use is even, and it only trains the router rows (the character states are
detached). There is also a classic Switch-style auxiliary loss with weight `pool_aux = 0.01`.

---

## 9. Step 5 – The training loop, line by line

Command: `python3 train.py read data/train --save --weights-dir weights_mine --held-out data/val --sample-every 10`

`cmd_read()` in `train.py` does the following.

### 9.1 Startup

1. **Collect files** under the given paths (`ingest.collect`) and print how many.
2. **Create a model if none exists.** The test is "is there a `core.npz`?" (not "does the folder
   exist"). `minagi.create.create()` writes a fresh directory from `config.yaml`:
   64 experts, d_model 512, 24 max rows, 265 vocab.
3. **`build_paged()`** builds the model object, the `PagedPool` (with its tiers), loads the trunk
   from `core.npz` / `routers.npz`, loads telemetry, and reads pool settings from `config.yaml`
   *every time* (`balance`, `capacity_factor`, `select_temperature`) because they are choices
   about *how to train*, not properties of the weights.
4. **Apply depth settings** from config over whatever the checkpoint said, and print warnings if
   the config looks written for another architecture.
5. **Build the optimizer:** AdamW, betas (0.9, 0.95), weight decay 0.1, fused on CUDA, with
   **two groups**:
   - `trunk` (embeddings, attention, norms, adapter, halting head) at `lr × 0.1`,
   - `pool` (expert slots, gates, routers) at `lr` (3e-4).
6. **Held-out "before" score** with `FolderEvaluator` (e.g. 5.578 on our fresh model).
7. **Restore** Adam moments, step counter, learning-rate controller, context size and character
   count from `manifest.json` – so a restart *continues*.
8. **Create `AutoGrow`** and set the survival window for experts.
9. **Lanes** – one per top-level folder in `data/train`.

### 9.2 One training step (this repeats forever)

```
for lane in lanes:                         # round-robin over subjects
    open a random file + random offset     # lane.open(...)
    for j in range(turn):                  # turn = 32,768 / 2,048 = 16 chunks
        1. set lr:  group.lr = base_lr × plasticity.scale
        2. zero gradients
        3. loss = reader.step(learn=True)  # FORWARD
        4. loss.backward()                 # BACKWARD
        5. clip gradient norm to 1.0
        6. optimizer.step()                # moments filled just before (paging hook)
        7. bookkeeping, maybe: sample, checkpoint, grow/prune, context growth
```

**What `FileReader.step` does (step 3):**

- the **window** is the last `context` characters ending at the cursor (up to 4,096),
- it takes the window as input `x` and the same window shifted by one as the target `y`
  ("predict the next character at every position"),
- it **re-forwards the whole window** with no KV cache, so the gradient reaches all of it,
- it adds the auxiliary and balance losses.

So, each step handles 2,048 *new* characters, but the model sees and learns from the previous
2,048 as well (cheap rehearsal). Each character is forwarded ≈ window/chunk ≈ 2× in total.

**The loss in one line:**

```
loss = Σ_rows p_row · CE_row   (PonderNet expected cross-entropy)
     + 0.01 · KL(p ‖ geometric prior)
     + 0.01 · classic router auxiliary
     + balance term
```

### 9.3 Periodic jobs (inside the same loop)

| Trigger | Action | Where |
|---------|--------|-------|
| every `--sample-every` minutes | held-out evaluation per lane; 9 sample prompts, each *raw* and *adapted*; write `runs/*.txt`; redraw plots; **checkpoint** | `sample_now`, `write_samples` |
| every `--save-every` min (5) | checkpoint only | `weights_store.save` |
| every `context_every_chars` (65,536) | measure `context_gain` | by-position loss |
| every `context_grow_every_chars` (100,000) | window `+= 1` character, if gain > 0.015 | |
| every `growth.every_chars` (2M) | `AutoGrow.step` – maybe add 1 expert; prune old ones | `pool.prune`, `grower.step` |
| Ctrl-C or SIGTERM | finish the chunk in flight, checkpoint, exit cleanly | signal handler |

### 9.4 What the sample prompts are

Nine prompts, one per subject (in `SAMPLE_PROMPTS`): a story start, a Python function header, an
`add` problem, two chat questions, a reasoning prompt, a Wikipedia heading, a chess game start,
and "how do you decide which experts to use?". Each is generated greedily for 140 characters in two
ways:

- **raw** – plain greedy, no guard (tells you what the *model* predicts),
- **adapted** – with the anti-repetition trace (see Step 10).

The "repeated 8-grams %" number counts how often an 8-character piece was already seen in the same
text. A good model drives the **raw** number down.

### 9.5 Resume

Run the same command again. `read_chars`, `step`, `plasticity`, `context_now` and the Adam
moments all come from `manifest.json`. The random generator for lane files is seeded with
`[seed, lane, read_chars]` so a restart reads **new** text rather than replaying the same
sequence (the README calls this "the single most expensive bug in this project").

---

## 10. Step 6 – The learning-rate controller

File: `minagi/plasticity.py` (rewritten on 10–12 Oct).

**Problem:** a model that never stops training has no "end", so a cosine schedule (which assumes
an end) is wrong.

**Solution:** watch the **held-out loss** and nudge the rate after every evaluation.

### 10.1 The evidence

Two exponentially-weighted straight-line fits to held-out loss versus evaluation number:

- a **slow** fit (remembers ≈ 66 evaluations),
- a **fast** fit (remembers ≈ 12).

From them it computes `t` (is the downward slope distinguishable from noise?) and `e` (slope in
units of noise). The final verdict is `min(t_slow, 45·e_slow, 45·e_fast)` – each term can only
say "less than you think".

### 10.2 The nudge (new in this version)

```
no evidence    → rate × ≈ 0.9986 per evaluation      (gentle decay)
getting worse  → rate × ≈ 0.996  at most
clearly better → rate × up to 1.0008
the training log states a per-check cap of about 0.4 %
floor: × 0.01 of lr, ceiling: × 1.0
```

So the *resting posture is a slow decay*, and only clear improvement holds the rate up.
(The log line of our run says: "eased at every check unless held-out is clearly improving, by at
most 0.4% a check … floor x0.01".)

### 10.3 Things that are *not* evidence

1. **A reading far from the line** (> 5σ): held back until the next evaluation confirms it.
2. **No reading:** fewer than 32 optimizer steps since the last evaluation.
3. **A restart:** the first 5 readings after restore re-level the fits (so changing
   `capacity_factor` is not mistaken for progress or harm).

### 10.4 Regime change

If held-out jumps by several standard errors (and at least 0.25 nats, `FLOOR_JUMP`) **and stays
up**, the rate is multiplied by 2 (`UP`, never above `CEIL = 1.0`) and the fits are reset:
"new material arrived".

### 10.5 Current state of the author's model

`scale ≈ 0.0515` → effective expert learning rate ≈ 1.5e-5 (3e-4 × 0.0515).

---

## 11. Step 7 – Not forgetting (continual learning)

The author's central experiment (`replication/forgetting_probe.py`):

- Take a trained model (428 M characters read).
- Switch it to **PG19** (19th-century novels), a domain it never saw.
- Make it read **1,048,576 consecutive characters of only PG19** at batch 1 – the *worst case*.
- Measure how much loss rises on the **8 subjects it already knew**.

| Configuration | PG19 | 8 known subjects | Verdict |
|---|---|---|---|
| experts frozen, trunk lr = expert lr | −0.291 | **+1.266** | forgot most of it |
| swapping experts, trunk lr = expert lr | −0.311 | **+1.263** | forgot most of it |
| **swapping, trunk lr = 0.1×** (what the run uses) | **−0.422** | **+0.130** | ~10× less forgetting |
| interleaved (PG19 as a 9th lane) | −0.312 | −0.0065 | nothing forgotten |

**Why the trunk?** The trunk (embeddings, attention, routers, halting head) is shared by every
character and carries ≈ 98 % of the squared gradient norm. The author's finding: the **expert pool
is not what prevents forgetting; the slow trunk learning rate is.** The pool *confines* damage:
only 44 of 174 experts got any gradient in the massed run.

The run itself never reads one subject for a million characters. It reads 32,768 characters of one
subject, then moves on, and returns to the same subject about every 262,144 characters.

Caveat: this is one new domain, one checkpoint, and CUDA runs are not deterministic (about 0.014
nats apart run to run, so treat < 0.03 as noise).

---

## 12. Step 8 – Growing and pruning experts

Files: `minagi/pool.py` (`AutoGrow`), `minagi/paged.py` (`add_experts`, `prune`).

### 12.1 When it grows

Every **2,000,000 characters**, the model considers adding `k = 1` expert. It grows only if
**all** brakes agree:

| Brake | Meaning | Config |
|-------|---------|--------|
| **ROOM** | disk (≤ 10 GB) and VRAM have space | `max_disk_gb`, `mem_frac` |
| **USED** | not too many experts are "dying" (< 35 %) | `dying_frac_max` |
| **EARNING** | recently added experts are still being asked for (≥ 35 %) | `keep_ratio_min` |
| **FITS** | ≤ 50 experts are inside their trial at once | `max_in_flight` |
| **HONEST** | held-out minus train < 0.4 (not memorising) | `max_gap` |

Growth and pruning happen **only with `--save`**, because they create and delete files.

### 12.2 How a new expert is built (recombination)

A "hidden unit" is the triple `(w1[u], w3[u], w2[:,u])`. Units are interchangeable. A new expert
takes each of its 2,048 units from a **random one of 16 parent experts**. Result: novel (clones are
not) yet made of trained parts (random experts are not). Its router row is the parents' rows
averaged, weighted by how many units each gave. It starts at gate `0.001`.

### 12.3 When it dies (pruning)

"**Dead means unaddressed.**" If no forward has admitted an expert for **100,000,000 characters**
(`prune.survival_chars`), it is deleted and its file removed. A newborn is protected for the same
window. The gate is **not** used: the author found the smallest gates belong to the *busiest*
experts, so a gate test is anti-predictive.

### 12.4 Files and ids

Expert files are named by a stable **uid** (`e00123.npz`), not by position, so pruning never
renames files.

---

## 13. Step 9 – The weights folder is the model

File: `minagi/store.py`.

```
weights/
├── manifest.json      what exists, its shape, config, step, counters, plasticity state, telemetry
├── core.npz           embeddings, attention, norms, adapter, halting head (24 tensors)
├── routers.npz        router rows (one per expert), depth embedding, gates
├── optim.npz          Adam moments for the trunk and routers
└── experts/
    ├── e00000.npz     w1, w3, w2  (fp32)  +  w1_m, w3_m, w2_m, w1_v, w3_v, w2_v (bf16 bits)
    └── …
```

Key facts:

- **Weights stay fp32** everywhere (bf16 rounding would erase small updates after many page-outs).
- **Adam moments are stored in bf16** (exponent range is what they need). This saves two thirds
  of each file: ≈ 25 MB per expert.
- **Atomic writes:** every file is written as `.tmp` then renamed, so a crash cannot leave a half
  file.
- The directory holds the **latest saved** state, rewritten at each checkpoint.
  `ls weights/experts | wc -l` is the expert count in the most literal sense (397 for the snapshot).
- A paged model is **never fully loaded** into memory. Loading costs only the trunk plus the
  experts that a forward admits.

---

## 14. Step 10 – Talking to the model (`serve.py`)

Command: `python3 serve.py --weights weights --no-learn --prime-chars 0 --port 8765`

### 14.1 The pieces

- A **Flask** app with three routes: `/` (the chat page), `/api/chat` (POST, streams Server-Sent
  Events), `/api/state` (learning and card status).
- One global **lock**: only one conversation at a time (two would corrupt the shared KV cache).
- `load_any()` → `build_paged()` – the same loading path as training. With `--no-learn` the
  pool is opened **read-only**, so nothing is written back.

### 14.2 What happens when you press Enter

1. **`build_prompt`** wraps the conversation: `<user>\n…\n</user>\n<bot>\n` and keeps the newest
   turns within 90 % of the context (the *end* is kept, because the open `<bot>` asks for the
   reply).
2. **Prefill:** `stream()` reads the prompt in **512-character chunks** into the KV cache. Each chunk
   adds to the vote, so the right experts get admitted.
3. **Generate:** for every new character, one forward pass runs (positions continue in the cache).
   Experts load only when the vote shifts.
4. **`pick_next`**: choose the next byte greedily, after subtracting an **adaptation trace**.
5. Stop when the text ends with `</bot>`, or at `max_new` (400) characters. Characters stream to the
   browser as events (`{"t": "…"}`), along with "swap" events showing which experts moved.
6. If the window fills, the newest half is re-read from position 0 (positions in a cache cannot be
   slid).

### 14.3 The anti-loop trick ("adaptation trace")

Greedy decoding loves loops. Every recently used character gets a penalty that **decays**:

```
logit[c] −= 2.5 × Σ 0.88^(age of each use of c)
```

One use costs 2.5 logits now and about 0.9 eight characters later; a character stuck in a loop
converges to a penalty of 2.5 / (1 − 0.88) ≈ 20.8 logits, which no loop survives. The author
measured repeated 8-grams: greedy alone 73.5 %, +repetition-penalty 59 %, +adaptation 11.6 %.
No random numbers are used, so the same chat gives the same reply (up to GPU non-determinism).

### 14.4 Live learning (off with `--no-learn`)

Without `--no-learn`, `LiveLearner` joins everything you and the model say into one stream and
takes an optimizer step every `chunk` (2,048) characters, then saves every 8 steps.
**This changes `weights/` on disk.** The author warns: chat is a much denser diet than documents,
and the cost of that is not measured.

---

## 15. Step 11 – Teaching it your own files

```bash
python3 train.py read ~/notes                       # dry run: nothing saved
python3 train.py read ~/src ~/docs --passes 3 --save --weights-dir weights_mine
```

- No preparation: any text file is already in the model's "language" (bytes).
- Binaries are skipped by sampling their content.
- Each file is read **start to finish** (a document has an order).
- Default `lr` for `read` comes from `config.yaml` (3e-4 scaled by the plasticity scale); the
  README suggests a lower rate (5e-5) for adjusting rather than overwriting – pass `--lr 5e-5`.
- It prints the held-out score **before and after**, and tells you if reading your files cost
  ground elsewhere.

---

## 16. What happened when we ran it

### 16.1 Update check

`git fetch` showed new commits on 10–12 Oct (for example "Plasticity adjustments + training
process update", "further plasticity adjustments", "training update + new weights snapshot").
The main changes: a rewritten learning-rate controller (`minagi/plasticity.py`), `capacity_factor`
1.5 → 0, `balance` 2.5e-4 → 3.5e-5, and a new weights snapshot. We fast-forwarded to `60f7ae3`.

### 16.2 Published snapshot (12 Oct)

```
weights/  step 914,508   397 experts   9.8 GiB on disk (403 files, ~18 min to download)
core 8.27M · routers 0.20M · experts 1,248.9M · total 1,257.3M
read_chars 1,872,164,864 (23.7 % of a 7.88 B-character corpus)
context 4,096   plasticity scale 0.0515   held-out ≈ 0.619 nats (0.893 bits/char)
```

### 16.3 Chat test (read-only server)

```
Q: What are you?
A: My name is mini-AGI. I am a language model that reads and writes single characters - the 256
   byte values and 9 markers for structure - so I need no tokenizer. Most of what I know lives
   in a pool of 397 small experts kept as files on disk, and I keep learning from everything
   I read.        (283 characters, 45 char/s)
```

```
Q: how do you decide which experts to use?
A: With the router, and by the whole text: one row of numbers per expert, always on the card,
   so an expert on disk is scored exactly like one that is loaded. Every character scores all
   397 and asks for its 8 best, each request weighted by the probability the router gave it,
   and those requests are the t…        (cut at our 300-character limit)
```

Both answers are about the model itself. That subject was in its training text
(`corpora/self_knowledge.yaml`), so it answers well here. On other subjects (code, arithmetic,
Wikipedia) the samples in `runs/samples.txt` are weaker.

### 16.4 Our own 6-minute training test

```
before: 5.5781 ± 0.0096  →  after: 2.5796 ± 0.0227 nats   (change −2.9984)
3.58 M characters read in 6.0 min (9.9 k char/s over the whole session,
                                   10,977 and 12,165 char/s in the two sample blocks)
pool grew 64 → 66 experts, 100.7 M parameters in VRAM, RAM hit rate 0.99
weights_mine/ = 2.0 GiB
```

At 6 minutes the samples are gibberish (loops of "1. 1. 1." in chess, repeated word pieces in
stories). That is normal. The author's run needed 1.87 **billion** characters, about 21 days
of running, to reach its current output. Our fresh model is *learning* (loss −3 nats) but is
nowhere near useful.

### 16.5 Reading speed

We saw about 10–12 k char/s on the RTX 4090 with a 66-expert pool. The README reports about
1.2–1.4 k char/s on the author's laptop GPU with a 397-expert pool. The GPUs differ, and a
bigger pool means more paging, so the two numbers are not a like-for-like comparison.

---

## 17. How to read the training log

This is a **real block** from our own 6-minute run (`runs/my_samples.txt`):

```
step 642   1.3M of 876M characters (0.15%)   2 min   65 experts
context 2,060 characters of 4,096   reading 10,977 char/s   writing 99.7 char/s
grad norm 4.96 against a clip of 1   clipping
train loss 2.7249   lr 3.00e-04   evidence t +0.00 over 0.0 (effect +0.0000)   rate x1.000
held-out loss 2.7508 +/-0.0499 nats   3.9685 bits/char   perplexity 15.65   gap +0.0258
  arithmetic 2.619   chat 2.838   chess 2.013   code 2.961   reasoning 3.020   stories 2.943   wikipedia 2.861
repeats 93% of 8-grams, greedy with no guard
```

| Item | Meaning |
|------|---------|
| `step` | optimizer steps in the model's whole life |
| `X of Y characters` | progress through the first pass of the corpus |
| `reading … char/s` | characters trained per second |
| `writing … char/s` | characters generated per second (samples) |
| `still gaining` | does the far end of the window still help? (negative = no) |
| `grad norm … clipping` | gradient size before clipping; > 1 means clipped |
| `train loss` | recent average loss on training text |
| `lr` | the *expert* learning rate now (`3e-4 × rate`) |
| `evidence t` | the controller's verdict; above 2.2 → lr can rise |
| `rate x0.052` | the plasticity scale |
| `held-out loss ± se` | loss on never-trained text, with standard error |
| `gap` | held-out − train (large = memorising) |
| per-lane numbers | held-out loss for each subject |
| `repeats` | share of repeated 8-grams in the raw samples |

Console lines during the run (real examples from our log):

- `pool 65 experts (+1)  100.7M in VRAM  vram 2067MB` – a growth event (a prune would show a minus).
- `4.5 experts loaded onto the card per chunk, RAM hit rate 0.99` – paging health.
- `weights/ checkpointed` – a save finished.

For the author's snapshot the numbers are different: the README reports
`1,871.5M` characters read, best held-out `0.6188 nats / 0.8928 bits/char`, and
397 experts. Our `weights/` manifest shows `plasticity scale 0.0515`.

---

## 18. Config reference (`config.yaml`)

**model** *(shape settings are fixed when a model is created)*

| Key | Value | Meaning |
|-----|-------|---------|
| `d_model` | 512 | trunk width |
| `n_head` | 8 | attention heads |
| `d_ff` | 1408 | dense feed-forward width (prelude) |
| `n_prelude / n_recur / n_coda` | 2 / 1 / 0 | block counts |
| `max_steps` | 24 | most rows per character |
| `halt_prior` | 0.072 | mean ≈ 13.9 rows |
| `halt_thresh` | 0.9 | inference stops here |
| `halt_freeze` | true | halted characters stop computing |
| `ponder_beta` | 0.01 | KL weight |
| `bptt_window` | 24 | rows with gradient |
| `train_steps_mean` | 12.8 | Poisson mean for sampled depth |
| `context_start / end` | 2048 / 4096 | window grows between these |
| `context_grow_every_chars` | 100,000 | +1 character per this many read |
| `context_gain_min` | 0.015 | gain needed to keep growing |

**pool**

| Key | Value | Meaning |
|-----|-------|---------|
| `experts` | 64 | starting pool size |
| `width` / `depth` | 2048 / 1 | expert hidden size / stacked blocks |
| `top_k` | 8 | experts per character per row |
| `resident` | 32 | GPU slots |
| `ram_cache` | 96 | experts kept in RAM |
| `capacity_factor` | 0 | 0 = never drop assignments |
| `balance` | 3.5e-5 | balance-term weight |
| `select_temperature` | 0 | 0 = deterministic vote |

**training**: `precision` bf16 · `chunk` 2048 · `lr` 3e-4 · `trunk_lr_mult` 0.1 · `weight_decay` 0.1 · `clip` 1.0 · `save_every` 5 min.

**growth**: `every_chars` 2M · `k` 1 · `max_gap` 0.4 · `dying_frac_max` 0.35 · `max_in_flight` 50 · `keep_ratio_min` 0.35 · `max_disk_gb` 10 · `mem_frac` 0.95 · `birth_gate` 0.001.

**prune**: `survival_chars` 100M · `dying_at` 0.65.

**data**: `train` data/train · `held_out` data/val · `weights` weights · `passage` 32,768.

**decoding**: `adapt_strength` 2.5 · `adapt_decay` 0.88 · `rep_penalty` 1.0 (off).

Command-line flags override the file (`python3 train.py read --help` lists them all).
Do **not** put the old keys `lr_horizon`, `lr_warmup`, `lr_floor` in `training:` – the program
stops with an error.

---

## 19. Known quirks and honest limits

Things we **verified** in the code or in the run:

1. **`self-knowledge` lane is not created.** `corpora/expand.py` writes the generated chat corpus
   to `data/train/chat/`, not `data/train/self-knowledge/`. So `corpora all` reports
   "self-knowledge produced nothing", you train on **7** lanes, and `serve.py` has nothing to
   prime from (use `--prime-chars 0`).
2. **Tiny `--limit` builds are lopsided.** With `--limit 5000` the `stories` lane had 16 KB of
   training text but 4.7 MB of validation text. Use a full build for real training.
3. **The "repairs itself on divergence" claim** (`--revert-factor`) is implemented only in the old
   `stream` command, not in `read`.
4. **`minagi/__init__.py` lists modules** (`schedule`, `notify`) that do not exist.
5. **The code lane is built from Python on *your* machine**, so your corpus differs from the
   author's.
6. **`datasets` library version:** `datasets 2.14.4` breaks with `pyarrow 25`
   (`PyExtensionType` error). Upgrading `datasets` fixed it for us.
7. **CUDA is non-deterministic:** the README says two identical runs differ by ≈ 0.014 nats.
8. **The README credits "Claude Opus" 5 and 5.5 models** with most of the code, with the author
   giving the ideas and direction. The code has very long explanatory comments.
9. **Port 8080 was already used** on your machine by another program, so we served chat on 8765.

Limits of the model itself:

- It is toy-scale. Code, arithmetic and Wikipedia samples degenerate into repetition.
- The forgetting result covers one new domain at one checkpoint.
- Final weights are not published; the snapshot is "undertrained" by the author's own words.

---

## 20. Command cheat sheet

All commands run from `~/Downloads/mini-AGI`. Use `.venv/bin/python` (or `source .venv/bin/activate`).

```bash
# --- setup (once) ---
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python torch numpy pyyaml matplotlib flask scipy chess zstandard datasets huggingface_hub
.venv/bin/python tools/device_check.py

# --- stay up to date ---
git pull --ff-only

# --- data ---
.venv/bin/python -m corpora all --limit 5000        # small test slice
.venv/bin/python -m corpora all                     # sampled build (a few GB)
.venv/bin/python -m corpora all --only wikipedia    # one lane

# --- author's weights ---
.venv/bin/python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download('Volotat/mini-AGI-undertrained', local_dir='hf_snapshot', max_workers=8)
EOF
mv hf_snapshot/weights weights
.venv/bin/python -m minagi.store weights            # summary of any weights folder

# --- chat ---
.venv/bin/python serve.py --weights weights --no-learn --prime-chars 0 --port 8765

# --- train your own (never use --weights-dir weights unless you mean it) ---
.venv/bin/python train.py read data/train --save --weights-dir weights_mine \
    --held-out data/val --sample-every 10 --sample-log runs/my_samples.txt
# long run in the background
nohup .venv/bin/python train.py read data/train --save --weights-dir weights_mine \
    --held-out data/val --sample-every 10 --sample-log runs/my_samples.txt > train.log 2>&1 &
tail -f train.log          # watch
kill -INT <pid>            # stop cleanly (checkpoints first)

# --- your own files ---
.venv/bin/python train.py read ~/notes                              # dry run
.venv/bin/python train.py read ~/notes --passes 3 --save --weights-dir weights_mine --lr 5e-5
```

---

## 21. FAQ

**Is this "AGI"?** No. It is a research toy about continual learning. The name is a half-joke.

**Why bytes instead of words?** It can read any file with no tokenizer to fit, and no unknown
characters. The cost is long sequences.

**Why do experts live on disk?** So the model can have 1.25 B parameters while only ≈ 100 M must be
on the GPU. The cost is occasional loading; "the text chooses" makes loads rare.

**Why is the trunk learning rate 0.1×?** Measured: it cut forgetting from +1.27 to +0.13 nats on a
new domain, and the new domain was learned *faster*.

**Why can the model "think longer"?** The recurrent block can run up to 24 times, and the halting
head decides when each character stops. Easy characters use few rows.

**Why does the learning rate keep going down?** The controller's resting behaviour is a gentle decay
unless held-out loss is clearly improving, because overshooting costs far more than undershooting.

**Can I run it on 8 GB?** The README targets 8 GB cards: VRAM depends on the 32 slots, not on the
pool size. Our run showed `vram 2067 MB` to `3826 MB` for the pool part at 65–66 experts, plus
the activations.

**Is it safe to chat with `--no-learn` off?** It changes the weights on disk. Keep a backup.

**How long to train a useful model from scratch?** The author's run: 1.87 B characters in about
21 days on a laptop GPU, and it is still undertrained. A faster GPU helps, but expect days to
weeks before samples look coherent.

**Where do I start reading the code?** `minagi/recur.py` (`RecurCoder.forward`), then
`minagi/paged.py` (`admit`, `_place`), then `train.py` (`cmd_read`).
