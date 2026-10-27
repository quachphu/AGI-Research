# mini-AGI – end-to-end diagrams (Mermaid)

Other diagram files in this folder:

| File | What it is |
|------|-----------|
| `architecture-interactive.html` | **Interactive page** – open it in any browser (no internet needed) |
| `architecture.excalidraw` | One big editable drawing – open at <https://excalidraw.com> (File → Open) |
| `architecture-preview.png` | A picture of the Excalidraw drawing |
| `interactive-overview-preview.png` | A picture of the interactive page's first tab |
| `png/` | The 10 Mermaid diagrams below, already rendered as PNG images |
| `diagrams.md` | This file: 9 focused diagrams (10 Mermaid blocks), one for each stage |

---

## 1. The whole system on one page

```mermaid
flowchart LR
    subgraph BUILD["1 · Build data"]
        direction TB
        SRC["Hugging Face datasets<br/>Lichess · local .py files<br/>generators"] --> CB["python -m corpora all"]
        CB --> LANES["data/train/&lt;lane&gt;/*.txt<br/>data/val/&lt;lane&gt;/*.txt<br/>8 lanes"]
    end

    subgraph TRAIN["2 · Train (train.py read)"]
        direction TB
        FR["FileReader<br/>random file + offset<br/>window ≤ 4096"] --> MODEL
        MODEL["RecurCoder forward<br/>(see diagram 4)"] --> LOSS["loss = Σ p·CE + KL<br/>+ router terms"]
        LOSS --> BWD["backward · clip 1.0<br/>AdamW (trunk 0.1×)"]
        BWD --> PLAS["held-out check →<br/>plasticity controller"]
    end

    subgraph POOL["3 · Paged expert pool"]
        direction TB
        DISK[("Disk<br/>397 × 25 MB")] <--> RAM[("RAM cache<br/>96")]
        RAM <--> VRAM[("GPU slots<br/>32")]
    end

    subgraph STORE["4 · weights/"]
        W["manifest.json · core.npz<br/>routers.npz · optim.npz<br/>experts/eNNNNN.npz"]
    end

    subgraph SERVE["5 · serve.py"]
        direction TB
        UI["browser chat"] --> API["/api/chat (SSE)"]
        API --> GEN["prefill 512 · 1 forward / char<br/>greedy + adaptation trace"]
    end

    LANES --> FR
    MODEL <--> VRAM
    BWD --> W
    PLAS --> W
    DISK <--> W
    W --> SERVE
    GEN <--> VRAM
```

---

## 2. Building the data (`python -m corpora all`)

```mermaid
flowchart TD
    A["python -m corpora all"] --> B{"lane already<br/>has files?"}
    B -- yes --> SKIP["skip (use --force)"]
    B -- no --> SUB["run lane in its own subprocess"]

    SUB --> D1["fetch.py<br/>wikipedia · stories · chat (Hermes)"]
    SUB --> D2["reasoning.py<br/>OpenThoughts"]
    SUB --> G1["arithmetic.py"]
    SUB --> G2["code.py<br/>local Python files"]
    SUB --> G3["chess_games.py<br/>Lichess stream"]
    SUB --> G4["chat.py<br/>generated chat + self-knowledge"]

    D1 --> TXT
    D2 --> TXT
    G1 --> BIN["data_*_char/*.bin<br/>uint16 token ids"]
    G2 --> BIN
    G3 --> BIN
    G4 --> BIN
    BIN --> EXP["expand.py<br/>decode via tokenizer<br/>~200,000 chars per file"]
    EXP --> TXT["data/train/&lt;lane&gt;/ and data/val/&lt;lane&gt;/"]
    TXT --> SUM["print per-lane sizes<br/>+ 'lanes that produced nothing'"]
```

---

## 3. One training step (and the loop around it)

```mermaid
sequenceDiagram
    autonumber
    participant L as Lane scheduler
    participant R as FileReader
    participant M as Model + PagedPool
    participant O as AdamW
    participant P as Plasticity
    participant S as weights/ (disk)

    loop for each lane, round-robin
        L->>R: open random file at random offset (32,768 chars per visit)
        loop 16 chunks of 2,048 chars
            R->>M: window = last ≤4,096 chars (x) and shifted-by-1 (y)
            M->>M: admit 32 most-voted experts, load missing
            M->>M: forward (≈13.8 rows sampled) → loss
            M->>M: backward (gradient reaches the whole window)
            M->>O: clip grad norm to 1.0
            O->>M: pre-step hook: put each expert's own Adam moments in its slot
            O->>M: update trunk (lr × 0.1) and pool (lr)
        end
    end
    Note over M,P: every --sample-every minutes
    M->>P: held-out loss per lane
    P->>O: new rate scale (slow decay unless clearly improving)
    M->>S: checkpoint (atomic .tmp then rename)
    Note over M,S: every 2M chars: maybe grow 1 expert, prune unused ones
```

---

## 4. The model: one character, many rows

```mermaid
flowchart TD
    IN["character id (0-264)"] --> EMB["Embedding 265 → 512<br/>(tied with output head)"]
    EMB --> P1["Prelude block 1<br/>attention + dense SwiGLU 1408"]
    P1 --> P2["Prelude block 2"]
    P2 --> X(["x : what the text says"])

    H0(["h = 0"]) --> ADA
    X --> ADA["Adapter Linear [h ‖ x] 1024 → 512<br/>init [I | I] : re-injects x every row"]

    ADA --> ATT["Recurrent block · attention<br/>RoPE · KV cache of earlier chars"]
    ATT --> MOE["Mixture of experts<br/>router scores → top-8 of 32 admitted<br/>(see diagram 5)"]
    MOE --> HNEW(["new h"])

    HNEW --> NORM["RMSNorm"]
    NORM --> LOGIT["Output head → 265 logits"]
    NORM --> HALT["Halting head → λ (stop now?)"]

    HALT --> DEC{"cumulative stop mass<br/>≥ 0.9 (inference)<br/>or row = 24?"}
    DEC -- "no : next row" --> ADA
    DEC -- yes --> OUT["output = halting-weighted logits"]
    LOGIT --> OUT
    OUT --> NEXT["next-character probabilities"]

    DEC -. "stopped character is FROZEN:<br/>no more experts, later chars read its final K/V" .-> FRZ["halt_freeze"]
```

**Training loss:** `Σ_rows p_row · CE_row` + `0.01 · KL(p ‖ geometric prior 0.072)` + router auxiliary + balance term.
Training depth is sampled `Poisson(12.8) + 1` (clipped to 1…24).

---

## 5. Choosing and paging experts ("the text chooses")

```mermaid
flowchart TD
    T["characters of the text so far"] --> SC["every character scores ALL experts<br/>(router has 1 row per expert, even those on disk)"]
    SC --> RQ["each character requests its top-8<br/>weighted by router probability"]
    RQ --> VOTE["vote = sum of all requests<br/>since position 0"]
    VOTE --> ADM["admit the 32 most-voted experts"]
    ADM --> Q{"already in a slot?"}
    Q -- yes --> KEEP["stays in its slot"]
    Q -- no --> WHERE{"where is it?"}
    WHERE -- "RAM cache" --> LOADR["copy RAM → free slot"]
    WHERE -- "disk only" --> LOADD["read eNNNNN.npz → RAM → slot"]
    LOADR --> SLOT
    LOADD --> SLOT["slot choice:<br/>empty slot first, then longest-ago admitted"]
    KEEP --> ROUTE
    SLOT --> ROUTE["each character routes among the 32 admitted<br/>(if its request was not admitted: best admitted)"]
    ROUTE --> RUN["run each expert on only its own characters<br/>y = w2( silu(w1 x) ⊙ (w3 x) )"]
    RUN --> MIX["weighted sum · × gate · add back"]

    EVICT["evicted expert"] --> DIRTY{"did an optimizer<br/>step change it?"}
    DIRTY -- yes --> WB["write back (weights + its Adam moments)"]
    DIRTY -- no --> DROP["drop, nothing written"]
```

```mermaid
flowchart LR
    subgraph Tiers
        direction LR
        D[("DISK<br/>all experts<br/>~25 MB each<br/>slowest")] -- "LRU miss" --> R[("RAM cache<br/>96 experts<br/>medium")]
        R -- "admit" --> V[("GPU slots<br/>32 experts<br/>fastest")]
        V -- "evict (if changed)" --> R
        R -- "evict" --> D
    end
```

---

## 6. Growing and pruning the pool

```mermaid
flowchart TD
    EVERY["every 2,000,000 characters read<br/>(only with --save)"] --> B1{"ROOM<br/>disk ≤ 10 GB, VRAM ok?"}
    B1 -- no --> NO["do not grow"]
    B1 -- yes --> B2{"USED<br/>dying experts &lt; 35 %?"}
    B2 -- no --> NO
    B2 -- yes --> B3{"EARNING<br/>≥ 35 % of recent births still asked for?"}
    B3 -- no --> NO
    B3 -- yes --> B4{"FITS<br/>≤ 50 experts in trial?"}
    B4 -- no --> NO
    B4 -- yes --> B5{"HONEST<br/>held-out − train &lt; 0.4?"}
    B5 -- no --> NO
    B5 -- yes --> NEW["add 1 expert<br/>units drawn from 16 parents<br/>router row = unit-weighted mean<br/>gate 0.001"]

    EVERY --> PR{"expert not admitted<br/>for 100M characters?"}
    PR -- yes --> DEL["delete eNNNNN.npz<br/>(a newborn is protected for the same window)"]
    PR -- no --> LIVE["keep"]
```

---

## 7. The learning-rate controller

```mermaid
flowchart TD
    CH["held-out check (after ≥ 32 optimizer steps)"] --> OUT{"reading &gt; 5σ from the fit?"}
    OUT -- yes --> HOLD["hold it back until the next check confirms"]
    OUT -- no --> FITS["update 2 exponentially-weighted line fits<br/>slow (λ 0.97, ~66 checks) · fast (λ 0.85, ~12 checks)"]
    FITS --> EVID["evidence t = min( t_slow, 45·e_slow, 45·e_fast )"]
    EVID --> JUMP{"jump up and it stays<br/>(≥ 0.25 nats)?"}
    JUMP -- yes --> RES["rate × 2 (max 1.0) · reset fits<br/>'new material arrived'"]
    JUMP -- no --> N["nudge(t)"]
    N --> A["no evidence → slow decay (≈ × 0.9986)"]
    N --> B["clearly improving → hold / up to × 1.0008"]
    N --> C["getting worse → faster decay (≈ × 0.996)"]
    A --> SC["scale ∈ [0.01, 1.0]"]
    B --> SC
    C --> SC
    SC --> LRN["lr = 3e-4 × scale (pool) · × 0.1 (trunk)"]
```

---

## 8. Serving a chat reply

```mermaid
sequenceDiagram
    autonumber
    actor U as You (browser)
    participant F as Flask serve.py
    participant B as build_prompt
    participant S as stream() generator
    participant M as Model + PagedPool
    participant LL as LiveLearner

    U->>F: POST /api/chat {messages, max_new=400}
    F->>F: take global lock (one chat at a time)
    F->>B: wrap turns as <user>…</user><bot>
    B-->>F: prompt (kept to 90 % of context)
    F->>S: start generation
    S->>M: prefill prompt in 512-char chunks (fills KV cache, votes for experts)
    loop each new character
        S->>M: 1 forward (positions continue in the cache)
        M-->>S: logits (halting-weighted)
        S->>S: pick_next = argmax − adaptation trace (2.5, decay 0.88)
        S-->>U: SSE {"t": "char"} (+ "swap" events if the card changed)
    end
    S-->>F: stop at </bot> or max_new
    opt server started WITHOUT --no-learn
        F->>LL: add prompt + reply to the stream
        LL->>M: 1 learning step per chunk · save after 8 steps
    end
```

---

## 9. The `weights/` folder

```mermaid
flowchart LR
    W["weights/"] --> MJ["manifest.json<br/>cfg · step · read_chars · plasticity · telemetry"]
    W --> CORE["core.npz<br/>embedding · attention · norms<br/>adapter · halting head"]
    W --> RT["routers.npz<br/>router rows · depth embedding · gates"]
    W --> OP["optim.npz<br/>Adam moments of trunk + routers"]
    W --> EX["experts/"]
    EX --> E0["e00000.npz"]
    EX --> E1["e00001.npz"]
    EX --> EN["… e00396.npz"]
    E0 -.-> CONT["w1 · w3 · w2 (fp32)<br/>+ Adam m,v (bf16 bits)<br/>≈ 25 MB"]
```
