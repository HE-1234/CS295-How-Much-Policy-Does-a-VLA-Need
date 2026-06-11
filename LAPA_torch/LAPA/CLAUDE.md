# LAPA Backbone Scaling Study — Project Guide

## Context

The original LAPA repo (`latent_pretraining/`) uses a **JAX/Flax LLaMA-7B** backbone. It is not designed for backbone swapping. This scaling study replaces that with a clean **PyTorch + HuggingFace** rewrite (`train_policy.py`) that uses the same pretraining objective but supports easy backbone swaps via `AutoModelForCausalLM`. The JSONL data files are already preprocessed by the original pipeline and require no changes.

---

## Research Question

How does LLM backbone size affect latent action learning quality in LAPA-style pretraining?

## Experiment Design

Fix all variables except the LLM backbone. Swap only the backbone (Pythia series) and measure latent action token accuracy and downstream task success rate.

**Controlled variables:** visual input (VQGAN tokens), dataset, learning rate, batch size, epochs  
**Independent variable:** LLM backbone size  
**Dependent variables:** val token accuracy, downstream success rate

---

## Architecture

```
vision (256 VQGAN tokens, vocab=8192)
        ↓
nn.Embedding(8192 → llm_hidden_size)
        ↓
[text tokens from tokenizer]
        ↓
Pythia backbone (only this changes across experiments)
        ↓
predict delta: 4 latent action tokens (vocab=8, i.e. 8^4 codebook)
```

No CLIP. No projection layer. Visual tokens feed directly into the LLM embedding space. This keeps the visual processing **identical across all backbone sizes**, making comparisons clean.

---

## Experiment Matrix

| Model | Params | HuggingFace ID |
|-------|--------|----------------|
| Pythia-160m | 160M | `EleutherAI/pythia-160m` |
| Pythia-410m | 410M | `EleutherAI/pythia-410m` |
| Pythia-1b | 1B | `EleutherAI/pythia-1b` |
| Pythia-1.4b | 1.4B | `EleutherAI/pythia-1.4b` |

GPT-2 variants (`gpt2`, `gpt2-medium`, `gpt2-large`) are available as fallback, but Pythia is preferred because all sizes were trained under identical conditions — cleaner for a scaling study.

---

## Data

**Source:** LAPA official pretraining data (Open-X Embodiment, pre-labeled with LAQ)

```bash
# Already downloaded and split:
train_small.jsonl   # 20,000 samples
val_small.jsonl     #  1,000 samples
```

**Data format** (one line = one timestep):
```json
{
  "dataset_name": "fractal20220817_data",
  "instruction": "<s> You are a helpful assistant. USER: pick rxbar chocolate... ASSISTANT:",
  "vision": ["6780", "2783", ...],   // 256 VQGAN token ids (strings)
  "delta": ["1", "1", "6", "0"],     // latent action: 4 ints in range [0, 7]
  "raw_action": [0.04, -0.03, ...],  // ground-truth robot action (not used in pretraining)
  "is_last": false,
  "fields": "[instruction],[vision],delta"
}
```

**Key insight:** `delta` is the latent action label produced by LAPA's LAQ (VQ-VAE). We do not need to re-run LAQ — the labels are already in the file.

Random baseline accuracy: 1/8 = **12.5%** per token. Target after pretraining: **>25%**.

---

## File Structure

```
LAPA/
├── train_small.jsonl                        # training data (20K samples, ready)
├── val_small.jsonl                          # validation data (1K samples, ready)
├── latent_action_pretraining_openx.jsonl    # full Open-X dataset (use for longer runs)
├── train_policy.py                          # ← CREATE THIS (code below)
├── latent_pretraining/                      # original JAX codebase (do not modify)
├── laq/                                     # original LAQ module (do not modify)
└── ckpt_*/                                  # saved checkpoints per backbone
```

---

## Running the Experiments

### Step 1 — Install dependencies
```bash
pip install transformers torch accelerate wandb tqdm
```

### Step 2 — Sanity check (CPU, ~2 min, no GPU needed)
```bash
python3 -c "
import json, torch
from train_policy import LAPAPolicy, LatentActionDataset
from torch.utils.data import DataLoader

ds = LatentActionDataset('train_small.jsonl')
dl = DataLoader(ds, batch_size=4)
vision, instructions, delta = next(iter(dl))

model = LAPAPolicy('EleutherAI/pythia-160m')
loss = model(vision, instructions, delta)
print('Loss:', loss.item())
print('Sanity check PASSED')
"
```

Expected output: a finite loss value and `Sanity check PASSED`.

### Step 3 — MVP training (pythia-160m only)
```bash
python train_policy.py
```

Watch for:
- Loss decreasing across steps
- Val accuracy rising above 12.5% (random baseline)

### Step 4 — Full scaling study
Change `llm_name` in `__main__` and rerun for each backbone:
```
EleutherAI/pythia-160m   →  160M
EleutherAI/pythia-410m   →  410M
EleutherAI/pythia-1b     →  1B
EleutherAI/pythia-1.4b   →  1.4B
```

---

## Evaluation Metrics

| Metric | When | How |
|--------|------|-----|
| Latent action token accuracy | During training (val set) | `compute_accuracy()` in training loop |
| Loss curve | During training | Logged to wandb |
| Downstream success rate | After training | Finetune on SIMPLER, run SIMPLER evaluator |

The primary result is a **scaling curve**: val accuracy (y-axis) vs. backbone parameter count (x-axis).

Random baseline = **12.5%** (1/8 per token).  
A model that has learned something useful should exceed this consistently.

---

## Project Timeline

| Week | Goal |
|------|------|
| Week 1 | Sanity check passes; MVP (pythia-160m) trains and loss goes down |
| Week 2 | All four Pythia sizes trained; scaling curve plotted |
| Week 3 | SIMPLER finetuning and evaluation for each checkpoint |
| Week 4 | Write report; frame as first systematic backbone scaling study for latent action pretraining |

---

## Key Design Decisions and Rationale

**Why Pythia over GPT-2?**  
All Pythia checkpoints were trained on the same data (The Pile) with the same hyperparameters. This makes size the only difference — exactly what a scaling study needs. GPT-2 sizes were trained under inconsistent conditions.

**Why VQGAN tokens instead of CLIP?**  
The LAPA data file stores vision as 256 pre-computed VQGAN tokens, not raw images. Using them directly avoids re-downloading 43GB of image data, removes the need for a projection layer, and keeps visual processing identical across all backbone sizes.

**Why not use JAX / the original LAPA codebase?**  
The original code is JAX-based, uses a custom LLaMA implementation incompatible with HuggingFace, and is not designed for swapping backbones. Rewriting the policy training in PyTorch costs less time than learning JAX from scratch and enables direct use of HuggingFace model hub.

**What are we not replicating from LAPA?**  
We are not replicating LAPA's absolute performance numbers. We are replicating its *pretraining objective* (predict latent action tokens from visual + language context) with smaller backbones, to answer a question the original paper did not address.

---

## Troubleshooting

**CUDA out of memory**  
Reduce `batch_size` (try 8 or 4). For pythia-1.4b, you may need `batch_size=4` and gradient checkpointing:
```python
model.llm.gradient_checkpointing_enable()
```

**Loss not decreasing after 500 steps**  
Check that `act_token_start` is correctly set and that labels are not all `-100`. Add a debug print:
```python
print("act_token_start:", model.act_token_start)
print("sample act_ids:", (delta + model.act_token_start)[0])
```

**Val accuracy stuck at ~12.5% (random)**  
This is expected for the first few hundred steps. If it does not improve after 1 full epoch, check that `compute_accuracy` is indexing the correct logit positions.

---

## How This Differs from Original LAPA

This section documents precisely what was kept, changed, and dropped compared to the original LAPA paper and codebase (`latent_pretraining/`). The goal is to make the relationship between the two clear for anyone reading the code or the final report.

### 1. Framework and Backbone

| | Original LAPA | This Study |
|---|---|---|
| **Framework** | JAX + Flax | PyTorch + HuggingFace |
| **Backbone** | Custom LLaMA from scratch | Pretrained Pythia (HF Hub) |
| **Default size** | LLaMA-7B | Pythia-160m → 1.4b |
| **Backbone init** | Random (trained from scratch on robot data) | Pretrained on The Pile (NLP corpus) |
| **Backbone swapping** | Manual config edit in `llama.py` | One string change (`llm_name`) |

The original LAPA trains the LLaMA backbone **from scratch** on 43GB of Open-X robot data — it is a pretraining run, not fine-tuning. This study **fine-tunes pretrained** Pythia models on 20K samples. This is a deliberate change: we are testing whether pretrained language priors help or scale better, not replicating LAPA's exact training regime.

### 2. Model Architecture

**Original LAPA** (`latent_pretraining/delta_llama.py`):
- Three separate output heads baked into the LLM module: `lm_head` (text), `vision_head` (visual reconstruction), `delta_head` (action prediction)
- `delta_head` is a dedicated linear layer projecting hidden states → `delta_vocab_size=32`
- Vision embedding (`vte`) is an embedding table **inside** the LLM module, part of the model graph
- Delta embedding (`dte`) is similarly internal
- All three modalities share a flat token sequence, distinguished by boolean masks (`vision_masks`, `delta_masks`)

**This study** (`train_policy.py`):
- Single output head: the pretrained `lm_head` from the backbone (no new linear layers)
- Action tokens are appended to the LLM's vocabulary as 8 special tokens (`<ACT_0>`–`<ACT_7>`), so the existing `lm_head` already covers them after `resize_token_embeddings`
- Vision embedding (`vision_embed`) is an **external** `nn.Embedding` concatenated into `inputs_embeds` before the LLM — not part of the LLM internals
- Sequence layout: `[vis_emb (256) | text_emb (T) | act_emb (4)]` concatenated as embeddings, not token ids

The external vision embedding is the most important architectural difference. In the original, VQGAN tokens are routed through a learned `vte` table that is co-trained with the full LLM. Here, `vision_embed` is randomly initialized and learned only on the 20K samples, while the LLM body retains its pretrained weights.

### 3. Training Objective and Loss

**Original LAPA** (`latent_pretraining/train.py`, `vision,text,delta` modality):
```python
loss = 0.99 * delta_loss + 0.01 * text_loss
```
The model is jointly supervised to predict latent actions (99%) and reconstruct text tokens (1%). Vision tokens are reconstructed through `vision_head` but do not contribute to the scalar loss. This mixed objective helps the model not forget how to do language.

**This study**:
```python
# Only action positions have non -100 labels
labels[:, vis_len + text_len:] = act_ids   # the rest is -100
```
Only the 4 action token positions are supervised. There is no text reconstruction loss and no vision reconstruction loss. This is simpler and isolates the action prediction signal, but removes the regularizing effect of the text loss.

### 4. Action Token Vocabulary

| | Original LAPA | This Study |
|---|---|---|
| **Codebook size** | `delta_vocab_size=32` | 8 (matching `delta` field range [0,7]) |
| **Representation** | Separate linear head output | Special tokens in LLM vocab |
| **Prediction** | `delta_head` linear projection | Standard `lm_head` over extended vocab |

The original uses a 32-token codebook for delta actions. The JSONL data has `delta` values in [0, 7], so this study uses 8 tokens. Both represent the same VQ-VAE codebook; the difference is only in how the head is structured.

### 5. Infrastructure and Scale

| | Original LAPA | This Study |
|---|---|---|
| **Training data** | ~43 GB Open-X (full dataset) | 20K samples (pre-split subset) |
| **Distributed training** | pjit, FSDP, tensor parallelism, ring attention | Single GPU (DataLoader + AdamW) |
| **Checkpointing** | Streaming msgpack (custom `StreamingCheckpointer`) | `torch.save()` state dict |
| **Optimizer** | Custom `OptimizerFactory` (likely Adam + LR schedule) | Plain AdamW, constant LR |
| **Evaluation** | Seen + unseen eval sets, delta/vision/text accuracy | Val set only, action accuracy |

### 6. What Is Faithfully Preserved

Despite the differences above, the core research artifact is preserved:

1. **Same pretraining objective**: predict discrete latent action tokens from (vision, language) context using cross-entropy loss
2. **Same data**: identical JSONL files produced by the original LAPA LAQ pipeline — the `delta` labels are unchanged
3. **Same visual representation**: 256 VQGAN token IDs per frame, vocab=8192, processed identically across all backbone sizes
4. **Same task structure**: the model sees a robot instruction and a visual observation, and must predict what latent action happened next
5. **Same evaluation metric**: per-token accuracy against the ground-truth `delta` labels, with 12.5% random baseline

### 7. Implications for Interpreting Results

Because this study **fine-tunes** pretrained models rather than training from scratch, results are not directly comparable to LAPA's reported numbers. Concretely:

- Higher accuracy here may reflect language model priors rather than true visual-action grounding
- The scaling curve (accuracy vs. param count) may look different from what a from-scratch study would produce
- The 20K sample size is tiny relative to the original 43GB dataset; results may be dominated by overfitting or data scale rather than backbone capacity

These are limitations to address in the report. For the purpose of this study — systematically varying backbone size under controlled conditions — the design is sound.
