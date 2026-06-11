# LAPA Backbone Scaling Study — Progress Report

*Last updated: 2026-06-08*

This document records what we set out to do, why we made the choices we made, what we
actually built, and what the experiments have shown so far. It is written to be read
start-to-finish by someone who has not seen the code, and it reflects the **current** state
of the repository (which has moved well beyond the original plan in `CLAUDE.md`).

---

## 1. Motivation and Research Question

**LAPA** (Latent Action Pretraining from Videos) trains a robot policy to predict *latent
action tokens* — discrete codes produced by a VQ-VAE ("LAQ") — from a visual observation and
a language instruction. The original work uses a **JAX/Flax LLaMA-7B** backbone trained
*from scratch* on ~43 GB of Open-X Embodiment robot video.

Our research question is narrower and was never addressed by the original paper:

> **How does the size of the LLM backbone affect the quality of latent action learning in
> LAPA-style pretraining?**

To answer this cleanly we fix every variable except the backbone — same visual input (VQGAN
tokens), same dataset, same learning rate / batch size / epochs — and sweep only the model
size, using the **Pythia** series (160M → 6.9B). Pythia is ideal here because every size was
trained on the same data (The Pile) under identical conditions, so parameter count is the
*only* difference between checkpoints. The primary deliverable is a **scaling curve**:
validation token accuracy vs. backbone parameter count.

**Baselines.** For the latent-action (`delta`) task, the codebook makes random guessing
`1/8 = 12.5%` per token. Anything consistently above that means the model has learned
something.

---

## 2. Why We Did *Not* Use the Original JAX Codebase

The original LAPA code (`latent_pretraining/`) is JAX + Flax with a custom-from-scratch
LLaMA implementation that is **not** compatible with HuggingFace and **not** designed for
swapping backbones — changing the model means hand-editing `llama.py`. For a study whose
whole point is to vary the backbone, that is the wrong tool.

So the strategy was to keep the original objective and data but re-implement the policy
training in **PyTorch + HuggingFace**, where swapping a backbone is a one-line change
(`AutoModelForCausalLM.from_pretrained(name)`). The JSONL data files produced by LAPA's LAQ
pipeline are reused unchanged — we never re-run the VQ-VAE; the `delta` labels are already in
the files.

---

## 3. Two Approaches We Built

We ended up building **two** PyTorch policies. The first was the originally-planned clean
rewrite; the second (the one that produced our headline results) reuses LAPA's own frozen
weights to make the comparison faithful.

### 3.1 Approach A — Clean PyTorch rewrite (`train_policy.py`)

This is the simplest possible faithful re-implementation:

```
vision (256 VQGAN tokens) → nn.Embedding(8192 → hidden)
instruction text          → backbone tokenizer + embeddings
latent action (4 tokens)  → 8 new special tokens <ACT_0>..<ACT_7> in the LLM vocab
        ↓ concatenate [vision | text | action] as embeddings
   Pythia backbone (the only thing that changes)
        ↓
   standard lm_head over the extended vocab → predict the 4 action tokens
```

Key design decisions:

- **External, randomly-initialized vision embedding** (`nn.Embedding(8192, hidden)`) instead
  of CLIP or a projection layer. The VQGAN tokens feed straight into the LLM embedding space,
  keeping visual processing *identical* across all backbone sizes.
- **Action tokens added to the vocabulary** (`resize_token_embeddings`) so the *existing*
  `lm_head` already covers them — no new output head.
- **Loss only on the 4 action positions** (everything else is masked to `-100`). No text or
  vision reconstruction loss.

This is a clean isolation of the action-prediction signal, but it has one important
weakness for *our* question: because the vision embedding and action tokens are randomly
initialized and learned only on our data, the backbone's pretrained NLP prior does much of
the work, and the visual grounding is learned from scratch. It answers "do bigger pretrained
LMs fine-tune better here," not "does LAPA's latent-action representation scale."

### 3.2 Approach B — Frozen-LAPA Adapter (`adapter_distill/`) ← main experiment

To make the comparison faithful to LAPA itself, we built a second policy that **reuses the
original LAPA model's own embedding and head weights, frozen**, and only learns a small
backbone in the middle plus two projection layers:

```
frozen LAPA vte (vision) / wte (text) / dte (delta)   →  4096-d LAPA embedding space
        ↓  input_proj: Linear(4096 → small_hidden)
   small Pythia causal LM  (trainable — the variable under study)
        ↓  output_proj: Linear(small_hidden → 4096)
        ↓  frozen LAPA delta_head (4096 → 8)
   predict the 4 latent action tokens
```

What is **frozen** (copied out of the official `params` checkpoint): the vision-token
embedding `vte (8448×4096)`, text embedding `wte (32000×4096)`, latent-action embedding
`dte (8×4096)`, and the `delta_head (4096×8)`. What is **trainable**: the Pythia backbone and
the two 4096↔hidden projection bridges.

Why this is the better experiment:

- The visual, textual, and action representations are exactly LAPA's — so we are genuinely
  measuring how well a small backbone can do LAPA's job inside LAPA's own representation
  space, not learning a new one from scratch.
- The frozen tensors (≈660 MB, stored once in `artifacts/lapa_frozen.npz`) are deliberately
  kept *out* of the saved adapter checkpoints (`persistent=False` buffers), so each
  checkpoint is small.

The extraction step (`extract_lapa_weights.py`) runs on CPU JAX so it does not need a GPU and
avoids CUDA/cuDNN loader conflicts; it writes `.npz` so the PyTorch side never imports JAX.

The same module (`FrozenLAPAAdapterPolicy` in `adapter_distill/models.py`) also supports a
parallel **action** path (`ate` / `action_head`, 245-way) used later for SIMPLER downstream
fine-tuning (Section 7).

---

## 4. Environment Setup

A non-trivial part of the work was getting JAX (for the original LAPA model / VQGAN /
weight extraction) and PyTorch (for our training) to coexist on the lab's RTX A6000 GPUs.
The full reproducible procedure is in [`walk through.md`](walk%20through.md). The crucial
lessons:

- Use a dedicated conda env (`lapa_gpu`, Python 3.10), `pip` for everything.
- Pin **JAX 0.4.23 with cuDNN 8.9** (`nvidia-cudnn-cu12>=8.9,<9`). A later cuDNN 9.x silently
  breaks JAX GPU init (`CUDA backend failed to initialize`).
- **Never install `tensorflow[and-cuda]`** in this env — it drags in a conflicting CUDA/cuDNN
  stack. Plain `tensorflow` is fine (the JAX training code imports it).
- Always `unset LD_LIBRARY_PATH` before running JAX.
- Install PyTorch from the **cu118** wheels so it doesn't overwrite the cuDNN package JAX
  depends on.

---

## 5. Data

- **Latent-action pretraining:** `data/latent_action_pretraining_openx.jsonl` — the official
  LAPA Open-X pretraining file, **277,758 records**, already labeled with LAQ. Each line has
  256 VQGAN vision token ids, a formatted instruction, and a 4-token `delta` latent action in
  `[0,7]`. We split 97% train / 3% val → **269,425 train / 8,333 val**.
- **SIMPLER downstream:** `data/simpler.jsonl` — **2,482 records** with a 7-DoF `action`
  field plus an `action_scale` CSV (`data/simpler.csv`) used to map discrete action tokens
  back to continuous robot commands.

No data was re-generated; the `delta` labels come straight from LAPA's LAQ pipeline.

---

## 6. Results — Latent Action Scaling (the main result)

All runs: `lr=1e-4`, AdamW, 269K train / 8.3K val, frozen-LAPA adapter (Approach B). Batch
size was reduced for larger models to fit 48 GB. Best epoch = highest val token accuracy.

| Backbone | Params | Batch | Epochs run | **Best val acc** | (Best epoch) | Notes |
|---|---|---|---|---|---|---|
| Pythia-160m | 160M | 64 | 15 | **0.471** | 0 | overfits hard after ~ep2 |
| Pythia-410m | 410M | 32 | 7 (of 15) | **0.478** | 3 | best result so far |
| Pythia-1b   | 1.0B | 16 | 3 (of 15) | **0.469** | 2 | run truncated early |
| Pythia-6.9b | 6.9B | 3  | 0 complete | — | — | started via 3-GPU torchrun/FSDP, no epoch logged yet |

**Random baseline = 12.5%.** Every trained backbone reaches **~47%**, almost 4× random —
the model is clearly learning the latent-action structure.

Key observations:

- **The scaling effect is small in this regime.** Going 160M → 410M → 1B moves best val
  accuracy only within ~0.469–0.478. With the larger models trained for far fewer epochs, the
  ordering is not yet conclusive — 410m currently leads, but 1B and 6.9b runs are incomplete.
- **Strong overfitting is the dominant dynamic.** For 160m, val accuracy *peaks at epoch 0*
  (0.471) and then declines to 0.41 by epoch 14 while train loss keeps dropping (1.55 → 0.52)
  and val loss climbs (1.47 → 2.42). The frozen LAPA embeddings/head plus a capable backbone
  fit the train set quickly; the useful signal is in the **early** epochs.
- **Takeaway for the report:** in the fine-tuning-with-frozen-LAPA-representation regime,
  backbone size from 160M–1B is *not* the bottleneck — data/regularization and early stopping
  matter more. The clean scaling story the project originally hypothesized is muted here.

> To complete the curve we still need: finish 1B (15 epochs), get at least a few epochs of
> 6.9B, and ideally add early-stopping-based "best" numbers with multiple seeds.

Also built but secondary: `latent_pretraining/sample_openx_loss.py` evaluates the **official
7B LAPA model's** loss on the same records, giving a reference point against which to compare
our small-backbone adapters.

---

## 7. Downstream Evaluation on SIMPLER

The latent-action accuracy is an intrinsic metric; the extrinsic one is **task success** in a
simulator. We wired the adapter into **SimplerEnv** (Bridge / WidowX tasks).

**Pipeline built:**

1. **Action fine-tuning** (`adapter_distill/train_simpler_adapter.py`): starting from a
   pretrained latent-action adapter (`pythia160m/best.pt`), fine-tune the parallel 245-way
   action head on `data/simpler.jsonl` to predict the 7-DoF discretized robot action.
2. **Policy wrapper** (`SimplerEnv/.../policies/lapa_adapter/lapa_adapter_model.py`):
   `LAPAAdapterInference` encodes the camera image with the **original LAPA VQGAN** → 256
   tokens, formats the instruction, runs the adapter's `generate_action` (7 tokens
   autoregressively), and maps tokens → continuous action via the `action_scale` CSV, with
   the usual SIMPLER gripper/sticky-action logic.
3. **Eval driver** (`SimplerEnv/scripts/lapa_adapter_bridge.sh`): `smoke` runs one task,
   `full` runs the four Bridge tasks (stack cube, carrot-on-plate, spoon-on-cloth,
   eggplant-in-basket). Hooked into `main_inference_lapa.py` via a new `lapa-adapter` policy
   option in `argparse.py`.

**SIMPLER action-prediction results (160m, 2,234 train / 248 val):** training overfits
almost immediately — train loss → ~0.01 while val loss climbs from ~4.8 to ~15. Best val
**action-token** accuracy was only **0.126** (epoch 33), essentially flat across 200 epochs.

**Interpretation:** with only ~2.2K SIMPLER samples the action head memorizes the training set
and does not generalize, so we should not expect meaningful simulated success rates from this
checkpoint yet. The infrastructure (fine-tune → wrap → evaluate end-to-end in SimplerEnv) is
complete and runs; the bottleneck is downstream **data quantity** and **regularization**, not
plumbing.

---

## 8. What Is Faithfully Preserved vs. Changed

**Preserved from original LAPA:** the objective (predict discrete latent action tokens from
vision+language via cross-entropy), the data and `delta` labels, the 256-token VQGAN visual
representation, the task structure, and — in Approach B — LAPA's actual frozen
vision/text/delta embeddings and delta head.

**Deliberately changed:** JAX→PyTorch; from-scratch LLaMA-7B → fine-tuned pretrained Pythia;
multi-host pjit/FSDP → single-node (with optional `torchrun`/FSDP for 6.9B); custom streaming
checkpointer → `torch.save`; mixed `0.99·delta + 0.01·text` loss → delta-only loss; full 43 GB
training → the pre-split OpenX JSONL.

**Implication for interpreting results:** because we *fine-tune* pretrained models inside
LAPA's frozen representation rather than pretraining from scratch, our numbers are **not**
directly comparable to LAPA's reported figures. High accuracy may partly reflect the strength
of LAPA's frozen embeddings and the LMs' priors rather than new visual-action grounding. This
is a limitation to state plainly in the final write-up.

---

## 9. Status Summary

| Component | Status |
|---|---|
| Environment (JAX + PyTorch coexistence on A6000) | ✅ done, documented |
| Frozen LAPA weight extraction → `lapa_frozen.npz` | ✅ done |
| Approach A clean rewrite (`train_policy.py`) | ✅ runs (MVP/sanity path) |
| Approach B frozen-LAPA adapter | ✅ built, trained |
| Scaling sweep 160m / 410m | ✅ complete enough to compare (~47%) |
| Scaling sweep 1B | ◐ 3/15 epochs |
| Scaling sweep 6.9B | ◐ launched (FSDP), no epoch logged |
| SIMPLER action fine-tune + eval pipeline | ✅ built end-to-end; overfits on 2.2K samples |

**Headline finding so far:** small Pythia backbones (160M–1B), wrapped around LAPA's frozen
representation, all learn latent actions to ~47% token accuracy (vs. 12.5% random), but
**backbone size is not the dominant factor in this fine-tuning regime** — overfitting and
data scale are. Downstream SIMPLER success is blocked by the small (~2.2K) action dataset
rather than by the policy architecture.

---

## 10. Next Steps

1. Finish the 1B run (15 epochs) and get ≥3 epochs of 6.9B so the scaling curve is complete.
2. Add **early stopping / weight decay / dropout** — val accuracy peaks at epoch 0–3, so the
   reported number should be best-epoch, and regularization may lift the larger models.
3. Run each size with **multiple seeds** to put error bars on the ~47% plateau.
4. Expand SIMPLER fine-tuning data (or augment) so the downstream success rate becomes
   meaningful, then run `lapa_adapter_bridge.sh full` for each checkpoint.
5. Compare against the **official 7B LAPA loss** (`sample_openx_loss.py`) as an upper-reference.
6. Plot the final scaling curve (val acc vs. params) for the report.
</content>
</invoke>
