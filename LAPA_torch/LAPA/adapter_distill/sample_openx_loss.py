import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adapter_distill.data import LapaCollator, LatentActionDataset, load_jsonl_records, load_lapa_tokenizer
from adapter_distill.models import load_adapter_checkpoint, load_frozen_lapa


def main() -> None:
    parser = argparse.ArgumentParser(description="Print per-sample OpenX delta loss for a trained adapter checkpoint.")
    # This is the trained PyTorch adapter checkpoint, not the original full
    # JAX LAPA checkpoint. Swap this path to probe another adapter run.
    parser.add_argument("--checkpoint", default="adapter_distill/checkpoints/pythia160m/best.pt")
    # Frozen tensors extracted from the official LAPA checkpoint:
    # vision/text/delta embeddings plus the delta prediction head.
    parser.add_argument("--frozen_lapa", default="adapter_distill/artifacts/lapa_frozen.npz")
    # LAPA/LWM SentencePiece tokenizer, used for the instruction text.
    parser.add_argument("--vocab_file", default="lapa_checkpoints/tokenizer.model")
    # OpenX pretraining JSONL. Each row already has VQGAN vision tokens and
    # four latent-action target tokens under the "delta" field.
    parser.add_argument("--data_path", default="data/latent_action_pretraining_openx.jsonl")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_text_len", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    # Read a small contiguous slice so line_index maps back to the JSONL file.
    records = load_jsonl_records(args.data_path, start=args.start, limit=args.count, skip_bad_records=True)
    if not records:
        raise ValueError(f"No records found in {args.data_path} from --start {args.start}")

    # The collator converts JSON rows into the model inputs:
    #   vision: [B, 256] VQGAN token ids
    #   text_ids/text_lengths: tokenized instructions
    #   delta: [B, 4] target latent-action ids
    tokenizer = load_lapa_tokenizer(args.vocab_file)
    collate = LapaCollator(tokenizer=tokenizer, max_text_len=args.max_text_len)
    loader = DataLoader(LatentActionDataset(records), batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    # Rebuild the adapter model with frozen LAPA embedding/head tensors, then
    # load the trained adapter/backbone weights from --checkpoint.
    frozen = load_frozen_lapa(args.frozen_lapa, device=device)
    model, metadata = load_adapter_checkpoint(args.checkpoint, frozen_lapa=frozen, map_location=device)
    model.to(device)
    model.eval()

    line_index = args.start
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            # model(**batch) returns:
            #   loss: batch-averaged CE from the model's forward pass
            #   logits: [B, 4, 8], one 8-way distribution for each delta token
            _, logits = model(**batch)
            # Recompute CE with reduction="none" so we keep one loss per
            # latent-action token instead of collapsing the whole batch.
            token_losses = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                batch["delta"].reshape(-1),
                reduction="none",
            ).reshape(batch["delta"].shape)
            # Average the four delta-token losses into one scalar per sample.
            per_sample_loss = token_losses.mean(dim=-1)
            pred = logits.argmax(dim=-1)
            # Token accuracy over the four latent-action codes.
            per_sample_acc = (pred == batch["delta"]).float().mean(dim=-1)

            for i in range(batch["delta"].shape[0]):
                record = records[line_index - args.start]
                result = {
                    "line_index": line_index,
                    "dataset_name": record.get("dataset_name"),
                    "delta": batch["delta"][i].detach().cpu().tolist(),
                    "pred": pred[i].detach().cpu().tolist(),
                    "delta_loss": float(per_sample_loss[i].detach().cpu()),
                    "delta_acc": float(per_sample_acc[i].detach().cpu()),
                    "checkpoint": args.checkpoint,
                    "model_name": metadata.get("model_name"),
                }
                print(json.dumps(result))
                line_index += 1


if __name__ == "__main__":
    main()
