import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adapter_distill.data import (
    LapaActionCollator,
    SimplerActionDataset,
    load_jsonl_records,
    load_lapa_tokenizer,
)
from adapter_distill.models import load_adapter_checkpoint, load_frozen_lapa, save_adapter_checkpoint


def optional_positive_int(value: str) -> int | None:
    if value.lower() in {"none", "all"}:
        return None
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("sample counts must be non-negative, 'none', or 'all'")
    return parsed


def append_metric(path: Path, record: dict) -> None:
    record = {"time": time.time(), **record}
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def split_records(records: list[dict], train_samples: int | None, val_samples: int | None, val_fraction: float):
    n_records = len(records)
    if n_records == 0:
        raise ValueError("Training data is empty")

    if train_samples is None and val_samples is None:
        val_count = int(round(n_records * val_fraction))
        val_count = min(max(val_count, 1), max(n_records - 1, 0))
        train_count = n_records - val_count
    elif train_samples is None:
        val_count = min(val_samples or 0, n_records)
        train_count = n_records - val_count
    elif val_samples is None:
        train_count = min(train_samples, n_records)
        val_count = n_records - train_count
    else:
        train_count = min(train_samples, n_records)
        val_count = min(val_samples, max(n_records - train_count, 0))

    if train_count <= 0:
        raise ValueError("No training records selected; reduce validation size or fraction")

    return records[:train_count], records[train_count : train_count + val_count]


def evaluate(model, loader, device) -> tuple[float, float]:
    model.eval()
    losses, accs = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, logits = model.forward_action(**batch)
            pred = logits.argmax(dim=-1)
            losses.append(loss.item())
            accs.append((pred == batch["action"]).float().mean().item())
    return sum(losses) / max(len(losses), 1), sum(accs) / max(len(accs), 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_checkpoint", default="adapter_distill/checkpoints/pythia160m/best.pt")
    parser.add_argument("--frozen_lapa", default="adapter_distill/artifacts/lapa_frozen.npz")
    parser.add_argument("--vocab_file", default="lapa_checkpoints/tokenizer.model")
    parser.add_argument("--data_path", default="data/simpler.jsonl")
    parser.add_argument("--output_dir", default="adapter_distill/checkpoints/pythia160m_simpler")
    parser.add_argument("--train_samples", type=optional_positive_int, default=None)
    parser.add_argument("--val_samples", type=optional_positive_int, default=None)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--max_text_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--metrics_path", default=None, help="Defaults to <output_dir>/metrics.jsonl.")
    parser.add_argument("--log_every", type=int, default=20, help="Write train-step loss every N steps. Use 0 to disable.")
    parser.add_argument("--strict_jsonl", action="store_true", help="Fail instead of skipping malformed JSONL records.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_lapa_tokenizer(args.vocab_file)
    collate = LapaActionCollator(tokenizer=tokenizer, max_text_len=args.max_text_len)

    all_records = load_jsonl_records(args.data_path, skip_bad_records=not args.strict_jsonl)
    train_records, val_records = split_records(all_records, args.train_samples, args.val_samples, args.val_fraction)
    train_loader = DataLoader(
        SimplerActionDataset(train_records),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        SimplerActionDataset(val_records),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
    )

    frozen = load_frozen_lapa(args.frozen_lapa, device=device)
    model, metadata = load_adapter_checkpoint(args.base_checkpoint, frozen, map_location=device, strict=False)
    if args.freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad_(False)
    model.to(device)
    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(args.metrics_path) if args.metrics_path else output_dir / "metrics.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    append_metric(
        metrics_path,
        {
            "event": "run_start",
            "args": vars(args),
            "base_metadata": metadata,
            "device": str(device),
            "total_records": len(all_records),
            "train_records": len(train_records),
            "val_records": len(val_records),
        },
    )

    best_acc = -1.0
    global_step = 0
    for epoch in range(args.num_epochs):
        model.train()
        total = 0.0
        progress = tqdm(train_loader, desc=f"epoch {epoch}", file=sys.stdout, dynamic_ncols=True)
        for step, batch in enumerate(progress):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, _ = model.forward_action(**batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step_loss = loss.item()
            total += step_loss
            global_step += 1
            progress.set_postfix(loss=f"{step_loss:.4f}")
            if args.log_every > 0 and global_step % args.log_every == 0:
                print(
                    f"train_step epoch={epoch} step={step + 1}/{len(train_loader)} "
                    f"global_step={global_step} loss={step_loss:.4f} lr={optimizer.param_groups[0]['lr']:.2e}",
                    flush=True,
                )
                append_metric(
                    metrics_path,
                    {
                        "event": "train_step",
                        "epoch": epoch,
                        "step": step,
                        "global_step": global_step,
                        "loss": step_loss,
                        "lr": optimizer.param_groups[0]["lr"],
                    },
                )

        val_loss, val_acc = evaluate(model, val_loader, device)
        train_loss = total / max(len(train_loader), 1)
        append_metric(
            metrics_path,
            {
                "event": "epoch",
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
            },
        )
        print(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        save_adapter_checkpoint(
            str(output_dir / "last.pt"),
            model,
            model_name=model.model_name,
            max_text_len=args.max_text_len,
            freeze_backbone=args.freeze_backbone,
            task="simpler_action",
        )
        if val_acc > best_acc:
            best_acc = val_acc
            save_adapter_checkpoint(
                str(output_dir / "best.pt"),
                model,
                model_name=model.model_name,
                max_text_len=args.max_text_len,
                freeze_backbone=args.freeze_backbone,
                task="simpler_action",
            )


if __name__ == "__main__":
    main()
