import argparse
import functools
import json
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch.amp import GradScaler, autocast
from torch.distributed.fsdp import FullStateDictConfig, FullyShardedDataParallel as FSDP, MixedPrecision, StateDictType
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXLayer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adapter_distill.data import LapaCollator, LatentActionDataset, load_jsonl_records, load_lapa_tokenizer
from adapter_distill.models import FrozenLAPAAdapterPolicy, load_frozen_lapa, save_adapter_checkpoint


def distributed_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def is_rank0(rank: int) -> bool:
    return rank == 0


def optional_positive_int(value: str) -> int | None:
    if value.lower() in {"none", "all"}:
        return None
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("sample counts must be non-negative, 'none', or 'all'")
    return parsed


def append_metric(path: Path, record: dict, rank: int = 0) -> None:
    if not is_rank0(rank):
        return
    record = {"time": time.time(), **record}
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def setup_distributed(strategy: str) -> dict:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if strategy not in {"auto", "fsdp", "none"}:
        raise ValueError(f"Unknown distributed strategy: {strategy}")
    if strategy == "fsdp" and world_size == 1:
        raise ValueError("--distributed_strategy fsdp requires torchrun with WORLD_SIZE > 1")
    if strategy == "none" and world_size > 1:
        raise ValueError("--distributed_strategy none cannot be used with WORLD_SIZE > 1")

    use_fsdp = world_size > 1 and strategy in {"auto", "fsdp"}
    if use_fsdp:
        if not torch.cuda.is_available():
            raise RuntimeError("FSDP training requires CUDA")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device("cuda", local_rank)
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return {
        "use_fsdp": use_fsdp,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "device": device,
        "strategy": "fsdp" if use_fsdp else "none",
    }


def cleanup_distributed() -> None:
    if distributed_is_initialized():
        dist.destroy_process_group()


def resolve_precision(name: str, device: torch.device) -> tuple[str, torch.dtype]:
    if name == "auto":
        if device.type != "cuda":
            name = "fp32"
        elif torch.cuda.is_bf16_supported():
            name = "bf16"
        else:
            name = "fp16"

    mapping = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }
    if name not in mapping:
        raise ValueError(f"Unknown precision: {name}")
    if device.type != "cuda" and name != "fp32":
        raise ValueError(f"--precision {name} requires CUDA")
    return name, mapping[name]


def make_fsdp_mixed_precision(dtype: torch.dtype) -> MixedPrecision | None:
    if dtype == torch.float32:
        return None
    return MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)


def make_grad_scaler(dtype: torch.dtype, use_fsdp: bool, device: torch.device):
    if device.type != "cuda" or dtype != torch.float16:
        return None
    if use_fsdp:
        return ShardedGradScaler()
    return GradScaler("cuda")


def reduce_sums(values: list[float], device: torch.device) -> list[float]:
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    if distributed_is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.cpu().tolist()


def set_seed(seed: int, rank: int) -> None:
    seed = seed + rank
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_records(records: list[dict], train_samples: int | None, val_samples: int | None, val_fraction: float):
    n_records = len(records)
    if n_records == 0:
        raise ValueError("Training data is empty")

    if train_samples is None and val_samples is None:
        # Default path: consume the full file with a deterministic train/val split.
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

    train_records = records[:train_count]
    val_records = records[train_count : train_count + val_count]
    return train_records, val_records


def evaluate(model, loader, device) -> tuple[float, float]:
    model.eval()
    loss_sum, correct, n_tokens, n_samples = 0.0, 0.0, 0.0, 0.0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, logits = model(**batch)
            # Token-level accuracy over the four latent action codes.
            pred = logits.argmax(dim=-1)
            batch_size = float(batch["delta"].shape[0])
            loss_sum += loss.item() * batch_size
            correct += float((pred == batch["delta"]).sum().item())
            n_tokens += float(batch["delta"].numel())
            n_samples += batch_size
    loss_sum, correct, n_tokens, n_samples = reduce_sums([loss_sum, correct, n_tokens, n_samples], device)
    return loss_sum / max(n_samples, 1.0), correct / max(n_tokens, 1.0)


def maybe_enable_gradient_checkpointing(model: FrozenLAPAAdapterPolicy, enabled: bool) -> None:
    if not enabled:
        return
    if hasattr(model.backbone, "gradient_checkpointing_enable"):
        model.backbone.gradient_checkpointing_enable()
    if hasattr(model.backbone.config, "use_cache"):
        model.backbone.config.use_cache = False


def wrap_fsdp(model: FrozenLAPAAdapterPolicy, device: torch.device, dtype: torch.dtype) -> FSDP:
    has_gpt_neox_layer = any(isinstance(module, GPTNeoXLayer) for module in model.modules())
    if not has_gpt_neox_layer:
        raise ValueError("FSDP auto-wrap currently expects a GPT-NeoX/Pythia backbone")

    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={GPTNeoXLayer},
    )
    return FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=make_fsdp_mixed_precision(dtype),
        device_id=device,
        use_orig_params=True,
        limit_all_gathers=True,
    )


def save_checkpoint(path: Path, model, rank: int, use_fsdp: bool, **metadata) -> None:
    if use_fsdp:
        config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, config):
            state_dict = model.state_dict()
        if is_rank0(rank):
            torch.save({"model_state": state_dict, "metadata": metadata}, path)
        if distributed_is_initialized():
            dist.barrier()
        return

    if is_rank0(rank):
        save_adapter_checkpoint(str(path), model, **metadata)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_lapa", default="adapter_distill/artifacts/lapa_frozen.npz")
    parser.add_argument("--vocab_file", default="lapa_checkpoints/tokenizer.model")
    parser.add_argument("--data_path", default="data/latent_action_pretraining_openx.jsonl")
    parser.add_argument("--model_name", default="EleutherAI/pythia-6.9b")
    parser.add_argument("--output_dir", default="adapter_distill/checkpoints/pythia6p9b")
    parser.add_argument("--train_samples", type=optional_positive_int, default=None)
    parser.add_argument("--val_samples", type=optional_positive_int, default=None)
    parser.add_argument("--val_fraction", type=float, default=0.03)
    parser.add_argument("--max_text_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--metrics_path", default=None, help="Defaults to <output_dir>/metrics.jsonl.")
    parser.add_argument("--log_every", type=int, default=50, help="Write train-step loss every N steps. Use 0 to disable.")
    parser.add_argument("--strict_jsonl", action="store_true", help="Fail instead of skipping malformed JSONL records.")
    parser.add_argument("--distributed_strategy", choices=("auto", "fsdp", "none"), default="auto")
    parser.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument("--no_gradient_checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    dist_info = setup_distributed(args.distributed_strategy)
    device = dist_info["device"]
    rank = dist_info["rank"]
    world_size = dist_info["world_size"]
    use_fsdp = dist_info["use_fsdp"]
    precision_name, train_dtype = resolve_precision(args.precision, device)
    set_seed(args.seed, rank)
    tokenizer = load_lapa_tokenizer(args.vocab_file)
    collate = LapaCollator(tokenizer=tokenizer, max_text_len=args.max_text_len)

    # Deterministic contiguous split keeps runs reproducible without writing
    # split files. With the default caps of None, every JSONL record is used.
    all_records = load_jsonl_records(args.data_path, skip_bad_records=not args.strict_jsonl)
    train_records, val_records = split_records(all_records, args.train_samples, args.val_samples, args.val_fraction)
    train_dataset = LatentActionDataset(train_records)
    val_dataset = LatentActionDataset(val_records)
    train_sampler = (
        DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed)
        if use_fsdp
        else None
    )
    val_sampler = (
        DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
        if use_fsdp
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        collate_fn=collate,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=collate,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    # Frozen tensors are buffers inside the model; optimizer only sees Pythia
    # and the trainable adapter/action projection layers.
    frozen = load_frozen_lapa(args.frozen_lapa, device=device, dtype=train_dtype)
    model = FrozenLAPAAdapterPolicy(
        frozen,
        model_name=args.model_name,
        freeze_backbone=args.freeze_backbone,
        backbone_dtype=train_dtype,
    )
    if train_dtype != torch.float32:
        model.to(dtype=train_dtype)
    maybe_enable_gradient_checkpointing(model, enabled=use_fsdp and not args.no_gradient_checkpointing)
    if use_fsdp:
        model = wrap_fsdp(model, device=device, dtype=train_dtype)
    else:
        model.to(device)
    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    scaler = make_grad_scaler(train_dtype, use_fsdp=use_fsdp, device=device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(args.metrics_path) if args.metrics_path else output_dir / "metrics.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    append_metric(
        metrics_path,
        {
            "event": "run_start",
            "args": vars(args),
            "device": str(device),
            "rank": rank,
            "world_size": world_size,
            "precision": precision_name,
            "distributed_strategy": dist_info["strategy"],
            "global_batch_size": args.batch_size * world_size,
            "total_records": len(all_records),
            "train_records": len(train_records),
            "val_records": len(val_records),
        },
        rank=rank,
    )

    best_acc = -1.0
    global_step = 0
    for epoch in range(args.num_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        train_loss_sum, train_samples_seen = 0.0, 0.0
        progress = tqdm(train_loader, desc=f"epoch {epoch}", disable=not is_rank0(rank))
        for step, batch in enumerate(progress):
            batch = {k: v.to(device) for k, v in batch.items()}
            amp_context = (
                autocast(device_type="cuda", dtype=train_dtype)
                if device.type == "cuda" and train_dtype != torch.float32
                else nullcontext()
            )
            with amp_context:
                loss, _ = model(**batch)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step_loss = loss.item()
            batch_size = float(batch["delta"].shape[0])
            train_loss_sum += step_loss * batch_size
            train_samples_seen += batch_size
            global_step += 1
            if is_rank0(rank):
                progress.set_postfix(loss=f"{step_loss:.4f}")
            if is_rank0(rank) and args.log_every > 0 and global_step % args.log_every == 0:
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
                    rank=rank,
                )

        val_loss, val_acc = evaluate(model, val_loader, device)
        train_loss_sum, train_samples_seen = reduce_sums([train_loss_sum, train_samples_seen], device)
        train_loss = train_loss_sum / max(train_samples_seen, 1.0)
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
            rank=rank,
        )
        if is_rank0(rank):
            print(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        save_checkpoint(
            output_dir / "last.pt",
            model,
            rank,
            use_fsdp,
            model_name=args.model_name,
            max_text_len=args.max_text_len,
            freeze_backbone=args.freeze_backbone,
            distributed_strategy=dist_info["strategy"],
            precision=precision_name,
        )
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(
                output_dir / "best.pt",
                model,
                rank,
                use_fsdp,
                model_name=args.model_name,
                max_text_len=args.max_text_len,
                freeze_backbone=args.freeze_backbone,
                distributed_strategy=dist_info["strategy"],
                precision=precision_name,
            )

    cleanup_distributed()


if __name__ == "__main__":
    main()
