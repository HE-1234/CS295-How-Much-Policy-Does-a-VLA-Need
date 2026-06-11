import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec as PS
from tux import (
    JaxDistributedConfig,
    StreamingCheckpointer,
    get_float_dtype_by_name,
    make_shard_and_gather_fns,
    match_partition_rules,
    set_random_seed,
    tree_apply,
    with_sharding_constraint,
)

from latent_pretraining.delta_llama import FlaxDeltaLaMAForCausalLMModule, VideoLLaMAConfig


DEFAULT_OFFICIAL_LAPA_CHECKPOINT = "params::lapa_checkpoints/params"
DEFAULT_OPENX_PRETRAIN_DATA = "data/latent_action_pretraining_openx.jsonl"
DEFAULT_LAPA_TOKENIZER = "lapa_checkpoints/tokenizer.model"


def read_records(path: str, start: int, count: int) -> list[tuple[int, dict]]:
    records = []
    with open(path, "r") as f:
        for idx, line in enumerate(f):
            if idx < start:
                continue
            if len(records) >= count:
                break
            if line.strip():
                records.append((idx, json.loads(line)))
    return records


def encode_record(record, tokenizer, tokens_per_delta: int):
    token_buffer = []
    loss_mask_buffer = []
    vision_mask = []
    delta_mask = []

    def extend(tokens, loss_mask: float, is_vision: bool = False, is_delta: bool = False):
        token_buffer.extend(tokens)
        loss_mask_buffer.extend([loss_mask] * len(tokens))
        vision_mask.extend([is_vision] * len(tokens))
        delta_mask.extend([is_delta] * len(tokens))

    token_buffer.append(tokenizer.bos_token_id)
    loss_mask_buffer.append(0.0)
    vision_mask.append(False)
    delta_mask.append(False)

    fields = record["fields"].split(",")
    vision_start = tokenizer.encode("<vision>")
    vision_end = tokenizer.encode("</vision>")
    delta_start = tokenizer.encode("<delta>")
    delta_end = tokenizer.encode("</delta>")

    for field in fields:
        masked = field.startswith("[") and field.endswith("]")
        field = field[1:-1] if masked else field
        loss_mask = 0.0 if masked else 1.0

        if field == "instruction":
            extend(tokenizer.encode(record["instruction"]), loss_mask)
        elif field == "vision":
            vision_tokens = [int(x) for x in record["vision"][:256]]
            extend(vision_start, loss_mask)
            extend(vision_tokens, loss_mask, is_vision=True)
            extend([8193], loss_mask, is_vision=True)
            extend(vision_end, loss_mask)
        elif field == "delta":
            delta_tokens = [int(x) for x in record["delta"][:tokens_per_delta]]
            if len(delta_tokens) != tokens_per_delta:
                raise ValueError(f"Expected {tokens_per_delta} delta tokens, got {len(delta_tokens)}")
            extend(delta_start, loss_mask)
            extend(delta_tokens, loss_mask, is_delta=True)
            extend(delta_end, loss_mask)
        else:
            extend(tokenizer.encode(record[field]), loss_mask)

    token_buffer.append(tokenizer.eos_token_id)
    loss_mask_buffer.append(1.0)
    vision_mask.append(False)
    delta_mask.append(False)

    return token_buffer, loss_mask_buffer, vision_mask, delta_mask


def encode_records(records, tokenizer, tokens_per_delta: int):
    encoded = []
    for line_index, record in records:
        tokens, loss_masks, vision_masks, delta_masks = encode_record(record, tokenizer, tokens_per_delta)
        encoded.append((line_index, record, tokens, loss_masks, vision_masks, delta_masks))
    return encoded


def round_up_to_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def iter_chunks(items, chunk_size: int):
    for start in range(0, len(items), chunk_size):
        yield start, items[start:start + chunk_size]


def build_batch(encoded_records, tokenizer, seq_length: int) -> dict[str, np.ndarray]:
    batch_size = len(encoded_records)
    batch = {
        "input_tokens": np.full((batch_size, seq_length), tokenizer.bos_token_id, dtype=np.int32),
        "target_tokens": np.full((batch_size, seq_length), tokenizer.bos_token_id, dtype=np.int32),
        "loss_masks": np.zeros((batch_size, seq_length), dtype=np.float32),
        "input_vision_masks": np.zeros((batch_size, seq_length), dtype=bool),
        "input_delta_masks": np.zeros((batch_size, seq_length), dtype=bool),
        "target_delta_masks": np.zeros((batch_size, seq_length), dtype=bool),
    }

    for row, (line_index, record, tokens, loss_masks, vision_masks, delta_masks) in enumerate(encoded_records):
        if len(tokens) > seq_length + 1:
            raise ValueError(
                f"JSONL line {line_index} produced {len(tokens)} tokens; "
                f"increase --seq_length beyond {seq_length}"
            )

        tokens = np.asarray(tokens, dtype=np.int32)
        loss_masks = np.asarray(loss_masks, dtype=np.float32)
        vision_masks = np.asarray(vision_masks, dtype=bool)
        delta_masks = np.asarray(delta_masks, dtype=bool)

        input_tokens = tokens[:-1]
        target_tokens = tokens[1:]
        input_vision_masks = vision_masks[:-1]
        input_delta_masks = delta_masks[:-1]
        target_delta_masks = delta_masks[1:]
        shifted_loss_masks = loss_masks[1:]

        n = len(input_tokens)
        batch["input_tokens"][row, :n] = input_tokens
        batch["target_tokens"][row, :n] = target_tokens
        batch["loss_masks"][row, :n] = shifted_loss_masks
        batch["input_vision_masks"][row, :n] = input_vision_masks
        batch["input_delta_masks"][row, :n] = input_delta_masks
        batch["target_delta_masks"][row, :n] = target_delta_masks

    return batch


def token_nll(logits, targets):
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.take_along_axis(log_probs, targets[..., None], axis=-1)[..., 0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Print per-sample latent-action loss for OpenX JSONL records using "
            "the official pretrained JAX/Flax LAPA checkpoint."
        )
    )
    parser.add_argument("--data_path", default=DEFAULT_OPENX_PRETRAIN_DATA)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=100,
        help="Number of sampled examples to score per GPU forward pass.",
    )
    parser.add_argument("--seq_length", type=int, default=384)
    parser.add_argument("--tokens_per_delta", type=int, default=4)
    parser.add_argument("--vocab_file", default=DEFAULT_LAPA_TOKENIZER)
    parser.add_argument(
        "--load_checkpoint",
        default=DEFAULT_OFFICIAL_LAPA_CHECKPOINT,
        help="Official LAPA checkpoint spec. Defaults to params::lapa_checkpoints/params.",
    )
    parser.add_argument("--mesh_dim", default="1,-1,1,1")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--jax_distributed", type=dict, default=JaxDistributedConfig.get_default_config())
    parser.add_argument("--load_llama_config", default="7b")
    parser.add_argument(
        "--update_llama_config",
        default=(
            "dict(delta_vocab_size=8,theta=50000000,max_sequence_length=2048,"
            "use_flash_attention=True,scan_attention=False,scan_mlp=False,scan_layers=True)"
        ),
    )
    parser.add_argument("--json_output", default="", help="Optional path to write JSONL results.")
    args = parser.parse_args()

    if args.count <= 0:
        raise ValueError("--count must be positive")
    if args.eval_batch_size <= 0:
        raise ValueError("--eval_batch_size must be positive")
    if "adapter_distill" in args.load_checkpoint:
        raise ValueError(
            "This script evaluates the original JAX LAPA checkpoint. "
            "Do not pass adapter_distill checkpoints here."
        )

    JaxDistributedConfig.initialize(args.jax_distributed)
    set_random_seed(args.seed)

    tokenizer_config = VideoLLaMAConfig.get_tokenizer_config()
    tokenizer_config.vocab_file = args.vocab_file
    tokenizer = VideoLLaMAConfig.get_tokenizer(tokenizer_config)

    records = read_records(args.data_path, args.start, args.count)
    if not records:
        raise ValueError(f"No records found in {args.data_path} from --start {args.start}")
    encoded_records = encode_records(records, tokenizer, args.tokens_per_delta)

    mesh = VideoLLaMAConfig.get_jax_mesh(args.mesh_dim)
    if args.load_llama_config:
        llama_config = VideoLLaMAConfig.load_config(args.load_llama_config)
        # Match latent_pretraining.train/inference: load the named 7B config,
        # then apply runtime-only scan/remat defaults from a config instance.
        updates = VideoLLaMAConfig()
        llama_config.update(
            dict(
                remat_block=updates.remat_block,
                remat_attention=updates.remat_attention,
                remat_mlp=updates.remat_mlp,
                scan_attention=updates.scan_attention,
                scan_mlp=updates.scan_mlp,
                scan_query_chunk_size=updates.scan_query_chunk_size,
                scan_key_chunk_size=updates.scan_key_chunk_size,
                scan_mlp_chunk_size=updates.scan_mlp_chunk_size,
                scan_layers=updates.scan_layers,
                param_scan_axis=updates.param_scan_axis,
            )
        )
    else:
        llama_config = VideoLLaMAConfig.get_default_config()
    if args.update_llama_config:
        llama_config.update(dict(eval(args.update_llama_config)))
    llama_config.update(
        dict(
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            mesh_dim=args.mesh_dim,
            sample_mode="all",
        )
    )
    required_seq_length = max(len(tokens) - 1 for _, _, tokens, *_ in encoded_records)
    seq_length = args.seq_length
    if required_seq_length > seq_length:
        seq_length = round_up_to_multiple(required_seq_length, 128)
        if seq_length > llama_config.max_sequence_length:
            raise ValueError(
                f"Sampled records require seq_length={required_seq_length}, "
                f"but model max_sequence_length is {llama_config.max_sequence_length}."
            )

    model = FlaxDeltaLaMAForCausalLMModule(
        llama_config,
        dtype=get_float_dtype_by_name(args.dtype),
    )

    with jax.default_device(jax.devices("cpu")[0]):
        _, params = StreamingCheckpointer.load_trainstate_checkpoint(
            args.load_checkpoint,
            disallow_trainstate=True,
            max_buffer_size=32 * 2**30,
        )

    model_ps = match_partition_rules(
        VideoLLaMAConfig.get_partition_rules(llama_config.scan_layers, llama_config.param_scan_axis),
        params,
    )
    shard_fns, _ = make_shard_and_gather_fns(model_ps, get_float_dtype_by_name(args.dtype))
    with mesh:
        params = tree_apply(shard_fns, params)

    def loss_fn(params, batch):
        batch = with_sharding_constraint(batch, PS(("dp", "fsdp"), "sp"))
        _, _, delta_logits = model.apply(
            params,
            batch["input_tokens"],
            batch["input_vision_masks"],
            batch["input_delta_masks"],
            deterministic=True,
        ).logits
        mask = batch["loss_masks"] * batch["target_delta_masks"]
        nll = jnp.where(mask > 0, token_nll(delta_logits, batch["target_tokens"]), 0.0)
        per_sample_tokens = jnp.maximum(mask.sum(axis=-1), 1.0)
        per_sample_loss = nll.sum(axis=-1) / per_sample_tokens
        pred = jnp.argmax(delta_logits, axis=-1)
        per_sample_acc = ((pred == batch["target_tokens"]) * mask).sum(axis=-1) / per_sample_tokens
        return per_sample_loss, per_sample_acc

    sharded_loss_fn = pjit(
        loss_fn,
        in_shardings=(model_ps, PS()),
        out_shardings=(PS(), PS()),
    )

    output_file = Path(args.json_output) if args.json_output else None
    output_handle = output_file.open("w") if output_file else None
    try:
        sample_results = []
        for batch_start, chunk in iter_chunks(encoded_records, args.eval_batch_size):
            batch = build_batch(chunk, tokenizer, seq_length)
            with mesh:
                per_sample_loss, per_sample_acc = sharded_loss_fn(params, batch)
            per_sample_loss = np.asarray(jax.device_get(per_sample_loss)).tolist()
            per_sample_acc = np.asarray(jax.device_get(per_sample_acc)).tolist()

            for (line_index, record, *_), loss, acc in zip(chunk, per_sample_loss, per_sample_acc):
                result = {
                    "model_source": "official_lapa_jax",
                    "checkpoint": args.load_checkpoint,
                    "line_index": line_index,
                    "dataset_name": record.get("dataset_name"),
                    "delta": [int(x) for x in record["delta"]],
                    "delta_loss": float(loss),
                    "delta_acc": float(acc),
                    "eval_batch_start": batch_start,
                }
                sample_results.append(result)
                print(json.dumps(result))
                if output_handle:
                    output_handle.write(json.dumps(result) + "\n")

        summary = {
            "event": "summary",
            "model_source": "official_lapa_jax",
            "checkpoint": args.load_checkpoint,
            "start": args.start,
            "count": len(sample_results),
            "eval_batch_size": args.eval_batch_size,
            "seq_length": seq_length,
            "required_seq_length": int(required_seq_length),
            "avg_delta_loss": float(np.mean([item["delta_loss"] for item in sample_results])),
            "avg_delta_acc": float(np.mean([item["delta_acc"] for item in sample_results])),
        }
        print(json.dumps(summary))
        if output_handle:
            output_handle.write(json.dumps(summary) + "\n")
    finally:
        if output_handle:
            output_handle.close()


if __name__ == "__main__":
    main()
