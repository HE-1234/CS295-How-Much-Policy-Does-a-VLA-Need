import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


PARAM_MAP = {
    # These are the only original LAPA tensors needed by the adapter model.
    "vte": "params/transformer/vte/embedding",
    "wte": "params/transformer/wte/embedding",
    "dte": "params/transformer/dte/embedding",
    "delta_head": "params/delta_head/kernel",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lapa_checkpoint", default="params::lapa_checkpoints/params")
    parser.add_argument("--output", default="adapter_distill/artifacts/lapa_frozen.npz")
    parser.add_argument(
        "--jax_platform",
        default="cpu",
        help="JAX platform to use while reading the checkpoint. Use 'auto' to leave JAX unchanged.",
    )
    args = parser.parse_args()

    # Extraction only copies checkpoint tensors, so CPU is enough and avoids
    # fragile cluster CUDA/cuDNN library resolution during JAX import.
    if args.jax_platform != "auto":
        os.environ.setdefault("JAX_PLATFORMS", args.jax_platform)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    from flax.traverse_util import flatten_dict
    from tux import JaxDistributedConfig, StreamingCheckpointer, set_random_seed

    JaxDistributedConfig.initialize(JaxDistributedConfig.get_default_config())
    set_random_seed(1234)
    # Load only params from the released streaming checkpoint. This can still
    # require substantial RAM because the released model is about 7B params.
    _, params = StreamingCheckpointer.load_trainstate_checkpoint(
        args.lapa_checkpoint,
        disallow_trainstate=True,
        max_buffer_size=32 * 2**30,
    )
    flat = flatten_dict(params, sep="/")

    weights = {}
    for name, key in PARAM_MAP.items():
        if key not in flat:
            raise KeyError(f"Missing {key}; available sample: {list(flat)[:8]}")
        # Store float32 arrays for portability across JAX/PyTorch environments.
        weights[name] = np.asarray(flat[key]).astype("float32")
        print(f"{name:10s} {weights[name].shape} {weights[name].dtype}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **weights, source_checkpoint=np.array(args.lapa_checkpoint))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
