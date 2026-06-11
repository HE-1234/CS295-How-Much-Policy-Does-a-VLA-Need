import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adapter_distill.data import LapaActionCollator, LapaCollator, load_lapa_tokenizer
from adapter_distill.models import load_adapter_checkpoint, load_frozen_lapa


def encode_image_with_vqgan(image_path: str, vqgan_checkpoint: str) -> torch.Tensor:
    # Import JAX/VQGAN lazily so JSONL-token inference can run in a pure
    # PyTorch environment.
    import albumentations
    import jax
    from PIL import Image
    from latent_pretraining.vqgan import VQGAN

    image = np.array(Image.open(image_path)).astype(np.uint8)
    preproc = albumentations.Compose([
        albumentations.LongestMaxSize(max_size=256),
        albumentations.Resize(256, 256),
    ])
    image = preproc(image=image)["image"]
    image = (image / 127.5 - 1.0).astype(np.float32)[None]
    vqgan = VQGAN(vqgan_checkpoint, replicate=False)
    tokens = jax.device_get(vqgan.encode(image))[1].astype(int).reshape(-1)
    return torch.tensor(tokens.tolist(), dtype=torch.long)


def sample_from_jsonl(path: str, index: int, mode: str) -> tuple[torch.Tensor, str, torch.Tensor | None]:
    # JSONL mode is the fastest way to test the adapter because vision tokens
    # are already precomputed by the original LAPA pipeline.
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if i == index:
                item = json.loads(line)
                vision = torch.tensor([int(v) for v in item["vision"]], dtype=torch.long)
                key = "action" if mode == "action" else "delta"
                target = torch.tensor([int(d) for d in item[key]], dtype=torch.long) if key in item else None
                return vision, item["instruction"], target
    raise IndexError(f"Index {index} not found in {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--frozen_lapa", default="adapter_distill/artifacts/lapa_frozen.npz")
    parser.add_argument("--vocab_file", default="lapa_checkpoints/tokenizer.model")
    parser.add_argument("--jsonl")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--mode", choices=["delta", "action"], default="delta")
    parser.add_argument("--image")
    parser.add_argument("--instruction")
    parser.add_argument("--vqgan_checkpoint", default="lapa_checkpoints/vqgan")
    args = parser.parse_args()

    if args.jsonl:
        # Use precomputed VQGAN tokens from LAPA's latent-action dataset.
        vision, instruction, target = sample_from_jsonl(args.jsonl, args.index, args.mode)
    else:
        if not args.image or not args.instruction:
            raise ValueError("Provide either --jsonl or both --image and --instruction")
        # Raw-image mode mirrors original LAPA inference by first tokenizing the
        # image through the original VQGAN.
        vision = encode_image_with_vqgan(args.image, args.vqgan_checkpoint)
        instruction, target = args.instruction, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frozen = load_frozen_lapa(args.frozen_lapa, device=device)
    model, metadata = load_adapter_checkpoint(args.checkpoint, frozen, map_location=device)
    model.to(device).eval()

    tokenizer = load_lapa_tokenizer(args.vocab_file)
    if args.mode == "action":
        collate = LapaActionCollator(tokenizer=tokenizer, max_text_len=int(metadata.get("max_text_len", 128)))
        batch = collate([{"vision": vision, "instruction": instruction, "action": torch.zeros(7, dtype=torch.long)}])
    else:
        collate = LapaCollator(tokenizer=tokenizer, max_text_len=int(metadata.get("max_text_len", 128)))
        batch = collate([{"vision": vision, "instruction": instruction, "delta": torch.zeros(4, dtype=torch.long)}])
    batch = {k: v.to(device) for k, v in batch.items()}
    if args.mode == "action":
        pred = model.generate_action(batch["vision"], batch["text_ids"], batch["text_lengths"], n_tokens=7)
    else:
        pred = model.generate_delta(batch["vision"], batch["text_ids"], batch["text_lengths"], n_tokens=4)

    print("instruction:", instruction)
    print(f"pred_{args.mode}:", pred[0].detach().cpu().tolist())
    if target is not None:
        print(f"target_{args.mode}:", target.tolist())


if __name__ == "__main__":
    main()
