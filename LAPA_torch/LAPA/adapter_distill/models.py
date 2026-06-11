from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, GPTNeoXConfig, GPTNeoXForCausalLM


FROZEN_KEYS = ("vte", "wte", "dte", "delta_head")
OPTIONAL_FROZEN_KEYS = ("ate", "action_head")


def load_frozen_lapa(path: str, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32) -> dict:
    # Extraction writes .npz so the JAX/LAPA env does not need PyTorch.
    # Legacy .pt loading is kept for convenience during local experiments.
    if path.endswith(".npz"):
        archive = np.load(path)
        weights = {key: torch.from_numpy(archive[key]) for key in FROZEN_KEYS}
        for key in OPTIONAL_FROZEN_KEYS:
            if key in archive:
                weights[key] = torch.from_numpy(archive[key])
    else:
        frozen = torch.load(path, map_location=device)
        weights = frozen.get("weights", frozen)
    return {key: weights[key].to(device=device, dtype=dtype) for key in weights}


def hidden_size_from_config(config) -> int:
    for name in ("hidden_size", "n_embd", "word_embed_proj_dim"):
        value = getattr(config, name, None)
        if value is not None:
            return int(value)
    raise ValueError(f"Cannot infer hidden size from {type(config).__name__}")


def load_backbone(model_name: str, torch_dtype: torch.dtype | None = None):
    kwargs = {}
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype
    try:
        return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    except Exception:
        if model_name != "EleutherAI/pythia-160m":
            raise
        config = GPTNeoXConfig(
            vocab_size=50304,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            max_position_embeddings=2048,
            rotary_pct=0.25,
            rotary_emb_base=10000,
            use_parallel_residual=True,
            layer_norm_eps=1e-5,
        )
        model = GPTNeoXForCausalLM(config)
        if torch_dtype is not None:
            model.to(dtype=torch_dtype)
        return model


class FrozenLAPAAdapterPolicy(nn.Module):
    def __init__(
        self,
        frozen_lapa: dict[str, torch.Tensor],
        model_name: str = "EleutherAI/pythia-160m",
        freeze_backbone: bool = False,
        backbone_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.backbone = load_backbone(model_name, torch_dtype=backbone_dtype)
        self.model_name = model_name
        if hasattr(self.backbone.config, "use_cache"):
            self.backbone.config.use_cache = False
        small_hidden = hidden_size_from_config(self.backbone.config)
        lapa_hidden = frozen_lapa["wte"].shape[1]

        # These are the only required shape bridges between LAPA's 4096-d space
        # and the smaller LM hidden size.
        self.input_proj = nn.Linear(lapa_hidden, small_hidden)
        self.output_proj = nn.Linear(small_hidden, lapa_hidden)
        action_vocab_size = int(frozen_lapa.get("ate", torch.empty(245, lapa_hidden)).shape[0])
        self.ate = nn.Embedding(action_vocab_size, lapa_hidden)
        self.action_head = nn.Linear(lapa_hidden, action_vocab_size, bias=False)

        # Persistent=False keeps large frozen LAPA tensors out of adapter checkpoints.
        for key in FROZEN_KEYS:
            self.register_buffer(key, frozen_lapa[key], persistent=False)
        if "ate" in frozen_lapa:
            with torch.no_grad():
                self.ate.weight.copy_(frozen_lapa["ate"])
        if "action_head" in frozen_lapa:
            with torch.no_grad():
                # JAX Dense kernels are stored as [in, out], while PyTorch
                # Linear weights are [out, in].
                self.action_head.weight.copy_(frozen_lapa["action_head"].T)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad_(False)

    def _embed_and_pad(
        self,
        vision: torch.Tensor,
        text_ids: torch.Tensor,
        text_lengths: torch.Tensor,
        token_prefix: torch.Tensor | None,
        prefix_kind: str = "delta",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Reuse the original LAPA modality embeddings exactly: vision ids use
        # vte, text ids use wte, and previous/generated target ids use their
        # modality-specific table.
        vision_emb = F.embedding(vision, self.vte)
        text_emb = F.embedding(text_ids.clamp_max(self.wte.shape[0] - 1), self.wte)
        if token_prefix is None:
            target_emb = None
        elif prefix_kind == "delta":
            target_emb = F.embedding(token_prefix, self.dte)
        elif prefix_kind == "action":
            target_emb = self.ate(token_prefix)
        else:
            raise ValueError(f"Unknown prefix kind: {prefix_kind}")

        seqs, prefix_lens = [], []
        for i in range(vision.shape[0]):
            text_len = int(text_lengths[i].item())
            parts = [vision_emb[i], text_emb[i, :text_len]]
            # prefix_len marks the first target prediction point. Logits at
            # prefix_len - 1 predict target token 0 under causal LM shifting.
            prefix_lens.append(parts[0].shape[0] + parts[1].shape[0])
            if target_emb is not None and target_emb.shape[1] > 0:
                parts.append(target_emb[i])
            seqs.append(torch.cat(parts, dim=0))

        max_len = max(seq.shape[0] for seq in seqs)
        hidden = seqs[0].shape[-1]
        inputs = seqs[0].new_zeros((len(seqs), max_len, hidden))
        attention = torch.zeros((len(seqs), max_len), dtype=torch.long, device=vision.device)
        for i, seq in enumerate(seqs):
            inputs[i, : seq.shape[0]] = seq
            attention[i, : seq.shape[0]] = 1
        return inputs, attention, torch.tensor(prefix_lens, dtype=torch.long, device=vision.device)

    def _run(
        self,
        vision: torch.Tensor,
        text_ids: torch.Tensor,
        text_lengths: torch.Tensor,
        token_prefix: torch.Tensor | None,
        prefix_kind: str = "delta",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs, attention, prefix_lens = self._embed_and_pad(
            vision, text_ids, text_lengths, token_prefix, prefix_kind=prefix_kind
        )
        # The small LM consumes projected LAPA embeddings through inputs_embeds;
        # its own token embedding table is not used in this experiment.
        projected = self.input_proj(inputs)
        if hasattr(self.backbone, "gpt_neox"):
            # Pythia/GPT-NeoX can return the final transformer state directly,
            # avoiding the memory cost of retaining every layer's hidden state.
            outputs = self.backbone.gpt_neox(
                inputs_embeds=projected,
                attention_mask=attention,
                use_cache=False,
                return_dict=True,
            )
            small_hidden = outputs.last_hidden_state
        else:
            # Generic HF fallback for non-GPT-NeoX causal LMs.
            outputs = self.backbone(
                inputs_embeds=projected,
                attention_mask=attention,
                output_hidden_states=True,
                use_cache=False,
            )
            small_hidden = outputs.hidden_states[-1]
        hidden_4096 = self.output_proj(small_hidden)
        return hidden_4096, prefix_lens

    def forward(self, vision: torch.Tensor, text_ids: torch.Tensor, text_lengths: torch.Tensor, delta: torch.Tensor):
        hidden_4096, prefix_lens = self._run(vision, text_ids, text_lengths, delta, prefix_kind="delta")
        # Apply the frozen original LAPA delta head rather than the small LM head.
        logits = hidden_4096.matmul(self.delta_head)
        # Teacher forcing: append all gold delta embeddings, then gather the four
        # shifted positions that should predict those same delta ids.
        offsets = torch.arange(delta.shape[1], device=delta.device)[None, :]
        positions = prefix_lens[:, None] - 1 + offsets
        batch_idx = torch.arange(delta.shape[0], device=delta.device)[:, None]
        delta_logits = logits[batch_idx, positions]
        loss = F.cross_entropy(delta_logits.reshape(-1, delta_logits.shape[-1]), delta.reshape(-1))
        return loss, delta_logits

    def forward_action(
        self,
        vision: torch.Tensor,
        text_ids: torch.Tensor,
        text_lengths: torch.Tensor,
        action: torch.Tensor,
    ):
        hidden_4096, prefix_lens = self._run(vision, text_ids, text_lengths, action, prefix_kind="action")
        logits = self.action_head(hidden_4096)
        offsets = torch.arange(action.shape[1], device=action.device)[None, :]
        positions = prefix_lens[:, None] - 1 + offsets
        batch_idx = torch.arange(action.shape[0], device=action.device)[:, None]
        action_logits = logits[batch_idx, positions]
        loss = F.cross_entropy(action_logits.reshape(-1, action_logits.shape[-1]), action.reshape(-1))
        return loss, action_logits

    @torch.no_grad()
    def generate_delta(
        self,
        vision: torch.Tensor,
        text_ids: torch.Tensor,
        text_lengths: torch.Tensor,
        n_tokens: int = 4,
    ) -> torch.Tensor:
        generated = torch.empty((vision.shape[0], 0), dtype=torch.long, device=vision.device)
        for step in range(n_tokens):
            # Autoregressive decoding: embed previously predicted delta tokens
            # with frozen dte, then predict the next token from the last position.
            hidden_4096, prefix_lens = self._run(vision, text_ids, text_lengths, generated, prefix_kind="delta")
            logits = hidden_4096.matmul(self.delta_head)
            positions = prefix_lens + step - 1
            next_logits = logits[torch.arange(vision.shape[0], device=vision.device), positions]
            generated = torch.cat([generated, next_logits.argmax(dim=-1, keepdim=True)], dim=1)
        return generated

    @torch.no_grad()
    def generate_action(
        self,
        vision: torch.Tensor,
        text_ids: torch.Tensor,
        text_lengths: torch.Tensor,
        n_tokens: int = 7,
    ) -> torch.Tensor:
        generated = torch.empty((vision.shape[0], 0), dtype=torch.long, device=vision.device)
        for step in range(n_tokens):
            hidden_4096, prefix_lens = self._run(vision, text_ids, text_lengths, generated, prefix_kind="action")
            logits = self.action_head(hidden_4096)
            positions = prefix_lens + step - 1
            next_logits = logits[torch.arange(vision.shape[0], device=vision.device), positions]
            generated = torch.cat([generated, next_logits.argmax(dim=-1, keepdim=True)], dim=1)
        return generated


def save_adapter_checkpoint(path: str, model: FrozenLAPAAdapterPolicy, **metadata) -> None:
    torch.save({"model_state": model.state_dict(), "metadata": metadata}, path)


def load_adapter_checkpoint(path: str, frozen_lapa: dict[str, torch.Tensor], map_location="cpu", strict: bool = False):
    checkpoint = torch.load(path, map_location=map_location)
    metadata = checkpoint.get("metadata", {})
    model = FrozenLAPAAdapterPolicy(
        frozen_lapa=frozen_lapa,
        model_name=metadata.get("model_name", "EleutherAI/pythia-160m"),
        freeze_backbone=metadata.get("freeze_backbone", False),
    )
    model.load_state_dict(checkpoint["model_state"], strict=strict)
    return model, metadata
