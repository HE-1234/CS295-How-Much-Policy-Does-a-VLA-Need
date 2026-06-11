import json
from dataclasses import dataclass
from typing import Iterable, Optional

import sentencepiece as spm
import torch
from torch.utils.data import Dataset


class LatentActionDataset(Dataset):
    def __init__(self, records: list[dict]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        item = self.records[idx]
        # The LAPA JSONL stores token ids as strings; convert only at batch time.
        return {
            "vision": torch.tensor([int(v) for v in item["vision"]], dtype=torch.long),
            "instruction": item["instruction"],
            "delta": torch.tensor([int(d) for d in item["delta"]], dtype=torch.long),
        }


class SimplerActionDataset(Dataset):
    def __init__(self, records: list[dict]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        item = self.records[idx]
        return {
            "vision": torch.tensor([int(v) for v in item["vision"]], dtype=torch.long),
            "instruction": item["instruction"],
            "action": torch.tensor([int(a) for a in item["action"]], dtype=torch.long),
        }


def load_jsonl_records(
    path: str,
    start: int = 0,
    limit: Optional[int] = None,
    skip_bad_records: bool = False,
) -> list[dict]:
    records: list[dict] = []
    end = None if limit is None else start + limit
    with open(path, "r") as f:
        for idx, line in enumerate(f):
            if idx < start:
                continue
            if end is not None and idx >= end:
                break
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                line_no = idx + 1
                if not skip_bad_records:
                    raise ValueError(f"Malformed JSONL record in {path} at line {line_no}: {exc}") from exc
                print(f"Skipping malformed JSONL record in {path} at line {line_no}: {exc}")
    return records


class LapaSentencePieceTokenizer:
    def __init__(self, vocab_file: str):
        # Use SentencePiece directly because HF LlamaTokenizer may only expose
        # special tokens for this LWM/LAPA tokenizer.model in newer environments.
        self.processor = spm.SentencePieceProcessor(model_file=vocab_file)
        self.pad_id = self.processor.eos_id()

    def encode(self, text: str, max_length: int) -> list[int]:
        ids = self.processor.encode(text, out_type=int)
        return ids[:max_length]


def load_lapa_tokenizer(vocab_file: str) -> LapaSentencePieceTokenizer:
    return LapaSentencePieceTokenizer(vocab_file)


@dataclass
class LapaCollator:
    tokenizer: LapaSentencePieceTokenizer
    max_text_len: int = 128

    def __call__(self, batch: Iterable[dict]) -> dict[str, torch.Tensor]:
        batch = list(batch)
        tokenized = [self.tokenizer.encode(item["instruction"], self.max_text_len) for item in batch]
        # Pad text only for batching. The model uses text_lengths to remove this
        # padding before concatenating [vision | text | delta] embeddings.
        max_len = max(len(ids) for ids in tokenized)
        text_ids = torch.full((len(batch), max_len), self.tokenizer.pad_id, dtype=torch.long)
        text_lengths = torch.tensor([len(ids) for ids in tokenized], dtype=torch.long)
        for i, ids in enumerate(tokenized):
            text_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        return {
            "vision": torch.stack([item["vision"] for item in batch]),
            "text_ids": text_ids,
            "text_lengths": text_lengths,
            "delta": torch.stack([item["delta"] for item in batch]),
        }


@dataclass
class LapaActionCollator:
    tokenizer: LapaSentencePieceTokenizer
    max_text_len: int = 128

    def __call__(self, batch: Iterable[dict]) -> dict[str, torch.Tensor]:
        batch = list(batch)
        tokenized = [self.tokenizer.encode(item["instruction"], self.max_text_len) for item in batch]
        max_len = max(len(ids) for ids in tokenized)
        text_ids = torch.full((len(batch), max_len), self.tokenizer.pad_id, dtype=torch.long)
        text_lengths = torch.tensor([len(ids) for ids in tokenized], dtype=torch.long)
        for i, ids in enumerate(tokenized):
            text_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        return {
            "vision": torch.stack([item["vision"] for item in batch]),
            "text_ids": text_ids,
            "text_lengths": text_lengths,
            "action": torch.stack([item["action"] for item in batch]),
        }
