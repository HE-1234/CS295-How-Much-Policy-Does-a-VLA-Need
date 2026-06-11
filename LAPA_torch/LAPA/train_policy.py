import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW
import wandb
from tqdm import tqdm


class LatentActionDataset(Dataset):
    def __init__(self, jsonl_path):
        with open(jsonl_path) as f:
            self.data = [json.loads(l) for l in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        vision = torch.tensor([int(v) for v in item['vision']], dtype=torch.long)
        delta  = torch.tensor([int(d) for d in item['delta']],  dtype=torch.long)
        return vision, item['instruction'], delta


class LAPAPolicy(nn.Module):
    def __init__(self, llm_name, vision_vocab=8192, num_act_tokens=8, act_seq_len=4):
        super().__init__()
        self.act_seq_len = act_seq_len

        # LLM backbone — only this changes across experiments
        self.llm       = AutoModelForCausalLM.from_pretrained(llm_name, dtype=torch.float32)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        llm_hidden = self.llm.config.hidden_size

        # Visual embedding: maps VQGAN vocab (8192) → LLM hidden dim
        # Kept separate from LLM tokenizer so vocab sizes never conflict
        self.vision_embed = nn.Embedding(vision_vocab, llm_hidden)

        # Add 8 discrete action tokens to LLM vocabulary
        act_tokens = [f"<ACT_{i}>" for i in range(num_act_tokens)]
        self.tokenizer.add_special_tokens({"additional_special_tokens": act_tokens})
        self.llm.resize_token_embeddings(len(self.tokenizer))
        self.act_token_start = self.tokenizer.convert_tokens_to_ids("<ACT_0>")

    def forward(self, vision, instructions, delta):
        device = vision.device
        B = vision.shape[0]

        # 1. Vision: [B, 256, H]
        vis_emb = self.vision_embed(vision)

        # 2. Language: [B, T, H]
        text_enc = self.tokenizer(
            instructions, return_tensors="pt",
            padding=True, truncation=True, max_length=64
        ).to(device)
        text_emb = self.llm.get_input_embeddings()(text_enc.input_ids)

        # 3. Action tokens: [B, 4, H]
        act_ids  = delta + self.act_token_start
        act_emb  = self.llm.get_input_embeddings()(act_ids)

        # 4. Concatenate: [vis(256) | text(T) | act(4)]
        inputs_embeds = torch.cat([vis_emb, text_emb, act_emb], dim=1)

        # 5. Labels: -100 masks vision+text; only action positions are supervised
        vis_len  = vis_emb.shape[1]
        text_len = text_emb.shape[1]
        labels = torch.full(
            (B, vis_len + text_len + self.act_seq_len),
            fill_value=-100, dtype=torch.long, device=device
        )
        labels[:, vis_len + text_len:] = act_ids

        return self.llm(inputs_embeds=inputs_embeds, labels=labels).loss

    def compute_accuracy(self, vision, instructions, delta):
        device = vision.device
        vis_emb  = self.vision_embed(vision)
        text_enc = self.tokenizer(
            instructions, return_tensors="pt",
            padding=True, truncation=True, max_length=64
        ).to(device)
        text_emb = self.llm.get_input_embeddings()(text_enc.input_ids)
        act_ids  = delta + self.act_token_start
        act_emb  = self.llm.get_input_embeddings()(act_ids)
        inputs_embeds = torch.cat([vis_emb, text_emb, act_emb], dim=1)

        with torch.no_grad():
            logits = self.llm(inputs_embeds=inputs_embeds).logits

        vis_len  = vis_emb.shape[1]
        text_len = text_emb.shape[1]
        # logits[i] predicts token[i+1], so logits at (vis+text-1)..(vis+text+2)
        # predict action tokens 0..3 respectively
        act_logits = logits[:, vis_len + text_len - 1 : vis_len + text_len + 3, :]
        pred = act_logits.argmax(dim=-1)
        return (pred == act_ids).float().mean().item()


def train(llm_name, train_path, val_path,
          num_epochs=3, batch_size=16, lr=1e-4):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | LLM: {llm_name}")

    model    = LAPAPolicy(llm_name).to(device)
    train_dl = DataLoader(LatentActionDataset(train_path), batch_size=batch_size,
                          shuffle=True,  num_workers=0)
    val_dl   = DataLoader(LatentActionDataset(val_path),   batch_size=batch_size,
                          shuffle=False, num_workers=0)

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    run_name = llm_name.replace("/", "_")
    wandb.init(project="lapa-scaling", name=run_name,
               config={"llm": llm_name, "epochs": num_epochs,
                       "batch_size": batch_size, "lr": lr})

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for step, (vision, instructions, delta) in enumerate(tqdm(train_dl)):
            vision, delta = vision.to(device), delta.to(device)
            loss = model(vision, instructions, delta)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            if step % 100 == 0:
                wandb.log({"train/loss": loss.item(), "epoch": epoch})

        model.eval()
        accs = []
        for vision, instructions, delta in val_dl:
            vision, delta = vision.to(device), delta.to(device)
            accs.append(model.compute_accuracy(vision, instructions, delta))

        avg_loss = total_loss / len(train_dl)
        avg_acc  = sum(accs) / len(accs)
        print(f"Epoch {epoch} | loss={avg_loss:.4f} | val_acc={avg_acc:.4f}")
        wandb.log({"train/avg_loss": avg_loss, "val/accuracy": avg_acc, "epoch": epoch})

    torch.save(model.state_dict(), f"ckpt_{run_name}.pt")
    print(f"Saved ckpt_{run_name}.pt")
    wandb.finish()


if __name__ == "__main__":
    train(
        llm_name   = "EleutherAI/pythia-410m",  # ← change only this line per experiment
        train_path = "train_small.jsonl",
        val_path   = "val_small.jsonl",
        batch_size = 8,
    )
