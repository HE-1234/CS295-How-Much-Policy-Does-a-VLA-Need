from pathlib import Path
from typing import Optional, Sequence
import csv
import sys

import albumentations
import jax
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transforms3d.euler import euler2axangle

current_file = Path(__file__).resolve()
lapa_root = current_file.parents[4]
sys.path.insert(0, str(lapa_root))

from adapter_distill.data import LapaActionCollator, load_lapa_tokenizer
from adapter_distill.models import load_adapter_checkpoint, load_frozen_lapa
from latent_pretraining.vqgan import VQGAN


class LAPAAdapterInference:
    def __init__(
        self,
        checkpoint_path: str,
        frozen_lapa_path: str,
        vocab_file: str,
        vqgan_checkpoint: str,
        action_scale_file: str,
        policy_setup: str = "widowx_bridge",
        image_size: int = 256,
        action_scale: float = 1.0,
        max_text_len: int = 128,
        device: str | None = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        frozen = load_frozen_lapa(frozen_lapa_path, device=self.device)
        self.model, metadata = load_adapter_checkpoint(checkpoint_path, frozen, map_location=self.device, strict=False)
        self.model.to(self.device).eval()

        tokenizer = load_lapa_tokenizer(vocab_file)
        self.collate = LapaActionCollator(tokenizer=tokenizer, max_text_len=int(metadata.get("max_text_len", max_text_len)))
        self.vqgan = VQGAN(vqgan_checkpoint, replicate=False)
        self.preproc = albumentations.Compose([
            albumentations.LongestMaxSize(max_size=image_size),
            albumentations.Resize(image_size, image_size),
        ])

        self.action_scale_list = []
        with open(action_scale_file, "r") as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                self.action_scale_list.append([float(value) for value in row if value.strip()])

        self.image_size = image_size
        self.action_scale = action_scale
        self.policy_setup = policy_setup
        self.task_description = None
        self.previous_gripper_action = None
        self.sticky_action_is_on = False
        self.sticky_gripper_action = 0.0
        self.gripper_action_repeat = 0
        self.sticky_gripper_num_repeat = 15

    def _encode_image(self, image: np.ndarray) -> torch.Tensor:
        image_vqgan = self.preproc(image=image)["image"]
        image_vqgan = (image_vqgan / 127.5 - 1.0).astype(np.float32)[None]
        tokens = jax.device_get(self.vqgan.encode(image_vqgan))[1].astype(int).reshape(-1)
        return torch.tensor(tokens.tolist(), dtype=torch.long)

    def _format_instruction(self, task_description: str) -> str:
        if task_description.strip().startswith("<s>"):
            return task_description
        return f"<s> You are a helpful assistant. USER: What action should the robot take to `{task_description}` ASSISTANT:"

    def _tokens_to_actions(self, indices: Sequence[int]) -> list[float]:
        averaged_values = []
        for row_idx, idx in enumerate(indices):
            try:
                idx = int(idx)
                value1 = self.action_scale_list[row_idx][idx]
                value2 = self.action_scale_list[row_idx][idx + 1]
                average = (value1 + value2) / 2
            except Exception:
                average = 1.0
            averaged_values.append(average)
        return averaged_values

    def reset(self, task_description: str) -> None:
        self.task_description = task_description

    def step(self, image: np.ndarray, task_description: Optional[str] = None, *args, **kwargs):
        if task_description is not None and task_description != self.task_description:
            self.reset(task_description)

        assert image.dtype == np.uint8
        vision = self._encode_image(image)
        instruction = self._format_instruction(task_description or self.task_description or "")
        batch = self.collate([{"vision": vision, "instruction": instruction, "action": torch.zeros(7, dtype=torch.long)}])
        batch = {k: v.to(self.device) for k, v in batch.items()}

        with torch.no_grad():
            action_tokens = self.model.generate_action(
                batch["vision"],
                batch["text_ids"],
                batch["text_lengths"],
                n_tokens=7,
            )[0].detach().cpu().tolist()
        raw_actions = self._tokens_to_actions(action_tokens)

        raw_action = {
            "world_vector": np.array(raw_actions[:3]),
            "rotation_delta": np.array(raw_actions[3:6]),
            "open_gripper": np.array(raw_actions[6:7]),
        }

        action = {}
        action["world_vector"] = raw_action["world_vector"] * self.action_scale
        roll, pitch, yaw = np.asarray(raw_action["rotation_delta"], dtype=np.float64)
        action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
        action["rot_axangle"] = action_rotation_ax * action_rotation_angle * self.action_scale

        if self.policy_setup == "google_robot":
            current_gripper_action = raw_action["open_gripper"]
            if self.previous_gripper_action is None:
                relative_gripper_action = np.array([0])
            else:
                relative_gripper_action = self.previous_gripper_action - current_gripper_action
            self.previous_gripper_action = current_gripper_action

            if np.abs(relative_gripper_action) > 0.5 and self.sticky_action_is_on is False:
                self.sticky_action_is_on = True
                self.sticky_gripper_action = relative_gripper_action
            if self.sticky_action_is_on:
                self.gripper_action_repeat += 1
                relative_gripper_action = self.sticky_gripper_action
            if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
                self.sticky_action_is_on = False
                self.gripper_action_repeat = 0
                self.sticky_gripper_action = 0.0
            action["gripper"] = relative_gripper_action
        elif self.policy_setup == "widowx_bridge":
            action["gripper"] = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0
        else:
            raise ValueError(f"Unknown policy setup: {self.policy_setup}")

        action["terminate_episode"] = np.array([0.0])
        return raw_action, action

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        return np.array(Image.fromarray(image).resize((self.image_size, self.image_size)))

    def visualize_epoch(self, predicted_raw_actions: Sequence[dict], images: Sequence[np.ndarray], save_path: str) -> None:
        images = [self._resize_image(image) for image in images]
        action_dim_labels = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]
        img_strip = np.concatenate(np.array(images[::3]), axis=1)

        figure_layout = [["image"] * len(action_dim_labels), action_dim_labels]
        plt.rcParams.update({"font.size": 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])

        pred_actions = np.array(
            [
                np.concatenate([a["world_vector"], a["rotation_delta"], a["open_gripper"]], axis=-1)
                for a in predicted_raw_actions
            ]
        )
        for action_dim, action_label in enumerate(action_dim_labels):
            axs[action_label].plot(pred_actions[:, action_dim], label="predicted action")
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel("Time in one episode")

        axs["image"].imshow(img_strip)
        axs["image"].set_xlabel("Time in one episode (subsampled)")
        plt.legend()
        plt.savefig(save_path)
        plt.close(fig)
