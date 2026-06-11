# LAPA GPU Environment Walkthrough

This walkthrough creates a new environment for running LAPA on the local RTX A6000 GPUs. It uses `conda` only to create the isolated Python environment, and plain `pip` for all Python packages.

The goal is to avoid the problem seen in `lapaTemp`: JAX was installed with a CUDA/cuDNN build that expected cuDNN 8.9, but the environment exposed incompatible CUDA/cuDNN libraries. The main rule for this setup is:

Do not install `tensorflow[and-cuda]` in the LAPA JAX environment.

Plain `tensorflow` is okay because `latent_pretraining/train.py` imports TensorFlow. The `[and-cuda]` extra is the part that pulls in a separate CUDA/cuDNN stack and can break JAX GPU initialization.

## 0. Starting Assumptions

Run these commands from the machine where the repo already exists:

```bash
cd /home/jianch2/CS295/LAPA_torch/LAPA
```

The machine has NVIDIA GPUs. Verify that first:

```bash
nvidia-smi
```

On this machine, `nvidia-smi` showed RTX A6000 GPUs and driver `570.211.01`, which is new enough for CUDA 12 JAX wheels. If `nvidia-smi` fails, stop and fix the driver/session before touching Python packages.

## 1. Create A Fresh Conda Env

Use a new env so the known-working-but-CPU `lapaTemp` env remains untouched.

```bash
conda create -n lapa_gpu python=3.10 -y
conda activate lapa_gpu
cd /home/jianch2/CS295/LAPA_torch/LAPA
```

Upgrade packaging tools:

```bash
python -m pip install --upgrade pip setuptools wheel
```

Check that the Python you are using is the new env:

```bash
which python
python --version
```

Expected path shape:

```text
/home/jianch2/miniconda3/envs/lapa_gpu/bin/python
```

Expected Python:

```text
Python 3.10.x
```

## 2. Keep System CUDA Libraries Out Of The Way

JAX is sensitive to `LD_LIBRARY_PATH`. The official JAX docs recommend making sure `LD_LIBRARY_PATH` does not override the CUDA libraries installed with the JAX wheel.

For this shell, run:

```bash
unset LD_LIBRARY_PATH
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_CPP_MIN_LOG_LEVEL=2
```

Use those same exports whenever you start a new shell for LAPA work.

## 3. Install JAX With Matching CUDA/cuDNN

This repo uses older JAX-era dependencies: `flax==0.7.0`, `optax==0.1.7`, and `chex==0.1.82`. Use `jax==0.4.23`, matching what was already present in `lapaTemp`, but install it with a controlled cuDNN 8.x stack.

```bash
python -m pip install \
  "jax[cuda12_pip]==0.4.23" \
  "nvidia-cudnn-cu12>=8.9,<9" \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Verify JAX before installing the rest of the repo:

```bash
unset LD_LIBRARY_PATH
python -c "import jax, jaxlib; print('jax:', jax.__version__); print('jaxlib:', jaxlib.__version__); print('backend:', jax.default_backend()); print('devices:', jax.devices())"
```

Good output should include:

```text
jax: 0.4.23
backend: gpu
CudaDevice
```

Bad output examples:

```text
backend: cpu
```

or:

```text
CUDA backend failed to initialize
```

If that happens, do not continue. First run:

```bash
echo "$LD_LIBRARY_PATH"
python -m pip show jax jaxlib nvidia-cudnn-cu12 nvidia-cublas-cu12
```

The most common fix is to make sure `LD_LIBRARY_PATH` is unset and reinstall the JAX command above.
We stopped here, everything seems to work so far, but we may need to let the model know about our updated code

## 4. Install LAPA Repo Dependencies Without TensorFlow CUDA Extras

The repo `requirements.txt` contains this line:

```text
tensorflow[and-cuda]
```

Do not install that line. Create a temporary filtered requirements file:

```bash
grep -v '^tensorflow\[and-cuda\]$' requirements.txt > /tmp/lapa_requirements_no_tf_cuda.txt
```

Install the filtered requirements:

```bash
python -m pip install -r /tmp/lapa_requirements_no_tf_cuda.txt
```

Now install plain TensorFlow, without CUDA extras:

```bash
python -m pip install "tensorflow==2.21.0"
```

Why this split matters:

- `latent_pretraining/train.py` imports TensorFlow, so the package is needed.
- `tensorflow[and-cuda]` can install a different cuDNN stack and break JAX.
- LAPA model execution should use JAX GPU, not TensorFlow GPU.

## 5. Install PyTorch For `train_policy.py`

The file `train_policy.py` is a separate PyTorch training script. It uses:

```python
torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Install PyTorch CUDA 11.8 wheels. CUDA 11.8 wheels are fine on this machine because the NVIDIA driver is new enough, and they avoid overwriting the CUDA 12 cuDNN package JAX depends on.

```bash
python -m pip install \
  torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
  --index-url https://download.pytorch.org/whl/cu118
```

Verify PyTorch GPU:

```bash
python -c "import torch; print('torch:', torch.__version__); print('torch cuda:', torch.version.cuda); print('cuda available:', torch.cuda.is_available()); print('device count:', torch.cuda.device_count()); print('device 0:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

Good output should include:

```text
cuda available: True
device 0: NVIDIA RTX A6000
```

## 6. Run A Final Package Sanity Check

Run:

```bash
python -m pip check
```

If it reports no broken requirements, continue.

Then verify JAX again after all installs:

```bash
unset LD_LIBRARY_PATH
python -c "import jax; print('backend:', jax.default_backend()); print('devices:', jax.devices())"
```

Then verify PyTorch again:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

Both JAX and PyTorch should see GPU before running models.

## 7. Download Or Verify LAPA Checkpoints

The inference script expects these files:

```text
lapa_checkpoints/tokenizer.model
lapa_checkpoints/vqgan
lapa_checkpoints/params
```

Check whether they already exist:

```bash
ls -lh lapa_checkpoints/tokenizer.model lapa_checkpoints/vqgan lapa_checkpoints/params
```

If they are missing, download them:

```bash
mkdir -p lapa_checkpoints

wget -nc -P lapa_checkpoints \
  https://huggingface.co/latent-action-pretraining/LAPA-7B-openx/resolve/main/tokenizer.model

wget -nc -P lapa_checkpoints \
  https://huggingface.co/latent-action-pretraining/LAPA-7B-openx/resolve/main/vqgan

wget -nc -P lapa_checkpoints \
  https://huggingface.co/latent-action-pretraining/LAPA-7B-openx/resolve/main/params
```

## 8. Run LAPA Inference On GPU

Use one GPU first:

```bash
conda activate lapa_gpu
cd /home/jianch2/CS295/LAPA_torch/LAPA

unset LD_LIBRARY_PATH
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_CPP_MIN_LOG_LEVEL=2

CUDA_VISIBLE_DEVICES=0 python -m latent_pretraining.inference
```

Expected model output shape:

```text
latent action is [[... ... ... ...]]
```

The exact token IDs can differ, but it should produce four latent action tokens. The important difference from `lapaTemp` is that there should be no message like:

```text
CUDA backend failed to initialize
```

To watch GPU usage while inference runs, open another terminal:

```bash
watch -n 1 nvidia-smi
```

## 9. Run The Local PyTorch Policy Training Script

`train_policy.py` is not the official JAX LAPA fine-tuning path. It is a smaller PyTorch policy-training experiment in this repo.

It expects:

```text
train_small.jsonl
val_small.jsonl
```

Each JSONL row must provide:

```json
{
  "vision": ["256 image-token ids here"],
  "instruction": "text instruction",
  "delta": ["4 latent action token ids here"]
}
```

Before training, verify the files exist:

```bash
ls -lh train_small.jsonl val_small.jsonl
```

If you do not want online Weights & Biases logging, run:

```bash
export WANDB_MODE=offline
```

Run PyTorch training on GPU 0:

```bash
conda activate lapa_gpu
cd /home/jianch2/CS295/LAPA_torch/LAPA

CUDA_VISIBLE_DEVICES=0 python train_policy.py
```

The script should print:

```text
Device: cuda | LLM: EleutherAI/pythia-410m
```

If it prints `Device: cpu`, stop and rerun the PyTorch verification command from section 5.

## 10. Official LAPA Fine-Tuning Notes

The official fine-tuning scripts are:

```text
scripts/finetune_real.sh
scripts/finetune_simpler.sh
```

They are JAX training scripts, not PyTorch scripts.

Before running either script, inspect it:

```bash
sed -n '1,140p' scripts/finetune_real.sh
sed -n '1,140p' scripts/finetune_simpler.sh
```

Important: both scripts currently contain this placeholder:

```bash
export absolute_path=
```

You must edit that to:

```bash
export absolute_path=/home/jianch2/CS295/LAPA_torch/LAPA
```

The scripts also assume large multi-GPU training. The README says the authors used 4x 80GB A100 GPUs for fine-tuning. This machine has 48GB RTX A6000 GPUs, so the default batch sizes may be too large.

Default examples in the scripts:

```bash
--mesh_dim='!-1,4,1,1'
--train_dataset.json_delta_action_dataset.batch_size=128
```

For a first smoke test, reduce the training size before trying a full run:

```bash
--total_steps=5
--save_milestone_freq=5
--train_dataset.json_delta_action_dataset.batch_size=4
```

Use 4 GPUs only after single-GPU inference works:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/finetune_real.sh
```

If the script OOMs, reduce the batch size first. If it still OOMs, reduce sequence length or use fewer scan/chunk settings only after checking the JAX error.

## 11. Official Latent Pretraining Notes

The official latent pretraining script is:

```text
scripts/latent_pretrain_openx.sh
```

The README says the authors used 8 H100 GPUs for 34 hours. Do not expect that script to run at the original settings on 4x RTX A6000 without reducing batch size and possibly other model settings.

For a setup check, do not start with 70,000 steps. First edit:

```bash
--total_steps=5
--save_milestone_freq=5
--train_dataset.json_delta_dataset.batch_size=4
```

Also set:

```bash
export absolute_path=/home/jianch2/CS295/LAPA_torch/LAPA
```

Then run only after JAX GPU verification passes:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/latent_pretrain_openx.sh
```

## 12. Common Failure Modes

### JAX says CPU only

Run:

```bash
unset LD_LIBRARY_PATH
python -c "import jax; print(jax.default_backend()); print(jax.devices())"
```

If it still says CPU, inspect packages:

```bash
python -m pip show jax jaxlib nvidia-cudnn-cu12 nvidia-cublas-cu12
```

You want:

```text
jax 0.4.23
jaxlib 0.4.23+cuda12.cudnn89
nvidia-cudnn-cu12 8.9.x
```

If `nvidia-cudnn-cu12` is `9.x`, reinstall:

```bash
python -m pip uninstall -y jax jaxlib nvidia-cudnn-cu12
python -m pip install \
  "jax[cuda12_pip]==0.4.23" \
  "nvidia-cudnn-cu12>=8.9,<9" \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### You see `CUDA backend failed to initialize`

This is the same class of issue as above. Most likely causes:

- `LD_LIBRARY_PATH` points to system CUDA/cuDNN.
- `tensorflow[and-cuda]` was installed after JAX.
- A later package install upgraded `nvidia-cudnn-cu12` to 9.x.

Fix in this order:

```bash
unset LD_LIBRARY_PATH
python -m pip show nvidia-cudnn-cu12 jaxlib
```

If cuDNN is 9.x, reinstall JAX as shown above.

### PyTorch says CUDA is unavailable

Run:

```bash
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

If `nvidia-smi` works but PyTorch says `False`, reinstall the CUDA PyTorch wheel:

```bash
python -m pip install --force-reinstall \
  torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
  --index-url https://download.pytorch.org/whl/cu118
```

### Training runs out of memory

For JAX LAPA fine-tuning:

- Lower `batch_size`.
- Lower `seq_length`.
- Start with `total_steps=5` as a smoke test.
- Use fewer visible GPUs only if the mesh config matches the visible GPU count.

For `train_policy.py`:

- Lower `batch_size` in the `train(...)` call at the bottom of the file.
- Use a smaller `llm_name`.
- Start with `EleutherAI/pythia-410m` as the script already does.

## 13. Daily Use Commands

Every new terminal:

```bash
conda activate lapa_gpu
cd /home/jianch2/CS295/LAPA_torch/LAPA
unset LD_LIBRARY_PATH
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_CPP_MIN_LOG_LEVEL=2
```

Check JAX:

```bash
python -c "import jax; print(jax.default_backend()); print(jax.devices())"
```

Check PyTorch:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

Run LAPA inference:

```bash
CUDA_VISIBLE_DEVICES=0 python -m latent_pretraining.inference
```

Run local PyTorch policy training:

```bash
CUDA_VISIBLE_DEVICES=0 python train_policy.py
```

## References

- JAX installation docs: https://docs.jax.dev/en/latest/installation.html
- PyTorch local installation selector: https://pytorch.org/get-started/locally/
- PyTorch previous versions: https://pytorch.org/get-started/previous-versions/
