# RoadAssestIdentification

## Navigation

- Repository root: `README.md`
- Configuration: `configs/hyperparams.yaml`
- Dataset config: `dataset/data.yaml`
- Requirements: `requirements.txt`, `requirements-gpu.txt`
- Training script: `src/train.py`
- Detection/inference script: `src/detect.py`

## Project structure

Top-level layout and purpose of each item:

- `configs/` – YAML configuration files controlling training hyperparameters and experiment settings. Edit `configs/hyperparams.yaml` to change learning rates, batch sizes, augmentation and other options.
- `dataset/` – dataset configuration and manifests. `dataset/data.yaml` contains paths and dataset split information used by the training script.
- `src/` – primary code for training and inference.
	- `src/train.py` – training script. Trains the model using configs and dataset manifests.
	- `src/detect.py` – detection / inference script. Run this to perform inference on images or video using trained weights.
- `requirements.txt` – CPU/standard Python dependencies.
- `requirements-gpu.txt` – GPU-enabled dependencies (a CUDA-capable PyTorch build and other GPU-accelerated libs). Use this if you have an NVIDIA GPU and compatible CUDA drivers.

## Getting started

These instructions assume a Windows environment (PowerShell). Recommended Python: 3.8–3.11.

1) Clone the repository

```powershell
git clone https://github.com/AverageDude0716/RoadAssestIdentification.git
cd RoadAssestIdentification
git checkout main
```

2) Create and activate a virtual environment (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3) Install dependencies

- If you do NOT have a CUDA-capable GPU or want the CPU-only install:

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

- If you have an NVIDIA GPU and want GPU-accelerated dependencies, ensure you have a compatible CUDA toolkit and drivers, then:

```powershell
pip install --upgrade pip
pip install -r requirements-gpu.txt
```

Notes:
- If `requirements-gpu.txt` pins a CUDA-specific PyTorch wheel, ensure the wheel and your local CUDA driver are compatible.
- If you run into dependency conflicts, consider using a fresh virtual environment or conda.

## Quick usage: training

High-level contract for `src/train.py`:
- Inputs: dataset manifest (`dataset/data.yaml`) and hyperparameter config (`configs/hyperparams.yaml`).
- Outputs: trained model weights (usually written to a `runs/` or `weights/` folder) and training logs.

A typical training run (example):

```powershell
# Example -- adjust flags to match your local scripts and config file names
python src/train.py --data dataset/data.yaml --cfg configs/hyperparams.yaml
```

What to change:
- Edit `configs/hyperparams.yaml` for learning rate, epochs, batch size and other options.
- Edit `dataset/data.yaml` to point to your train/val/test image folders and label formats.

Common tips:
- Use the GPU-enabled environment for faster training.
- Reduce batch size if you run out of GPU memory.

## Quick usage: detection / inference

High-level contract for `src/detect.py`:
- Inputs: trained weights, source images/videos.
- Outputs: detection results (annotated images, prediction files, or console output) depending on the script options.

A typical detection run (example):

```powershell
# Example usage - replace <weights> and <source> with your paths
python src/detect.py --weights runs/exp/weights/best.pt --source data/images/test.jpg --conf 0.25
```

Notes:
- If your script accepts video input, pass a video path or a webcam index (e.g. `--source 0`).
- Adjust `--conf` (confidence threshold) or similar flags to tune precision/recall for inference.

## Troubleshooting

- CUDA / GPU not detected: confirm CUDA drivers and that you installed GPU-specific dependencies. Run a simple PyTorch check:

```powershell
python -c "import torch; print('cuda', torch.cuda.is_available()); print(torch.__version__)"
```

- Dependency errors: create a fresh virtual environment and re-install. If a specific package fails, try installing it separately to get better error messages.

<!-- ## Next steps and recommended improvements

- Add an example `weights/` or `runs/` folder with a pre-trained model for quick inference tests.
- Add a `scripts/` or `Makefile` for common commands (train, evaluate, detect).
- Add unit tests or a minimal integration test that runs a tiny training loop to verify end-to-end functionality. -->

## Credits

Project: `RoadAssestIdentification` by AverageDude0716.

If you want any changes to this README (more examples, expanded usage for specific flags, or adding quick start scripts), tell me which area to expand and I will update it.
