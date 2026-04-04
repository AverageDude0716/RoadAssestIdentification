# RoadAssestIdentification

## Project structure

- `configs/`
	- `hyperparams.yaml` — training hyperparameters used by the model (learning rate, augmentation, loss gains, etc.).
- `dataset/`
	- `data.yaml` — dataset manifest describing `path`, `train`, `val`, `test`, `nc` (number of classes) and `names` (class labels).
    - Currently just a placeholder, soon to be replaced by the actual dataset
- `src/`
	- `train.py` — training script. Uses `ultralytics.YOLO` to train a model. Key CLI flags: `--model`, `--epochs`, `--patience`, `--imgsz`, `--batch`, `--device`.
	- `detect.py` — inference script. Uses `ultralytics.YOLO` to run inference. Key CLI flags: `--model`, `--source`, `--conf`, `--save`, `--show`.
- `requirements.txt` — CPU-only Python dependencies.
- `requirements-gpu.txt` — GPU (CUDA) compatible dependencies (points at PyTorch CUDA wheel index). Use only if you have matching NVIDIA drivers/CUDA.

## Getting started

The steps below show cloning the repo, creating a virtual environment (Windows and macOS examples), and installing dependencies.

1) Clone the repository

```powershell
git clone https://github.com/AverageDude0716/RoadAssestIdentification.git
cd RoadAssestIdentification
git checkout main
```

2) Create and activate a virtual environment

Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS (bash/zsh)

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3) Install dependencies

- CPU-only (works on most machines, including macOS Intel/Apple Silicon if you choose a CPU PyTorch build):

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

- GPU (NVIDIA CUDA) — Windows/Linux with compatible drivers: the `requirements-gpu.txt` points to the PyTorch cu130 wheel index. Make sure your CUDA driver version is compatible with the wheel before installing.

```powershell
pip install --upgrade pip
pip install -r requirements-gpu.txt
```

Notes for macOS (Apple Silicon):
- Recent PyTorch builds support MPS (Apple Metal). If you want GPU-like speed on M1/M2, follow https://pytorch.org/ and choose the macOS / MPS install option. Then install `ultralytics` and the other packages from `requirements.txt`.
- Test PyTorch device availability on your machine:

```bash
python -c "import torch; print('cuda', torch.cuda.is_available()); print('mps', getattr(torch.backends,'mps',None) and torch.backends.mps.is_available()); print(torch.__version__)"
```

## Training

What the `train.py` script does
- Loads a YOLO model (default `yolov8n.pt` but you can change it) using `ultralytics.YOLO` and calls `model.train(...)` with dataset `dataset/data.yaml` and hyperparameters from `configs/hyperparams.yaml` (you can edit that file for custom hyperparameters).

Important details (from `src/train.py`)
- Default model: `yolov8n.pt` (small, fast). You can pass larger models like `yolov8s.pt`, `yolov8m.pt`, `yolov8l.pt`, `yolov8x.pt` via `--model`.
- Default storage: training uses `project='runs/detect'` and `name='train'`. The best weights are saved to `runs/detect/train/weights/best.pt` (the script prints the exact save path after training).
- Device auto-detection: by default the script sets device to `cuda` if available, otherwise `cpu`. You can override with `--device`.

Example training commands

Windows (PowerShell)

```powershell
# simple train with defaults
python src/train.py --model yolov8n.pt --epochs 100 --batch 16

# specify device explicitly (cuda, cpu, 0, cuda:0)
python src/train.py --model yolov8n.pt --epochs 50 --batch 8 --device cuda
```

macOS (bash/zsh) — prefer `mps` if you installed an MPS-enabled PyTorch

```bash
python src/train.py --model yolov8n.pt --epochs 50 --batch 8 --device mps
```

Tuning notes
- To change learning-rate, augmentation, or loss weights, edit `configs/hyperparams.yaml`.
- If you run out of GPU memory, lower `--batch` and/or `--imgsz`.
- Early stopping is controlled via `--patience`.

## Inference / Detection

What the `detect.py` script does
- Loads a trained `.pt` model using `ultralytics.YOLO` and runs inference on the given source. Results are saved under `runs/detect/inference/` (project=name `inference`).

Important CLI flags (from `src/detect.py`)
- `--model` — path to the trained model `.pt` file (required)
- `--source` — path to an image, video, or directory (required)
- `--conf` — confidence threshold (default 0.25)
- `--save` — whether to save results (default True)
- `--show` — whether to display results interactively

Example inference commands

Windows (PowerShell)

```powershell
python src/detect.py --model runs/detect/train/weights/best.pt --source dataset/test/images/sample.jpg --conf 0.25 --save

# run on webcam (if supported)
python src/detect.py --model runs/detect/train/weights/best.pt --source 0 --conf 0.25 --show
```

macOS (bash/zsh)

```bash
python src/detect.py --model runs/detect/train/weights/best.pt --source dataset/test/images/sample.jpg --conf 0.25 --save
```

Output location
- Inference outputs (images/videos with detections) are written to `runs/detect/inference/` by default. The training script saves best weights to `runs/detect/train/weights/best.pt`.

## Dataset format

- `dataset/data.yaml` is used by Ultralytics training API and should contain `path`, `train`, `val`, (optional `test`), `nc`, and `names`. Example:

```yaml
path: ./dataset
train: train/images
val: val/images
test: test/images
nc: 3
names:
	0: pothole
	1: railing
	2: barrier
```

- Labels should follow the YOLO format (one txt file per image, class and normalized box coordinates), or adapt `data.yaml` and preprocessing accordingly.

## Troubleshooting

- PyTorch/CUDA not detected: ensure GPU drivers & CUDA toolkit are installed and compatible with the PyTorch wheel in `requirements-gpu.txt`. Check with:

```powershell
python -c "import torch; print('cuda', torch.cuda.is_available()); print(torch.__version__)"
```

- MPS (Apple Silicon) issues: use the official PyTorch macOS install instructions and test MPS availability as shown above.
- Ultralytics errors: ensure `ultralytics` version is compatible with your installed PyTorch. If API changes, consult https://docs.ultralytics.com/.
- Out of memory: reduce `--batch` and/or `--imgsz`, or train on a machine with more GPU memory.


