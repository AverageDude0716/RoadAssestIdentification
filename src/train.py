import argparse
import torch
from ultralytics import YOLO
import yaml

def get_default_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def train_model(model_name='yolov8n.pt', epochs=100, patience=10, img_size=640, batch_size=16, device=None):
    if device is None:
        device = get_default_device()
    
    print(f"Using device: {device}")
    
    model = YOLO(model_name)

    with open('configs/hyperparams.yaml', 'r') as f:
        hyp = yaml.safe_load(f)
    
    results = model.train(
        data='dataset/data.yaml',
        epochs=epochs,
        patience=patience,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        project='runs/detect',
        name='train',
        exist_ok=True,
        pretrained=True,
        verbose=True,
        **hyp
    )
    
    print(f"\nTraining completed! Best model saved at: {results.save_dir}/weights/best.pt")
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train YOLOv8 for Road Asset Detection')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                        help='Model name: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--device', type=str, default=None,
                        help='Device: cpu, cuda, 0, cuda:0. Auto-detects if not provided.')
    
    args = parser.parse_args()
    train_model(model_name=args.model, epochs=args.epochs, patience=args.patience,
                img_size=args.imgsz, batch_size=args.batch, device=args.device)