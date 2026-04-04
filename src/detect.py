import argparse
from ultralytics import YOLO
from pathlib import Path

def run_inference(model_path, source, conf_thres=0.25, save=True, show=False):
    
    model = YOLO(model_path)
    
    results = model(
        source=source,
        conf=conf_thres,
        save=save,
        show=show,
        project='runs/detect',
        name='inference',
        exist_ok=True
    )
    
    print(f"\nInference completed. Results saved to runs/detect/inference/")
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run YOLOv8 inference on images/videos')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model .pt file')
    parser.add_argument('--source', type=str, required=True, help='Path to image, video, or directory')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--save', action='store_true', default=True, help='Save results')
    parser.add_argument('--show', action='store_true', help='Display results')
    
    args = parser.parse_args()
    run_inference(model_path=args.model, source=args.source, conf_thres=args.conf,
                  save=args.save, show=args.show)