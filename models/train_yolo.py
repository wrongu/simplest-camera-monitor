from pathlib import Path

import torch
from ultralytics import YOLO


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, default="yolo26n")
    parser.add_argument("--pretrained", type=bool, default=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--freeze", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.pretrained:
        model_slug = args.base_model + ".pt"
    else:
        model_slug = args.base_model + ".yaml"

    # Load a YOLO26n PyTorch model
    model = YOLO(model_slug, task="detect")
    model.train(
        data=args.dataset,
        epochs=args.epochs,
        imgsz=args.imgsz,
        freeze=args.freeze,
        device=device,
    )


if __name__ == "__main__":
    main()
