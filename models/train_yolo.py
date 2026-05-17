from pathlib import Path

import torch
from ultralytics import YOLO


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, default=Path("models") / "yolo26n.pt")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--freeze", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--export", type=str, default=None, choices=["onnx", "engine", "coreml"])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = YOLO(str(args.base_model), task="detect")
    model.train(
        data=args.dataset,
        epochs=args.epochs,
        imgsz=args.imgsz,
        freeze=args.freeze,
        device=device,
    )

    if args.export:
        model.export(format=args.export)


if __name__ == "__main__":
    main()
