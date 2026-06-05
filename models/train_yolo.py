from pathlib import Path

from ultralytics import YOLO


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, default=Path("models") / "yolo26n.pt")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--cfg", type=Path, default=None, help="Further config options in yaml training config file"
    )
    parser.add_argument("--export", type=str, default=None, choices=["onnx", "engine", "coreml"])
    args = parser.parse_args()

    if args.cfg is not None:
        assert args.cfg.exists()

    model = YOLO(str(args.base_model), task="detect")
    model.train(data=args.dataset, cfg=args.cfg)

    if args.export:
        model.export(format=args.export)


if __name__ == "__main__":
    main()
