from pathlib import Path
from multiprocessing import freeze_support
from ultralytics import RTDETR

ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT / "datasets" / "DOTA_aircraft_AABB" / "dataset.yaml"


def main() -> None:
    model = RTDETR("rtdetr-l.pt")
    model.train(
        data=str(DATA_YAML),
        epochs=15,
        imgsz=640,
        batch=2,
        workers=0,
        device=0,
        project="runs/rtdetr",
        name="RTDETR_L_Aircraft",
        pretrained=True,
    )


if __name__ == "__main__":
    freeze_support()
    main()
