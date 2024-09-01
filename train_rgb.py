from ultralytics import YOLO

# Load YOLOv10n model from scratch
model = YOLO("yolov10n.pt")


if __name__ == "__main__":
    # Train the model
    results = model.train(
        data="datasets/visible/visible.yaml",
        epochs=100,
        imgsz=640,
        batch=-1,
    )

    # Save the model
    model.save("yolov10n_visible.pt")
