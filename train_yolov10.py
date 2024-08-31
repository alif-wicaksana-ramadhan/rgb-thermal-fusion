from ultralytics import YOLO

# Load YOLOv10n model from scratch
model = YOLO("yolov10x.yaml").load("yolov10x.pt")


if __name__ == "__main__":
    # Train the model
    results = model.train(data="datasets/visible.yaml", epochs=100)

    # Save the model
    model.save("yolov10x_visible.pt")

    # Print the results
    print(results)
