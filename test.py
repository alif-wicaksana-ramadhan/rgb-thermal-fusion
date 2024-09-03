from ultralytics import YOLO

import cv2

# model_rgb = YOLO("results/train_rgb/weights/best.pt")
# model_infra = YOLO("results/train_infra/weights/best.pt")
model_infra = YOLO("yolov10x.pt")
model_rgb = YOLO("yolov10x.pt")

image_id = "250074"

rgb_img_path = f"datasets/visible/images/train/{image_id}.jpg"
infra_img_path = f"datasets/infrared/images/train/{image_id}.jpg"

rgb_img = cv2.imread(rgb_img_path)
infra_img = cv2.imread(infra_img_path)

rgb_results = model_rgb(rgb_img)
infra_results = model_infra(infra_img)


for result in rgb_results:
    for box in result.boxes:
        print(box.xyxy)
        x1, y1, x2, y2 = box.xyxy[0]
        cv2.rectangle(rgb_img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        # print(f"Bounding Box: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

for result in infra_results:
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        cv2.rectangle(infra_img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        # print(f"Bounding Box: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

cv2.imshow("RGB", rgb_img)
cv2.imshow("Infra", infra_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
