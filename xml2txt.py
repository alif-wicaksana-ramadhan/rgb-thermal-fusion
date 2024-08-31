import os
import glob
import xml.etree.ElementTree as ET

# Paths
xml_dir = "dataset/Annotations"
output_dir = "dataset/labels"
os.makedirs(output_dir, exist_ok=True)

# Class mapping (adjust according to your classes)
class_mapping = {
    "person": 1,
}


def convert_xml_to_yolo(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    image_width = int(root.find("size/width").text)
    image_height = int(root.find("size/height").text)
    file_name = root.find("filename").text
    base_name = os.path.splitext(file_name)[0]

    with open(os.path.join(output_dir, base_name + ".txt"), "w") as f:
        for obj in root.findall("object"):
            class_name = obj.find("name").text
            if class_name not in class_mapping:
                continue

            class_id = class_mapping[class_name]
            bndbox = obj.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)

            # YOLO format
            x_center = (xmin + xmax) / 2 / image_width
            y_center = (ymin + ymax) / 2 / image_height
            width = (xmax - xmin) / image_width
            height = (ymax - ymin) / image_height

            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")


# Convert all XML files
xml_files = glob.glob(os.path.join(xml_dir, "*.xml"))
for xml_file in xml_files:
    convert_xml_to_yolo(xml_file)

print("Conversion completed!")
