import yaml
import cv2
import matplotlib.pyplot as plt
import os
with open("D:/ids project/dataset/data.yaml", 'r') as f:
    data = yaml.safe_load(f)

print(data)
# Example keys: 'train', 'val', 'nc', 'names'


def visualize_sample(image_path, label_path, class_names):
    image = cv2.imread(image_path)
    h, w, _ = image.shape

    with open(label_path, 'r') as f:
        for line in f:
            cls, x_center, y_center, box_w, box_h = map(float, line.strip().split())
            xmin = int((x_center - box_w / 2) * w)
            ymin = int((y_center - box_h / 2) * h)
            xmax = int((x_center + box_w / 2) * w)
            ymax = int((y_center + box_h / 2) * h)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
            cv2.putText(image, class_names[int(cls)], (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Example usage
image_path = 'D:/ids project/dataset/train/images/_nSEvvaT6rM_jpg.rf.db8a53567d1bab69878fa5e5c0d2d9c3.jpg'
label_path = 'D:/ids project/dataset/train/labels/_nSEvvaT6rM_jpg.rf.db8a53567d1bab69878fa5e5c0d2d9c3.txt'
class_names = data['names']
visualize_sample(image_path, label_path, class_names)
