import os
from yolov1_5.models.Yolo import Yolo
from utils.tools import decode
from my_utils import cv2_do_detections, plt_do_detections
import numpy as np
import cv2

model_weight_path = "C:/Users/Administrator/Desktop/car_detector.h5"
image_path = "C:/Users/Administrator/Desktop/my_test/"
dest_path = "C:/Users/Administrator/Desktop/detections/"
image_shape = (448, 448, 3)
classes = ['car']
classes_num = len(classes)
threshold = 0.5
yolo_version = 1
image_arr = []
preds = []
is_rgb = True

# fetch_images
images = os.listdir(image_path)
print(images)

# load model
yolo = Yolo(input_shape=(448, 448, 3), class_names=classes)
yolo.create_model(pretrained_weights=model_weight_path)

# fetch images from path
for g in images:
    img = cv2.imread(image_path + g)
    h, w, c = img.shape

    if h != image_shape[0] or w != image_shape[1]:
        img = cv2.resize(img, (image_shape[0], image_shape[0]), cv2.INTER_AREA)

    if is_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img * 1 / 255
    image_arr.append(img)

# for i in image_arr:
#     cv2.imshow("", i)
#     cv2.waitKey(0)

# convert image to numpy array
image_arr = np.array(image_arr)
print(image_arr.shape)

# do predictions
detections = yolo.model.predict(image_arr)
decoded_detections = []

# show predictions labels
for i in range(len(image_arr)):
    p = decode(detections[i], class_num=classes_num, threshold=threshold, version=yolo_version)
    decoded_detections.append(p)
    print(p)
    print("")

# show bbox in images with plt
plt_do_detections(image_arr, decoded_detections, image_shape[0], image_shape[1], classes, dest_path, threshold)
