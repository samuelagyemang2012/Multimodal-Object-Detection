import os
from yolov1_5 import Yolo
from utils.tools import decode
from my_utils import cv2_do_detections, plt_do_detections
import numpy as np
import cv2

model_weight_path = "D:/Datasets/infrared_dataset/trained_models/inf.h5"
image_path = "D:/Datasets/infrared_dataset/images/"
dest_path = "C:/Users/Sam/Desktop/dd/"
image_shape = (448, 448, 3)
classes = ['car']
image_arr = []
preds = []

# fetch_images
images = os.listdir(image_path)
images = images[300:400]
# choice_images = ["1121R-290.jpg", "1121R-287.jpg", "1121R-269.jpg", "1121R-247.jpg", "1121R-246.jpg", "1121R-245.jpg",
#                  "1121R-244.jpg", "1121R-235.jpg", "1121R-2111.jpg"]
# images = images + choice_images
# load model
yolo = Yolo(input_shape=(448, 448, 3), class_names=['car'])
yolo.create_model(pretrained_weights=model_weight_path)

# fetch images from path
for g in images:
    img = cv2.imread(image_path + g)
    img = img * 1 / 255
    image_arr.append(img)

# convert image to numpy array
image_arr = np.array(image_arr)
print(image_arr.shape)

# do predictions
detections = yolo.model.predict(image_arr)
decoded_detections = []

# show predictions labels
for i in range(len(image_arr)):
    p = decode(detections[i], class_num=1, threshold=0.5, version=1)
    decoded_detections.append(p)
    print(p)
    print("")

# show bbox in images with cv2
# for i in range(len(image_arr)):
#     cv2_do_detections(image_arr[i], decoded_detections[i], image_shape[0], image_shape[1])

# show bbox in images with plt
plt_do_detections(image_arr, decoded_detections, image_shape[0], image_shape[1], classes, dest_path)
