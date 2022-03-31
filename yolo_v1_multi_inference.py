import os
from yolov1_5.models.MyYolo import MyYolo
from utils.tools import decode
from my_utils import cv2_do_detections, plt_do_detections
import numpy as np
import pandas as pd
import cv2

model_weight_path = "D:/Datasets/infrared_dataset/trained_models/radar_image_yolo.h5"
image_path = "D:/Datasets/infrared_dataset/images/"
radar_path = "D:/Datasets/infrared_dataset/radar/radar_train_car_only.csv"
dest_path = "C:/Users/Sam/Desktop/dd/"
image_shape = (448, 448, 3)
radar_shape = (128,)
classes = ['car']
classes_num = len(classes)
threshold = 0.5
yolo_version = 1
image_arr = []
preds = []

# fetch_images
images = os.listdir(image_path)
images = images[0:20]

radar_df = pd.read_csv(radar_path)
signals = radar_df['signal'].tolist()
radar_arr = [np.array(t.replace("[", "").replace("]", "").replace("\n", "").split()).astype(np.float32) for t in
             signals]
radar_arr = np.array(radar_arr)
radar_arr = radar_arr[0:20]

# load model
yolo = MyYolo(input_shape1=image_shape,
              input_shape2=radar_shape,
              class_names=classes)

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
detections = yolo.model.predict([image_arr[0:20],
                                 radar_arr[0:20]])
decoded_detections = []

# show predictions labels
for i in range(len(image_arr[0:20])):
    p = decode(detections[i], class_num=classes_num, threshold=threshold, version=yolo_version)
    decoded_detections.append(p)
    print(p)
    print("")

# show bbox in images with cv2
# for i in range(len(image_arr)):
#     cv2_do_detections(image_arr[i], decoded_detections[i], image_shape[0], image_shape[1])

# show bbox in images with plt
plt_do_detections(image_arr, decoded_detections, image_shape[0], image_shape[1], classes, dest_path, threshold)
