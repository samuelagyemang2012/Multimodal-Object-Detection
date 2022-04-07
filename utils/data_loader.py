import os

import pandas as pd
import numpy as np
import cv2


def read_data(image_path_, label_, grid_size, num_classes):
    image_arr = cv2.imread(image_path_)
    img_h, img_w, img_c = image_arr.shape
    # image_arr = cv2.cvtColor(image_arr, cv2.COLOR_BGR2RGB)
    image_arr = np.array(image_arr, dtype='float32')
    image_arr = image_arr / 255.
    label_matrix = np.zeros([grid_size, grid_size, num_classes + 5])

    for l in label_:
        # get bbox
        x = (l[0] + l[1]) / 2 / img_w
        y = (l[2] + l[3]) / 2 / img_h
        w = (l[1] - l[0]) / img_w
        h = (l[3] - l[2]) / img_h
        cls = l[4]

        loc = [grid_size * x, grid_size * y]
        loc_i = int(loc[1])
        loc_j = int(loc[0])

        y = loc[1] - loc_i
        x = loc[0] - loc_j

        if label_matrix[loc_i, loc_j, num_classes] == 0:
            label_matrix[loc_i, loc_j, num_classes] = 1
            label_matrix[loc_i, loc_j, cls] = 1
            label_matrix[loc_i, loc_j, num_classes + 1:num_classes + 5] = x, y, h, w

    return image_arr, label_matrix


def prepare_data(csv_path, image_folder, grid_size, num_classes):
    df = pd.read_csv(csv_path)
    images = df['image'].unique().tolist()
    imgs = []
    labels = []
    names = []

    for i in images:
        names.append(i)
        image_path = image_folder + i
        bbox = df[df['image'] == i]
        bbox = bbox[["xmin", "ymin", "xmax", "ymax", "class_id"]].to_numpy()
        img_, label_ = read_data(image_path, bbox, grid_size, num_classes)
        imgs.append(img_)
        labels.append(label_)

    return np.array(imgs), np.array(labels), names

# data_path = "C:/Users/Administrator/Desktop/Self Driving Cars/labels_train.csv"
# image_base = "C:/Users/Administrator/Desktop/Self Driving Cars/images/"
#
#
# images, labels = prepare_data(data_path, image_base, 7, 5)
# print(images.shape)
# print(labels.shape)
