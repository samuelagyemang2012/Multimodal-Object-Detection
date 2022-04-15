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
    label_len = (5 + num_classes)
    label_matrix = np.zeros([grid_size, grid_size, label_len])

    for i, l in enumerate(label_):
        # print(label_)
        xmin = l[0]
        ymin = l[1]
        xmax = l[2]
        ymax = l[3]
        cls = l[4]

        x = (xmin + xmax) / 2 / img_w
        y = (ymin + ymax) / 2 / img_h

        w = ((xmax - xmin) / img_w)
        h = ((ymax - ymin) / img_h)

        loc_x = grid_size * x
        loc_y = grid_size * y

        cell_x = int(loc_x)
        cell_y = int(loc_y)

        x = loc_x - cell_x
        y = loc_y - cell_y

        idx_oh = 2 * 5 + cls

        if label_matrix[cell_y, cell_x, 0] == 0:
            label_matrix[cell_y, cell_x, 0] = 1
            label_matrix[cell_y, cell_x, 1:5] = [x, y, w, h]
            label_matrix[cell_y, cell_x, idx_oh] = 1

    return image_arr, label_matrix


def read_data2(image_path_, label_, grid_size, num_classes):
    image_arr = cv2.imread(image_path_)
    img_h, img_w, img_c = image_arr.shape
    image_arr = np.array(image_arr, dtype='float32')
    image_arr = image_arr / 255.

    label_matrix = np.zeros((grid_size, grid_size, num_classes + 5))

    for z, l in enumerate(label_):
        xmin = l[0]
        ymin = l[1]
        xmax = l[2]
        ymax = l[3]
        cls = l[4]

        x = (xmin + xmax) / 2 / img_w
        y = (ymin + ymax) / 2 / img_h
        w = ((xmax - xmin) / img_w)
        h = ((ymax - ymin) / img_h)

        i, j = int(grid_size * y), int(grid_size * x)
        x_cell, y_cell = grid_size * x - j, grid_size * y - i

        width_cell, height_cell = (
            w * grid_size,
            h * grid_size,
        )

        if label_matrix[i, j, num_classes] == 0:
            label_matrix[i, j, num_classes] = 1

            box_coordinates = np.array([x_cell, y_cell, width_cell, height_cell])
            label_matrix[i, j, num_classes + 1:num_classes + 5] = box_coordinates

            label_matrix[i, j, cls] = 1


    return image_arr, label_matrix


def prepare_data(csv_path, image_folder, grid_size, num_classes, columns):  # columns[xmin,ymin,xmax,ymax,class]
    df = pd.read_csv(csv_path)
    images = df[columns[0]].unique().tolist()
    imgs = []
    labels = []
    names = []

    for n, i in enumerate(images):
        names.append(i)
        image_path = image_folder + i
        # print(image_path)

        # bbox = df[df['image'] == i]
        bbox = df[df[columns[0]] == i]

        # bbox = bbox[["xmin", "ymin", "xmax", "ymax", "class_id"]].to_numpy()
        bbox = bbox[columns[1:6]].to_numpy()
        bbox = bbox.astype(int)

        img_, label_ = read_data2(image_path, bbox, grid_size, num_classes)
        imgs.append(img_)
        labels.append(label_)

    return np.array(imgs), np.array(labels), names

# data_path = "C:/Users/Administrator/Desktop/cars_resized/all.csv"
# image_base = "C:/Users/Administrator/Desktop/cars_resized/images/"
#
# images, labels, names = prepare_data(data_path, image_base, 7, 1, ['file', "x1", "y1", "x2", "y2", "class"])
# print(images.shape)
# print(labels.shape)
