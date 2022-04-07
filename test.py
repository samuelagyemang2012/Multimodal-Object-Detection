# from yolov1_5.models.MyYolo import MyYolo
# from utils.tools import get_class_weight
# from tensorflow.keras.optimizers import Adam

# INPUT_SHAPE_1 = (448, 448, 3)
# INPUT_SHAPE_2 = (128,)
# CLASS_NAMES = ["car"]
#
# myyolo = MyYolo(INPUT_SHAPE_1, INPUT_SHAPE_2, CLASS_NAMES)
# model = myyolo.create_model()
# myyolo.model.summary()

# ------------------------------------------------------------
import numpy as np
import pandas as pd
import cv2
from my_utils import cv2_draw_box

image_base = "C:/Users/Administrator/Desktop/resized/images/"
# image_base = "C:/Users/Administrator/Desktop/data_all/car_voc/VOCdevkit/VOC2012/JPEGImages/"
data_path = "C:/Users/Administrator/Desktop/resized/annotations_3.csv"
image_name = "1501R-2085.jpg"

df = pd.read_csv(data_path)
bb = df[df['image'] == image_name]
bb = np.array(bb)

img = cv2.imread(image_base + image_name)

for b in bb:
    print(b)
    img = cv2_draw_box(img, b[1], b[2], b[3], b[4], (0, 0, 255), 1)

cv2.imshow("", img)
cv2.waitKey(0)

# ------------------------------------------------------------
# import cv2

# import numpy as np
#
# from my_utils import cv2_draw_box
#
# image_base = "C:/Users/Administrator/Desktop/data_all/car_voc/VOCdevkit/VOC2012/JPEGImages/"
# image_path = "1121R-206.jpg"
#
# bbox = [[334, 279, 374, 313],  # car
#         [363, 310, 401, 347],  # car
#         [362, 341, 371, 361],  # person
#         [374, 340, 383, 360],  # person
#         [378, 334, 388, 355]]  # person
#
# img = cv2.imread(image_base + image_path)
# # xmin,ymin,xmax,ymax
# # 0      1    2   3
# for b in bbox:
#     img = cv2_draw_box(img, b[0], b[1], b[2], b[3], (0, 0, 255), 1)  # xmin,ymin,xmax,ymax
#
# cv2.imshow("", img)
# cv2.waitKey(0)
#
# img1 = cv2.imread(image_base + image_path)
# h, w, c = img1.shape
# img1 = cv2.resize(img1, (448, 448))
#
# bbox1 = [[233, 244, 261, 273],
#          [254, 271, 280, 303],
#          [253, 298, 259, 315],
#          [261, 297, 268, 315],
#          [264, 292, 271, 310]]  # person
#
# # xmin,ymin,xmax,ymax
# # 0      1    2   3
#
# for b1 in bbox1:
#     # rb = np.array(b1)
#     # rb = resize_bbox(rb, (h, w), (448, 448))
#     # print(rb)
#     img1 = cv2_draw_box(img1, b1[0], b1[1], b1[2], b1[3], (0, 0, 255), 1)  # xmin,ymin,xmax,ymax
# #
# cv2.imshow("", img1)
# cv2.waitKey(0)

# label.txt = xmin,ymin,xmax,ymax [
# -----------------------
# Resize annotations
# import numpy as np
# import pandas as pd
# from my_utils import resize_bbox
#
# df_path = "C:/Users/Administrator/Desktop/resized/annotations_2.csv"
# df = pd.read_csv(df_path)
#
# images = df['image'].tolist()
# class_ids = df['class_id'].tolist()
# bboxes = df[["xmin", "ymin", "xmax", "ymax"]]
# bboxes = np.array(bboxes)
# data = []
#
# for i, b in enumerate(bboxes):
#     rb = np.array(b)
#     rb = resize_bbox(rb, (512, 640), (448, 448))
#     data.append([images[i], rb[0], rb[1], rb[2], rb[3], class_ids[i]])
#
# new_df = pd.DataFrame(data, columns=["image", "xmin", "ymin", "xmax", "ymax", "class_id"], index=None)
# new_df.to_csv("C:/Users/Administrator/Desktop/resized/annotations_3.csv", index=False)
