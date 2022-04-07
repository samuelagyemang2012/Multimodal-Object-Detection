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

import pandas as pd

data_path = "C:/Users/Administrator/Desktop/Self Driving Cars/labels_train.csv"
df = pd.read_csv(data_path)

ii = "1478019953689774621.jpg"
bb = df[df['image'] == ii]
print(bb[["xmin", "xmax", "ymin", "ymax", "class_id"]])

# bboxes = df[["xmin", "xmax", "ymin", "ymax", "class_id"]].to_numpy()
# print(bboxes[0]
