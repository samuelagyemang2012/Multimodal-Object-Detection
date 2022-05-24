import os

from yolov1_5.models.MyYolo import MyYolo
from tensorflow.keras.utils import to_categorical
from utils.tools import get_class_weight
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np
import gc

gc.collect()

INPUT_SHAPE_1 = (448, 448, 3)
INPUT_SHAPE_2 = (448, 448, 3)
CLASS_NAMES = ["car", "cyclist", "pedestrian"]
BATCH_SIZE = 1
BBOX_NUM = 2
IMAGES_PATH = "C:/Users/Administrator/Desktop/resized/images/"
LABELS_PATH = "C:/Users/Administrator/Desktop/resized/annotations/"
RADAR_PATH = "C:/Users/Administrator/PycharmProjects/radar_classification/data/radar_train_car_only.csv"
EPOCHS = 300

# Load model
yolo = MyYolo(INPUT_SHAPE_1, INPUT_SHAPE_2, CLASS_NAMES)

# Load images and annotations data
imgs, labels = yolo.read_file_to_dataset(
    IMAGES_PATH,
    LABELS_PATH)

# Load radar data and labels
radar_df = pd.read_csv(RADAR_PATH)

radar_data = radar_df['signal'].tolist()
print(len(radar_data))
radar_data = radar_data[0:len(labels)]
radar_data = [np.array(t.replace("[", "").replace("]", "").replace("\n", "").split()).astype(np.float32) for t in
              radar_data]
radar_data = np.array(radar_data)

radar_labels = radar_df['label'].tolist()
radar_labels = radar_labels[0:len(labels)]
radar_labels_enc = []

for rl in radar_labels:
    if rl == 'car':
        radar_labels_enc.append(0)

radar_labels_enc = to_categorical(radar_labels_enc)

# seq = yolo.read_file_to_sequence(
#     IMAGES_PATH,
#     LABELS_PATH,
#     BATCH_SIZE)


# Split data
test_num = int(len(imgs) * 0.2)  # 0:150
val_num = int(len(imgs) * 0.1)  # 150: 150 +75
test_imgs = imgs[0: test_num]
test_labels = labels[0: test_num]
test_radar_data = radar_data[0:test_num]
test_radar_labels = radar_labels_enc[0:test_num]
print("shape of testing img:", test_imgs.shape)
print("shape of testing label:", test_labels.shape)
print("shape of testing radar data:", test_radar_data.shape)
print("shape of testing radar labels:", test_radar_labels.shape)
print()

valid_imgs = imgs[test_num:test_num + val_num]
valid_labels = labels[test_num:test_num + val_num]
valid_radar_data = radar_data[test_num:test_num + val_num]
valid_radar_labels = radar_labels_enc[test_num:test_num + val_num]
print("shape of validation img:", valid_imgs.shape)
print("shape of validation label:", valid_labels.shape)
print("shape of validation radar data:", valid_radar_data.shape)
print("shape of validation radar labels:", valid_radar_labels.shape)
print()

train_imgs = imgs[test_num + val_num:]
train_labels = labels[test_num + val_num:]
train_radar_data = radar_data[test_num + val_num:]
train_radar_labels = radar_labels_enc[test_num + val_num:]
print("shape of training img:", train_imgs.shape)
print("shape of training label:", train_labels.shape)
print("shape of training radar data:", train_radar_data.shape)
print("shape of training radar labels:", train_radar_labels.shape)
print()

# Create model
model = yolo.create_model()

# Compile model
binary_weight = get_class_weight(
    labels[..., 4:5],
    method='binary'
)

loss = yolo.loss(binary_weight)
metrics = yolo.metrics("obj+iou+recall0.5")
yolo.model.compile(optimizer=Adam(learning_rate=1e-4),
                   loss=loss,
                   metrics=metrics)

# Fit model
history = yolo.model.fit([train_imgs, train_radar_data],
                         [train_labels, train_radar_labels],
                         epochs=EPOCHS,
                         batch_size=4,
                         verbose=1,
                         validation_data=([valid_imgs, valid_radar_data],
                                          [valid_labels, valid_radar_labels]))

prediction = yolo.model.save('C:/Users/Administrator/Desktop/radar_image_yolo.h5')
