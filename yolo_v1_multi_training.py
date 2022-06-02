import os
from yolov1_5.models.MyYolo import MyYolo
from utils.tools import get_class_weight
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

INPUT_SHAPE = (448, 448, 3)
CLASS_NAMES = ["car", "cyclist", "pedestrian"]
CLASS_NUM = len(CLASS_NAMES)
BATCH_SIZE = 1
BBOX_NUM = 2
IMAGES_PATH = "C:/Users/Administrator/Desktop/datasets/CRUW/ALL/resized/images/"
CSV_PATH = "C:/Users/Administrator/Desktop/datasets/CRUW/ALL/resized/resized_data.csv"
RADAR_PATH = ""
EPOCHS = 5

# Load model
yolo = MyYolo(INPUT_SHAPE, CLASS_NAMES)

# Load images and annotations data
imgs, labels = yolo.read_file_to_dataset_csv(IMAGES_PATH, CSV_PATH, is_RGB=True)

# Split data
test_num = int(len(imgs) * 0.2)  # 0:150
val_num = int(len(imgs) * 0.1)  # 150: 150 +75

test_imgs = imgs[0: test_num]
test_labels = labels[0: test_num]
print("shape of testing img:", test_imgs.shape)
print("shape of testing label:", test_labels.shape)

valid_imgs = imgs[test_num:test_num + val_num]
valid_labels = labels[test_num:test_num + val_num]
print("shape of validation img:", valid_imgs.shape)
print("shape of validation label:", valid_labels.shape)

train_imgs = imgs[test_num + val_num:]
train_labels = labels[test_num + val_num:]
print("shape of training img:", train_imgs.shape)
print("shape of training label:", train_labels.shape)

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
print(yolo.model.summary())
plot_model(yolo.model)

# Fit model
# history = yolo.model.fit([train_imgs, train_radar_data],
#                          train_labels,
#                          epochs=EPOCHS,
#                          batch_size=4,
#                          verbose=1,
#                          validation_data=([valid_imgs, valid_radar_data],
#                                           valid_labels))
#
# prediction = yolo.model.save('C:/Users/Administrator/Desktop/radar_image_yolo.h5')
