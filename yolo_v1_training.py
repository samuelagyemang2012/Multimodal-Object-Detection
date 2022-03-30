from yolov1_5.models.Yolo import Yolo
from utils.tools import get_class_weight
from tensorflow.keras.optimizers import Adam

INPUT_SHAPE = (448, 448, 3)
CLASS_NAMES = ["car"]
BATCH_SIZE = 4
BBOX_NUM = 2
IMAGES_PATH = "D:/Datasets/infrared_dataset/images/"
LABELS_PATH = "D:/Datasets/infrared_dataset/annotations/"
EPOCHS = 50

# Load model
yolo = Yolo(INPUT_SHAPE, CLASS_NAMES)

# Load data
img, label = yolo.read_file_to_dataset(
    IMAGES_PATH,
    LABELS_PATH)

# seq = yolo.read_file_to_sequence(
#     IMAGES_PATH,
#     LABELS_PATH,
#     BATCH_SIZE)


# Split data
test_ = int(len(img) * 0.2)  # 0:150
val_ = int(len(img) * 0.1)  # 150: 150 +75

test_img = img[0: test_]
test_label = label[0: test_]
print("shape of testing img:", test_img.shape)
print("shape of testing label:", test_label.shape)
print()

valid_img = img[test_:test_ + val_]
valid_label = label[test_:test_ + val_]
print("shape of validation img:", valid_img.shape)
print("shape of validation label:", valid_label.shape)
print()

train_img = img[test_ + val_:]
train_label = label[test_ + val_:]
print("shape of training img:", train_img.shape)
print("shape of training label:", train_label.shape)

# Create model
model = yolo.create_model()

# Compile model
binary_weight = get_class_weight(
    label[..., 4:5],
    method='binary'
)

loss = yolo.loss(binary_weight)
metrics = yolo.metrics("obj+iou+recall0.5")
yolo.model.compile(optimizer=Adam(learning_rate=1e-4),
                   loss=loss,
                   metrics=metrics)

# Fit model
history = yolo.model.fit(img,label,EPOCHS)

prediction = yolo.model.predict(test_img)
