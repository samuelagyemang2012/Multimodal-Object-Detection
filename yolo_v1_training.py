from yolov1_5.models.Yolo import Yolo
from utils.tools import get_class_weight
from utils.measurement import PR_func
from tensorflow.keras.optimizers import Adam

INPUT_SHAPE = (448, 448, 3)
CLASS_NAMES = ["car"]
CLASS_NUM = len(CLASS_NAMES)
BATCH_SIZE = 1
BBOX_NUM = 2
IMAGES_PATH = "C:/Users/Administrator/Desktop/cars_resized/images/"
CSV_PATH = "C:/Users/Administrator/Desktop/cars_resized/mini-500.csv"
# COLUMNS = ['image', "xmin", "ymin", "xmax", "ymax", 'class_id']
EPOCHS = 5

# Load model
yolo = Yolo(INPUT_SHAPE, CLASS_NAMES)

# Load data
imgs, labels = yolo.read_file_to_dataset_csv(IMAGES_PATH, CSV_PATH, is_RGB=True)

# Split data
test_ = int(len(imgs) * 0.2)
val_ = int(len(imgs) * 0.1)

test_img = imgs[0: test_]
test_label = labels[0: test_]

print("shape of testing img:", test_img.shape)
print("shape of testing label:", test_label.shape)
print()

valid_img = imgs[test_:test_ + val_]
valid_label = labels[test_:test_ + val_]
print("shape of validation img:", valid_img.shape)
print("shape of validation label:", valid_label.shape)
print()

train_img = imgs[test_ + val_:]
train_label = labels[test_ + val_:]
print("shape of training img:", train_img.shape)
print("shape of training label:", train_label.shape)

vis_path = "C:/Users/Administrator/Desktop/"
for i in range(len(test_img[0:5])):
    figax = yolo.vis_img(test_img[i],
                         test_label[i],
                         nms_mode=0,
                         text_fontsize=0,
                         box_color="b",
                         point_radius=0,
                         return_fig_ax=True,
                         savefig_path=vis_path + str(i) + ".png")

# Compile model
binary_weight = get_class_weight(
    labels[..., 4:5],
    method='binary'
)

print(binary_weight)

# Create model
model = yolo.create_model()


loss = yolo.loss(binary_weight)
metrics = yolo.metrics("obj+iou+recall0.5")
yolo.model.compile(optimizer=Adam(learning_rate=1e-4),
                   loss=loss,
                   metrics=metrics)

# Fit model
history = yolo.model.fit(imgs,
                         labels,
                         epochs=EPOCHS)

# Save model
yolo.model.save('C:/Users/Administrator/Desktop/yolo_model.h5')

# print(PR_func.get_map(mode="voc2012")
prediction = yolo.model.predict(test_img)
