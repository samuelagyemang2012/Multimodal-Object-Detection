from yolov1_5 import Yolo
from utils.tools import get_class_weight
from tensorflow.keras.optimizers import Adam

INPUT_SHAPE = (448, 448, 3)
CLASS_NAMES = ["car"]

yolo = Yolo(INPUT_SHAPE, CLASS_NAMES)
model = yolo.create_model()
yolo.model.summary()

