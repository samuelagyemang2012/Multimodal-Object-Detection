from yolov1_5.models.MyYolo import MyYolo
from utils.tools import get_class_weight
from tensorflow.keras.optimizers import Adam

INPUT_SHAPE_1 = (448, 448, 3)
INPUT_SHAPE_2 = (128,)
CLASS_NAMES = ["car"]

myyolo = MyYolo(INPUT_SHAPE_1, INPUT_SHAPE_2, CLASS_NAMES)
model = myyolo.create_model()
myyolo.model.summary()
