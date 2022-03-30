from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, concatenate, Input, GlobalAveragePooling2D, Flatten, Reshape, add, \
    multiply
from tensorflow.keras.layers import BatchNormalization as BN
from yolov1_5.models.backbone import darknet_body, radarnet_body


def darknet(input_shape=(224, 224, 3), class_num=10):
    inputs = Input(input_shape)
    darknet_outputs = darknet_body(inputs)

    x = GlobalAveragePooling2D()(darknet_outputs)
    outputs = Dense(class_num, activation="softmax")(x)

    model = Model(inputs, outputs)

    return model


def radarnet(input_shape, class_num):
    inputs = Input(input_shape)
    radarnet_output = radarnet_body(inputs)
    outputs = Dense(class_num, activation="softmax")(radarnet_output)

    model = Model(inputs, outputs)
    return model


def yolo_body(input_shape=(448, 448, 3), pretrained_darknet=None):
    inputs = Input(input_shape)
    darknet = Model(inputs, darknet_body(inputs))

    if pretrained_darknet is not None:
        darknet.set_weights(pretrained_darknet.get_weights())

    return darknet


def my_yolo_body(input_shape1=(448, 448, 3), input_shape2=(128,), pretrained_darknet=None):
    input1 = Input(input_shape1)
    input2 = Input(input_shape2)

    darknet = Model(input1, darknet_body(input1))
    radarnet = Model(input2, radarnet_body(input2))

    return darknet, radarnet


def yolo_head(model_body, bbox_num=2, class_num=10):
    inputs = model_body.input
    output = model_body.output

    xywhc_output = Conv2D(5 * bbox_num, 1,
                          padding='same',
                          kernel_initializer='he_normal',
                          activation='sigmoid')(output)
    p_output = Conv2D(class_num, 1,
                      padding='same',
                      kernel_initializer='he_normal',
                      activation='softmax')(output)

    outputs = concatenate([xywhc_output, p_output], axis=3)

    model = Model(inputs, outputs)

    return model


def my_yolo_head(model_body1, model_body2, bbox_num=2, class_num=10):
    out = (7 * 7 * (bbox_num * 5 + class_num))

    inputs1 = model_body1.input
    inputs2 = model_body2.input

    output1 = model_body1.output
    output2 = model_body2.output

    # -------------------------------------------------------
    xywhc_output1 = Conv2D(5 * bbox_num, 1, padding='same', kernel_initializer='he_normal', activation='sigmoid')(
        output1)
    p_output1 = Conv2D(class_num, 1, padding='same', kernel_initializer='he_normal', activation='softmax')(output1)
    outputs1 = concatenate([xywhc_output1, p_output1], axis=3)
    print(output1.shape)
    # ----------------------------------------------------------

    flatten = Flatten()(output2)
    radar_xywhc = Dense(out, activation='sigmoid')(flatten)
    outputs2 = Reshape((7, 7, (bbox_num * 5 + class_num)))(radar_xywhc)
    final_conc = add([outputs1, outputs2])
    # ---------------------------------------------------------

    model = Model(inputs=[inputs1, inputs2], outputs=final_conc)
    return model

    # out_units = (7 * 7 * (bbox_num * 5 + class_num))

    # flatten = Flatten()(output)
    # bn = BN()(flatten)
    # dense = Dense(out_units, activation='sigmoid')(bn)
    # out = Reshape(target_shape=(7, 7, (bbox_num * 5 + class_num)))(dense)
    #
    # model = Model(inputs, out)
    # print(model.output_shape)

    # return model
