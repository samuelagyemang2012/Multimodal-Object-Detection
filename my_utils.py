import cv2
from matplotlib import pyplot as plt


def yolo_to_pasvoc(img_width, img_height, box):
    xmax = (box[0] * img_width) + (box[2] * img_width) / 2.0
    ymax = (box[1] * img_height) + (box[3] * img_height) / 2.0

    xmin = (box[0] * img_width) - (box[2] * img_width) / 2.0
    ymin = (box[1] * img_height) - (box[3] * img_height) / 2.0
    class_id = int(box[5])
    conf = box[4]

    return int(xmin), int(xmax), int(ymin), int(ymax), int(class_id), round(conf, 2)


def cv2_draw_box(img_array, x1, x2, y1, y2, color):
    img = cv2.rectangle(img_array, (x1, y1), (x2, y2), color, 2)
    return img


# x, y, w, h, conf, class_i, p
def cv2_do_detections(image_array, decoded_detections, img_width, img_height):
    for i, box in enumerate(decoded_detections):
        xmin, xmax, ymin, ymax, class_id, conf = yolo_to_pasvoc(img_width, img_height, box)
        image_array = cv2_draw_box(image_array, xmin, xmax, ymin, ymax, (0, 0, 255))

    cv2.imshow("", image_array)
    cv2.waitKey(0)


def plt_do_detections(image_array, decoded_detections, img_width, img_height, class_names, dest_path_):
    for i, d in enumerate(decoded_detections):

        plt.figure(figsize=(20, 12))
        current_axis = plt.gca()
        plt.axis(False)
        plt.imshow(image_array[i])

        for box in d:
            xmin, xmax, ymin, ymax, class_id, conf = yolo_to_pasvoc(img_width, img_height, box)

            label = '{}: {:.2f}'.format(class_names[class_id], conf)
            current_axis.add_patch(
                plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color="red", fill=False, linewidth=1))
            current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor': "red", 'alpha': 1.0})

        plt.savefig(dest_path_ + str(i) + ".jpg", dpi=100, bbox_inches="tight")
