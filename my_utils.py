import cv2
import glob
import pandas as pd
from matplotlib import pyplot as plt
import xml.etree.ElementTree as ET


def yolo_to_pasvoc(img_width, img_height, box):
    xmax = (box[0] * img_width) + (box[2] * img_width) / 2.0
    ymax = (box[1] * img_height) + (box[3] * img_height) / 2.0

    xmin = (box[0] * img_width) - (box[2] * img_width) / 2.0
    ymin = (box[1] * img_height) - (box[3] * img_height) / 2.0
    class_id = int(box[5])
    conf = box[4]

    return int(xmin), int(xmax), int(ymin), int(ymax), int(class_id), round(conf, 2)


def cv2_draw_box(img_array, xmin, ymin, xmax, ymax, color, line_width):
    img = cv2.rectangle(img_array, (xmin, ymin), (xmax, ymax), color, line_width)
    return img


# x, y, w, h, conf, class_i, p
def cv2_do_detections(image_array, decoded_detections, img_width, img_height):
    for i, box in enumerate(decoded_detections):
        xmin, xmax, ymin, ymax, class_id, conf = yolo_to_pasvoc(img_width, img_height, box)
        image_array = cv2_draw_box(image_array, xmin, xmax, ymin, ymax, (0, 0, 255), 1)

    cv2.imshow("", image_array)
    cv2.waitKey(0)


def plt_do_detections(image_array, decoded_detections, img_width, img_height, class_names, dest_path_, threshold):
    for i, d in enumerate(decoded_detections):

        plt.figure(figsize=(20, 12))
        current_axis = plt.gca()
        plt.axis(False)
        plt.imshow(image_array[i])

        for box in d:
            xmin, xmax, ymin, ymax, class_id, conf = yolo_to_pasvoc(img_width, img_height, box)

            if conf > threshold:
                label = '{}: {:.2f}'.format(class_names[class_id], conf)
                current_axis.add_patch(
                    plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color="red", fill=False, linewidth=1))
                current_axis.text(xmin, ymin, label, size='x-large', color='white',
                                  bbox={'facecolor': "red", 'alpha': 1.0})

            plt.savefig(dest_path_ + str(i) + ".jpg", dpi=100, bbox_inches="tight")


def xml_to_csv(path, dest_path, classes):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            bbx = member.find('bndbox')
            xmin = int(bbx.find('xmin').text)
            ymin = int(bbx.find('ymin').text)
            xmax = int(bbx.find('xmax').text)
            ymax = int(bbx.find('ymax').text)
            label = member.find('name').text
            class_id = classes.index(label) + 1

            value = (root.find('filename').text,
                     # int(root.find('size')[0].text),
                     # int(root.find('size')[1].text),
                     xmin,
                     ymin,
                     xmax,
                     ymax,
                     class_id
                     )
            xml_list.append(value)
    column_name = ['image', 'xmin', 'ymin', 'xmax', 'ymax', 'class_id']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    xml_df.to_csv(dest_path, index=False)


def resize_bbox(bbox, in_size, out_size):
    """Resize bounding boxes according to image resize.
    Args:
        bbox (~numpy.ndarray): See the table below.
        in_size (tuple): A tuple of length 2. The height and the width
            of the image before resized.
        out_size (tuple): A tuple of length 2. The height and the width
            of the image after resized.
    .. csv-table::
        xmin,ymin,xmax,ymax
    Returns:
        ~numpy.ndarray:
        Bounding boxes rescaled according to the given image shapes.
    """

    # 0-ymin, 1-xmin, 2-ymax, 3-xmax
    # 1-ymin, 0-xmin, 3-ymax, 2-xmax - ours

    bbox = bbox.copy()
    y_scale = float(out_size[0]) / in_size[0]
    x_scale = float(out_size[1]) / in_size[1]

    bbox[1] = y_scale * bbox[1]
    bbox[3] = y_scale * bbox[3]

    bbox[0] = x_scale * bbox[0]
    bbox[2] = x_scale * bbox[2]
    return bbox

# xml_to_csv("C:/Users/Administrator/Desktop/resized/annotations/",
#            "C:/Users/Administrator/Desktop/resized/annotations.csv", ["car"])
