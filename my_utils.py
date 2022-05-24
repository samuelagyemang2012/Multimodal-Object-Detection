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
    colors = ["#ED457B", "#1E6AC2", "#F28544", "#1DFA51", "#EDDC15"]
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
                    plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color=colors[class_id], fill=False,
                                  linewidth=1))
                current_axis.text(xmin, ymin, label, size='x-large', color='white',
                                  bbox={'facecolor': colors[class_id], 'alpha': 1.0})

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


def resize_data(size, image_folder, annotation_path, columns, dest_image_folder,
                dest_ann_path, limit):  # columns format(image file, xmin,ymin,xmax,ymax)
    new_bboxes = []
    df = pd.read_csv(annotation_path)
    images = df[columns[0]].tolist()
    bbox = df[columns[1:5]].to_numpy()
    classes = df[columns[5]].tolist()

    for i in range(limit):
        img_path = image_folder + images[i]
        img_array = cv2.imread(img_path)
        h, w, c = img_array.shape
        img_array = cv2.resize(img_array, size, interpolation=cv2.INTER_AREA)
        cv2.imwrite(dest_image_folder + images[i], img_array)
        new_bbox = resize_bbox(bbox[i], (h, w), size)
        new_bboxes.append([images[i], new_bbox[0], new_bbox[1], new_bbox[2], new_bbox[3], classes[i]])

    new_df = pd.DataFrame(new_bboxes, columns=['file', 'xmin', 'ymin', 'xmax', 'ymax', 'class'])
    new_df.to_csv(dest_ann_path, index=False)

# image_folder_ = "C:/Users/Administrator/Desktop/datasets/cars/cars_train/"
# annotation_path_ = "C:/Users/Administrator/Desktop/datasets/cars/cars_annotations_train.csv"
# dest_image_folder_ = "C:/Users/Administrator/Desktop/cars_resized/images/"
# dest_ann_path_ = "C:/Users/Administrator/Desktop/cars_resized/"
# columns_ = ["file", "x1", "y1", "x2", "y2", "class"]
# size_ = (448, 448)
#
# resize_data(size_, image_folder_, annotation_path_, columns_, dest_image_folder_, dest_ann_path_ + "all.csv", 2000)

# # Draw boxes
# tdf = pd.read_csv(dest_ann_folder_ + "test.csv")
# images = tdf['file'].tolist()
# bb = tdf[["x1", "y1", "x2", "y2"]].to_numpy()
#
# for i in range(len(images)):
#     img = cv2.imread(dest_image_folder_ + images[i])
#     img = cv2_draw_box(img, bb[i][0], bb[i][1], bb[i][2], bb[i][3], (0, 255, 0), 1)
#     cv2.imshow("", img)
#     cv2.waitKey(0)
