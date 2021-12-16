import os.path
import random
import xml.etree.ElementTree as ET
from tqdm import tqdm

import cv2
import numpy as np

random.seed(108)

__win_name = "image_view"
path = "PASCALVOC2007\VOCtest_06-Nov-2007\VOCdevkit\VOC2007\Yolo_Annotation"

# Dictionary that maps class names to IDs
__class_name_to_id_mapping = {
    # Person
    "person": 0,
    # Animal
    "bird": 1, "cat": 2, "cow": 3, "dog": 4, "horse": 5, "sheep": 6,
    # Vehicle
    "aeroplane": 7, "bicycle": 8, "boat": 9, "bus": 10, "car": 11, "motorbike": 12, "train": 13,
    # Indoor
    "bottle": 14, "chair": 15, "diningtable": 16, "pottedplant": 17, "sofa": 18, "tvmonitor": 19,
}


def extract_info(xml_file):
    root = ET.parse(xml_file).getroot()
    info_dict = {'bboxes': []}

    for elem in root:
        if elem.tag == "filename":
            info_dict['filename'] = elem.text
        elif elem.tag == "size":
            img_size = [int(subelem.text) for subelem in elem]
            info_dict['image_size'] = tuple(img_size)
        elif elem.tag == "object":
            bbox = {}
            for subelem in elem:
                if subelem.tag == 'name':
                    bbox["class"] = subelem.text
                elif subelem.tag == 'bndbox':
                    for subsubelem in subelem:
                        bbox[subsubelem.tag] = int(subsubelem.text)
            info_dict['bboxes'].append(bbox)

    return info_dict


# Convert to Yolov5 format and save it to the disk
def convert_to_yolov5(info_dict):

    print_buffer = []

    for bbox in info_dict['bboxes']:
        try:
            class_id = __class_name_to_id_mapping[bbox['class']]
        except KeyError:
            print("Class must be one from", __class_name_to_id_mapping.keys())

        # transform bbox into yolov5 format
        bbox_x_center = (bbox['xmin'] + bbox['xmax']) / 2
        bbox_y_center = (bbox['ymin'] + bbox['ymax']) / 2
        bbox_width = bbox['xmax'] - bbox['xmin']
        bbox_height = bbox['ymax'] - bbox['ymin']

        # Normalize the coordinates along with width and height of the image
        image_w, image_h, _ = info_dict['image_size']
        bbox_x_center /= image_w
        bbox_y_center /= image_h
        bbox_width /= image_w
        bbox_height /= image_h

        print_buffer.append("{} {:.3f} {:.3f} {:.3f} {:.3f}"
                            .format(class_id, bbox_x_center, bbox_y_center, bbox_width, bbox_height))
    save_file_name = os.path.join("", info_dict['filename'].replace('jpg', 'txt'))
    return print_buffer, save_file_name
    # print("save successfully")


def draw_bbox(image, annotation_text_file):
    with open(annotation_text_file, 'r') as file:
        annotation_copy = []
        annotation = file.read().split('\n')[:-1]
        annotation = [x.split(" ") for x in annotation]
        for x in annotation:
            for y in range(0, 5):
                x[y] = float(x[y])
            # print(x)
            annotation_copy.append(x)
        # print(annotation_copy)
        # annotation = [[float(x) for y in x] for x in annotation]

    annotations = np.array(annotation_copy)  # has a form [[class_id, bbox_x_center, bbox_y_center, bbox_width, bbox_height]]
    print(f"Annotation values before denormalization: {annotations}")

    h, w, _ = image.shape

    # denormalize annotations
    annotations_cp = np.copy(annotations)
    annotations_cp[:, [1, 3]] = annotations[:, [1, 3]] * w  # bbox_x_center, bbox_width
    annotations_cp[:, [2, 4]] = annotations[:, [2, 4]] * h  # bbox_y_center, bbox_height
    print(f"Annotation values after denormalization: {annotations_cp}")

    # Convert to (xmin, xmax, ymin, ymax) to draw cv2.rectangle
    annotations_cp[:, 1] = annotations_cp[:, 1] - annotations_cp[:, 3]/2
    annotations_cp[:, 2] = annotations_cp[:, 2] - annotations_cp[:, 4]/2
    annotations_cp[:, 3] = annotations_cp[:, 1] + annotations_cp[:, 3]
    annotations_cp[:, 4] = annotations_cp[:, 2] + annotations_cp[:, 4]

    print("annotations_cp:", + annotations_cp)

    for single_annotation in annotations_cp:
        obj_cls, x0, y0, x1, y1 = single_annotation
        cv2.rectangle(image, (int(x0), int(y0), int(x1), int(y1)), color=(255, 0, 0), thickness=2)
        print(x0, y0, x1, y1)
        cv2.imshow(__win_name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def data_generator(root_path: str):
    root_path = root_path
    annotation_dir = os.path.join(root_path, f"{'pascal_labels'}")
    print(f"annotations dir: {annotation_dir}")
    for xml_file_name in tqdm(os.listdir(annotation_dir)):
        xml_file_path = os.path.join(annotation_dir, xml_file_name)
        if not os.path.exists(f"./labels"):
            os.makedirs(f"./labels")
        yolo_annotation_dir = os.path.join(root_path, f"{'labels'}")
        extracted_info = extract_info(xml_file_path)
        print_buffer, save_file_name = convert_to_yolov5(extracted_info)
        yolo_annotation_file_path = os.path.join(yolo_annotation_dir, save_file_name)
        print('\n'.join(print_buffer), file=open(yolo_annotation_file_path, 'w'))

    print("xml to yolov5 txt conversion successful")




#image_file = "test.jpg"
#annotation_file = "test.txt"
#img = cv2.imread(image_file)
#draw_bbox(img, annotation_file)

#Test Dataset Path
#root_path = "PASCALVOC2007\VOCtest_06-Nov-2007\VOCdevkit\VOC2007"

#Train Dataset Path
root_path = "Dataset/test"

'''
xml_file_path = '000001.xml'
info_dict = extract_info(xml_file_path)
print_buffer, save_file_name = convert_to_yolov5(info_dict)
yolo_annotation_file_path = os.path.join(yolo_annotation_dir, save_file_name)
print(yolo_annotation_file_path)
print('\n'.join(print_buffer), file=open(yolo_annotation_file_path, 'w'))
'''


data_generator(root_path)

