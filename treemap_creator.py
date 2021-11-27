import os
import random
from pathlib import Path
import traceback
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from PIL import Image
import torchvision.transforms as T

model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def get_prediction(img_path, threshold):
    img = Image.open(img_path).convert('RGB')
    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    pred = model([img])
    pred_score = list(pred[0]['scores'].detach().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
    masks = (pred[0]['masks'] > 0.5).squeeze().detach().cpu().numpy()
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]
    masks = masks[:pred_t + 1]
    pred_boxes = pred_boxes[:pred_t + 1]
    pred_class = pred_class[:pred_t + 1]
    return masks, pred_boxes, pred_class


def random_colour_masks(image):
    colours = [[0, 255, 0], [0, 0, 255], [255, 0, 0], [0, 255, 255], [255, 255, 0], [255, 0, 255], [80, 70, 180],
               [250, 80, 190], [245, 145, 50], [70, 150, 250], [50, 190, 190]]
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    r[image == 1], g[image == 1], b[image == 1] = colours[random.randrange(0, 10)]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask


def instance_segmentation_api(img_path, threshold=0.2, rect_th=3, text_size=3, text_th=3) -> Image:
    masks, boxes, pred_cls = get_prediction(img_path, threshold)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = np.zeros(img.shape, dtype=np.uint8)
    for i in range(len(masks)):
        rgb_mask = random_colour_masks(masks[i])
        res = cv2.addWeighted(res, 1, rgb_mask, 0.5, 0)
        # cv2.rectangle(img, (int(boxes[i][0][0]), int(boxes[i][0][1])),
        #               (int(boxes[i][1][0]), int(boxes[i][1][1])), color=(0, 255, 0), thickness=rect_th)
        # cv2.putText(img, pred_cls[i], (int(boxes[i][0][0]), int(boxes[i][0][1])), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0), thickness=text_th)
    # plt.figure(figsize=(20, 30))
    return res


CUR_DIR = os.path.dirname(os.path.realpath(__file__))
SVG_DIR = os.path.join(CUR_DIR, 'svg')
RESULT_DIR = os.path.join(CUR_DIR, 'png_')
MASKS_DIR = os.path.join(CUR_DIR, 'masks')


def run_rasterize(input, output):
    os.system(f"convert {input} {output}")


if __name__ == '__main__':
    datasets = os.listdir(SVG_DIR)
    for dataset in datasets:
        print(f"FOUND DATASET {dataset}")

        dataset_vect_path = os.path.join(SVG_DIR, dataset)

        dataset_result_path = os.path.join(RESULT_DIR, dataset)
        Path(dataset_result_path).mkdir(parents=True, exist_ok=True)

        dataset_mask_path = os.path.join(MASKS_DIR, dataset)
        Path(dataset_mask_path).mkdir(parents=True, exist_ok=True)

        image_names = os.listdir(dataset_vect_path)

        for image_name in image_names:
            image_name_no_ext = os.path.splitext(image_name)[0]

            image_path = os.path.join(dataset_vect_path, image_name)
            img_result_path = f"{os.path.join(dataset_result_path, image_name_no_ext)}.png"
            img_mask_path = f"{os.path.join(dataset_mask_path, image_name_no_ext)}.png"

            try:
                run_rasterize(image_path, img_result_path)
                image = instance_segmentation_api(img_result_path)

                Image.fromarray(image, 'RGB').save(img_mask_path)

                print(f"DONE {image_path}")
            except Exception as e:
                print(f"FAIL {image_path}")
                traceback.print_exc()
