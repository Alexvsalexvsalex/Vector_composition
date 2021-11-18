import os
import numpy as np
from PIL import Image
from pathlib import Path
import traceback

from utils import read_image

CUR_DIR = os.path.dirname(os.path.realpath(__file__))
DATASETS_DIR = os.path.join(CUR_DIR, 'png')
RESULT_DIR = os.path.join(CUR_DIR, 'result2')
IMAGE_SIZE = (1000, 1000)


def cut_image(image, mask):
    return np.dstack((np.where(mask[:, :, :3] < 0.5, 0, image),
                      np.maximum(np.maximum(mask[:, :, 0], mask[:, :, 1]), mask[:, :, 2])))


def del_from_image(image, mask):
    mask = mask[:, :, :3]
    return np.where(mask > 0.5, 0, image)


def read_resized(image_path):
    original_image = Image.fromarray(read_image(image_path)).convert('RGB')
    original_image.thumbnail(IMAGE_SIZE)
    return np.array(original_image)


if __name__ == '__main__':
    datasets = os.listdir(DATASETS_DIR)
    for dataset in datasets:
        print(f"FOUND DATASET {dataset}")

        dataset_result_path = os.path.join(RESULT_DIR, dataset)

        images_path = os.path.join(DATASETS_DIR, dataset, 'input')
        masks_path = os.path.join(DATASETS_DIR, dataset, 'mask')

        back_result_f = os.path.join(dataset_result_path, 'back')
        fore_result_f = os.path.join(dataset_result_path, 'fore')
        Path(back_result_f).mkdir(parents=True, exist_ok=True)
        Path(fore_result_f).mkdir(parents=True, exist_ok=True)

        image_names = os.listdir(images_path)
        mask_names = os.listdir(masks_path)

        for image_name in image_names:
            image_name_no_ext = os.path.splitext(image_name)[0]
            mask_name = next(x for x in mask_names if x.startswith(image_name_no_ext))

            image_path = os.path.join(images_path, image_name)
            mask_path = os.path.join(masks_path, mask_name)
            back_result_path = f"{os.path.join(back_result_f, image_name_no_ext)}.png"
            fore_result_path = f"{os.path.join(fore_result_f, image_name_no_ext)}.png"

            image = read_resized(image_path)
            mask = read_resized(mask_path)
            try:
                Image.fromarray(del_from_image(image, mask)).save(back_result_path, 'png')
                Image.fromarray(cut_image(image, mask)).save(fore_result_path, 'png')

                print(f"DONE {image_path}")
            except Exception as e:
                print(f"FAIL {image_path}")
                traceback.print_exc()
                if os.path.isfile(back_result_path): os.remove(back_result_path)
                if os.path.isfile(fore_result_path): os.remove(fore_result_path)


