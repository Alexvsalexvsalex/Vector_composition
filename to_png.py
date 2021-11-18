import os
import traceback
from pathlib import Path

CUR_DIR = os.path.dirname(os.path.realpath(__file__))
SVG_DIR = os.path.join(CUR_DIR, 'original_svg')
RESULT_DIR = os.path.join(CUR_DIR, 'png')


def run_rasterize(input, output):
    os.system(f"convert {input} {output}")


if __name__ == '__main__':
    datasets = os.listdir(SVG_DIR)
    for dataset in datasets:
        print(f"FOUND DATASET {dataset}")

        dataset_vect_path = os.path.join(SVG_DIR, dataset)
        dataset_result_path = os.path.join(RESULT_DIR, dataset)
        Path(dataset_result_path).mkdir(parents=True, exist_ok=True)

        image_names = os.listdir(dataset_vect_path)

        for image_name in image_names:
            image_name_no_ext = os.path.splitext(image_name)[0]

            image_path = os.path.join(dataset_vect_path, image_name)
            img_result_path = f"{os.path.join(dataset_result_path, image_name_no_ext)}.png"

            try:
                run_rasterize(image_path, img_result_path)

                print(f"DONE {image_path}")
            except Exception as e:
                print(f"FAIL {image_path}")
                traceback.print_exc()

