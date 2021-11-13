import os
import traceback
from pathlib import Path

CUR_DIR = os.path.dirname(os.path.realpath(__file__))
RESULT_DIR = os.path.join(CUR_DIR, 'result')
VECTORIZED_DIR = os.path.join(CUR_DIR, 'vectorized')


def run_vectorize(input, output):
    os.system(f"/home/alexxx/.cargo/bin/vtracer --input {input} -g 2 -f 1 -p 8 -s 10 --output {output}")


if __name__ == '__main__':
    datasets = os.listdir(RESULT_DIR)
    for dataset in datasets:
        print(f"FOUND DATASET {dataset}")

        dataset_result_path = os.path.join(RESULT_DIR, dataset)
        dataset_vect_path = os.path.join(VECTORIZED_DIR, dataset)

        back_result_f = os.path.join(dataset_result_path, 'back')
        fore_result_f = os.path.join(dataset_result_path, 'fore')

        back_vect_f = os.path.join(dataset_vect_path, 'back')
        fore_vect_f = os.path.join(dataset_vect_path, 'fore')
        Path(back_vect_f).mkdir(parents=True, exist_ok=True)
        Path(fore_vect_f).mkdir(parents=True, exist_ok=True)

        image_names = os.listdir(back_result_f)

        for image_name in image_names:
            image_name_no_ext = os.path.splitext(image_name)[0]

            image_path = os.path.join(back_result_f, image_name)
            img_result_path = f"{os.path.join(fore_result_f, image_name_no_ext)}.png"

            back_vect_path = f"{os.path.join(back_vect_f, image_name_no_ext)}.svg"
            fore_vect_path = f"{os.path.join(fore_vect_f, image_name_no_ext)}.svg"

            try:
                run_vectorize(image_path, back_vect_path)
                run_vectorize(img_result_path, fore_vect_path)

                print(f"DONE {image_path}")
            except Exception as e:
                print(f"FAIL {image_path}")
                traceback.print_exc()

