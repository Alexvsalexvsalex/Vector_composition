import os
import traceback
from io import BytesIO
from pathlib import Path
import json
import requests
import base64

from PIL import Image

URL = "http://109.188.135.85:9333/harmonize"
CUR_DIR = os.path.dirname(os.path.realpath(__file__))
DATASETS_DIR = os.path.join(CUR_DIR, 'result')
RESULT_DIR = os.path.join(CUR_DIR, 'harmonized')


def make_query(bg, fg):
    return {"bg": bg.decode('utf-8'), "fg": fg.decode('utf-8'), "x": 0, "y": 0, "scale": 1}


if __name__ == '__main__':
    datasets = os.listdir(DATASETS_DIR)
    for dataset in datasets:
        print(f"FOUND DATASET {dataset}")

        dataset_result_path = os.path.join(DATASETS_DIR, dataset)
        result_path = os.path.join(RESULT_DIR, dataset)

        back_result_f = os.path.join(dataset_result_path, 'back')
        fore_result_f = os.path.join(dataset_result_path, 'fore')

        harm_result_path = os.path.join(result_path, 'harm')
        Path(harm_result_path).mkdir(parents=True, exist_ok=True)

        image_names = os.listdir(back_result_f)

        for image_name in image_names:
            image_name_no_ext = os.path.splitext(image_name)[0]

            back_path = os.path.join(back_result_f, image_name)
            fore_path = f"{os.path.join(fore_result_f, image_name_no_ext)}.png"
            harm_path = f"{os.path.join(harm_result_path, image_name_no_ext)}.png"

            try:
                with open(back_path, 'rb') as fin:
                    back_data = fin.read()
                with open(fore_path, 'rb') as fin:
                    fore_data = fin.read()

                response = requests.post(URL, json=make_query(base64.b64encode(back_data), base64.b64encode(fore_data)))

                res = json.loads(response.content)
                # encoded_image = base64.b64encode((res['result']))
                # decoded_image = base64.b64decode(encoded_image)

                if 'result' in res:
                    with open(harm_path, 'wb') as f:
                        f.write(base64.decodebytes(str(res['result']).encode('utf-8')))
                    print(f"DONE {harm_path}")

                else:
                    print(res['error'])
                # im = Image.open(BytesIO(decoded_image))
                # im.save(img_harm_path, 'png')

            except Exception as e:
                print(f"FAIL {harm_path}")
                traceback.print_exc()

