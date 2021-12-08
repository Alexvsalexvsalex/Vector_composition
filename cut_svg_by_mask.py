import os
import traceback
from pathlib import Path
import js2py
import svgpathtools
from svgpathtools import svg2paths2, wsvg
from selenium import webdriver
import numpy
from utils import read_image

CUR_DIR = os.path.dirname(os.path.realpath(__file__))
SVG_DIR = os.path.join(CUR_DIR, 'svg')
CUT_SVG_DIR = os.path.join(CUR_DIR, 'svg_cut')
MASK_PATHS_DIR = os.path.join(CUR_DIR, 'mask_paths')


if __name__ == '__main__':
    # with webdriver.Chrome(executable_path='./chromedriver') as driver:
    datasets = os.listdir(SVG_DIR)
    for dataset in datasets:
        print(f"FOUND DATASET {dataset}")

        dataset_vect_path = os.path.join(SVG_DIR, dataset)

        dataset_cut_in_svg_path = os.path.join(CUT_SVG_DIR, dataset, 'in')
        dataset_cut_out_svg_path = os.path.join(CUT_SVG_DIR, dataset, 'out')
        Path(dataset_cut_in_svg_path).mkdir(parents=True, exist_ok=True)
        Path(dataset_cut_out_svg_path).mkdir(parents=True, exist_ok=True)

        dataset_mask_path = os.path.join(MASK_PATHS_DIR, dataset)

        image_names = os.listdir(dataset_vect_path)

        for image_name in image_names:
            image_name_no_ext = os.path.splitext(image_name)[0]

            image_path = os.path.join(dataset_vect_path, image_name)
            img_result_in_path = f"{os.path.join(dataset_cut_in_svg_path, image_name_no_ext)}.svg"
            img_result_out_path = f"{os.path.join(dataset_cut_out_svg_path, image_name_no_ext)}.svg"
            img_mask_path = f"{os.path.join(dataset_mask_path, image_name_no_ext)}.svg"

            try:
                # driver.get('file:' + image_path)
                # arr = read_image(img_mask_path)
                # img_size = arr.shape[:2]
                mask_paths, mask_attributes, mask_svg_attributes = svg2paths2(img_mask_path)
                paths, attributes, svg_attributes = svg2paths2(image_path)
                in_paths = []
                in_attrs = []
                out_paths = []
                out_attrs = []
                for i in range(len(paths)):
                    inside = False
                    for mask_path in mask_paths:
                        inside = inside or paths[i].is_contained_by(mask_path)
                    if inside:
                        in_paths.append(paths[i])
                        in_attrs.append(attributes[i])
                        # in_svg_attrs.append(svg_attributes[i])
                    else:
                        out_paths.append(paths[i])
                        out_attrs.append(attributes[i])
                        # out_svg_attrs.append(svg_attributes[i])
                wsvg(in_paths, attributes=in_attrs, svg_attributes=svg_attributes, filename=img_result_in_path)
                wsvg(out_paths, attributes=out_attrs, svg_attributes=svg_attributes, filename=img_result_out_path)

                # with open(image_path, 'r') as file:
                #     raw_text = "".join(file.readlines()[2:])
                # r: svgpathtools.Path = None
                # r.is_contained_by()
                # path_id = [[[] for _ in range(img_size[1])] for _ in range(img_size[0])]
                # js_script = \
                #     f"const svg = document.getElementsByTagName(\"svg\")[0];" \
                #     f"const all = document.getElementsByTagName(\"path\");" \
                #     f"var res = [];" \
                #     f"for(var x = 100; x < 105; x++) for(var y = 0; y < {img_size[1]}; y++) for(var i = 0, max = all.length; i < max; i++) " \
                #     "{let point = svg.createSVGPoint();point.x = x;point.y = y; if (all[i].isPointInStroke(point)) res.push([x, y, i]);}" \
                #     f"return res;"
                # path_id = driver.execute_script(js_script)
                # for path in paths:
                #     p: svgpathtools.Path = path
                    # p.
                # wsvg(paths, attributes=attributes, svg_attributes=svg_attributes, filename=img_result_path)

                # image = instance_segmentation_api(img_result_path)

                # Image.fromarray(image, 'RGB').save(img_mask_path)

                print(f"DONE {image_path}")
            except Exception as e:
                print(f"FAIL {image_path}")
                traceback.print_exc()