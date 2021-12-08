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
import svgpathtools
from svgpathtools import wsvg, svg2paths2

from utils import read_image


CUR_DIR = os.path.dirname(os.path.realpath(__file__))
SVG_DIR = os.path.join(CUR_DIR, 'svg')
MASKS_DIR = os.path.join(CUR_DIR, 'masks')
MASK_PATHS_DIR = os.path.join(CUR_DIR, 'mask_paths')


if __name__ == '__main__':
    datasets = os.listdir(SVG_DIR)
    for dataset in datasets:
        print(f"FOUND DATASET {dataset}")

        dataset_vect_path = os.path.join(SVG_DIR, dataset)
        dataset_mask_path = os.path.join(MASKS_DIR, dataset)

        dataset_result_path = os.path.join(MASK_PATHS_DIR, dataset)
        Path(dataset_result_path).mkdir(parents=True, exist_ok=True)

        image_names = os.listdir(dataset_vect_path)

        for image_name in image_names:
            image_name_no_ext = os.path.splitext(image_name)[0]

            image_path = os.path.join(dataset_vect_path, image_name)
            img_mask_path = f"{os.path.join(dataset_mask_path, image_name_no_ext)}.png"
            img_result_path = f"{os.path.join(dataset_result_path, image_name_no_ext)}.svg"

            try:
                _, _, svg_attributes = svg2paths2(image_path)

                arr = read_image(img_mask_path)
                img_size = arr.shape[:2]

                mask = [[False for j in range(img_size[1])] for i in range(img_size[0])]
                for i in range(1, img_size[0] - 1):
                    for j in range(1, img_size[1] - 1):
                        white = arr[i][j].max() == 255
                        mask[i][j] = white

                used = [[False for j in range(img_size[1])] for i in range(img_size[0])]
                comp = [[0 for j in range(img_size[1])] for i in range(img_size[0])]
                comp_index = 0
                for i in range(1, img_size[0] - 1):
                    for j in range(1, img_size[1] - 1):
                        if not used[i][j] and mask[i][j]:
                            comp_index += 1
                            queue = [(i, j)]
                            used[i][j] = True
                            while len(queue) > 0:
                                x, y = queue[-1]
                                comp[x][y] = comp_index
                                queue.pop()
                                if 0 < x < img_size[0] and 0 < y < img_size[1]:
                                    for diff_x in range(-1, 2):
                                        for diff_y in range(-1, 2):
                                            xxx = diff_x + x
                                            yyy = diff_y + y
                                            if not used[xxx][yyy] and mask[xxx][yyy]:
                                                used[xxx][yyy] = True
                                                queue.append((xxx, yyy))
                c_used = [False for i in range(comp_index + 1)]
                paths = []
                if 'viewBox' in svg_attributes:
                    view_box = list(map(lambda x: float(x.strip(' ,')), svg_attributes['viewBox'].split(' ')))
                    width_vb = view_box[2] - view_box[0]
                    height_vb = view_box[3] - view_box[1]
                    offset_w = view_box[0]
                    offset_h = view_box[1]
                else:
                    width_vb = float(svg_attributes['width'])
                    height_vb = float(svg_attributes['height'])
                    offset_w = 0
                    offset_h = 0
                for i in range(img_size[0]):
                    for j in range(img_size[1] - 1):
                        if not mask[i][j] and mask[i][j + 1]:
                            cur_c = comp[i][j + 1]
                            if not c_used[cur_c]:
                                c_used[cur_c] = True
                                d_ = 0 # right, down, left, up
                                d_x = [0, 1, 0, -1]
                                d_y = [1, 0, -1, 0]
                                cur_x = i
                                cur_y = j
                                first = True
                                coords = []
                                while first or not (cur_x == i and cur_y == j and d_ == 0):
                                    first = False
                                    tup = (cur_x, cur_y)
                                    if len(coords) == 0 or tup != coords[-1]:
                                        coords.append(tup)
                                    nxt_d_ = (d_ + 1) % 4
                                    prev_d_ = (d_ - 1 + 4) % 4
                                    nxt_cur_x = cur_x + d_x[prev_d_]
                                    nxt_cur_y = cur_y + d_y[prev_d_]
                                    if not mask[nxt_cur_x][nxt_cur_y]:
                                        nxt2_cur_x = nxt_cur_x + d_x[d_]
                                        nxt2_cur_y = nxt_cur_y + d_y[d_]
                                        if not mask[nxt2_cur_x][nxt2_cur_y]:
                                            tup = (nxt_cur_x, nxt_cur_y)
                                            if len(coords) == 0 or tup != coords[-1]:
                                                coords.append(tup)
                                            d_ = nxt_d_
                                            cur_x = nxt2_cur_x
                                            cur_y = nxt2_cur_y
                                        else:
                                            cur_x = nxt_cur_x
                                            cur_y = nxt_cur_y
                                    else:
                                        d_ = prev_d_
                                if len(coords) == 0 or coords[0] != coords[-1]:
                                    coords.append(coords[0])
                                p_coords = list(map(lambda x: complex(
                                    x[1] * height_vb / img_size[0] + offset_w,
                                    x[0] * width_vb / img_size[1] + offset_h), coords))
                                lines = [svgpathtools.Line(p_coords[k], p_coords[k + 1]) for k in range(len(coords) - 1)]
                                paths.append(svgpathtools.Path(*lines))
                wsvg(paths, stroke_widths=[1] * len(paths), svg_attributes=svg_attributes, filename=img_result_path)

                print(f"DONE {image_path}")
            except Exception as e:
                print(f"FAIL {image_path}")
                traceback.print_exc()
