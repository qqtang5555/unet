import os
import json
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm

def convert_labelme_json_to_masks(json_dir, masks_dir):
    files = os.listdir(json_dir)
    print(f"Files in directory: {files}")

    # 定义一个标签值，用于表示病害区域
    damaged = 255  # 或者其他您选择的整数值

    for file in tqdm(files):
        if file.endswith(".json"):
            json_path = os.path.join(json_dir, file)
            print(f"Processing JSON file: {json_path}")

            with open(json_path, 'r') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError as e:
                    print(f"Error reading JSON file {json_path}: {e}")
                    continue

            # 检查 JSON 文件内容
            if 'shapes' not in data or 'imageHeight' not in data or 'imageWidth' not in data:
                print(f"Invalid JSON structure in {json_path}")
                continue

            img_shape = (data['imageHeight'], data['imageWidth'])
            mask = np.zeros(img_shape, dtype=np.uint8)

            for shape in data['shapes']:
                shape_type = shape.get('shape_type', None)
                print(f"Shape type: {shape_type}")

                if shape_type == 'polygon':
                    points = np.array(shape['points'], dtype=np.int32)
                    points = points.reshape((-1, 1, 2))  # 将点转换为多边形所需的格式
                    cv2.fillPoly(mask, [points], color=damaged)

                elif shape_type == 'circle':
                    # 假设圆形由 'cx', 'cy', 和 'r' 定义
                    cx, cy = map(int, shape['cx'], shape['cy'])  # 注意：这里需要您检查 JSON 的实际结构
                    r = int(shape['r'])
                    cv2.circle(mask, (cx, cy), r, color=damaged, thickness=-1)

                elif shape_type == 'rectangle':  # 如果您的数据中有矩形，可以添加此处理
                    x, y, w, h = map(int, shape['points'][0] + shape['points'][1])  # 假设矩形由对角线上的两个点定义
                    cv2.rectangle(mask, (x, y), (x+w, y+h), color=damaged, thickness=-1)

                elif shape_type == 'line' or shape_type == 'point':
                    # 对于线和点，您可能想要不同的处理方式，或者简单地忽略它们
                    print(f"Skipping unsupported shape type: {shape_type}")

                else:
                    print(f"Unsupported shape type: {shape_type}")

            # 检查掩码唯一值
            print(f"Unique mask values: {np.unique(mask)}")

            mask_filename = file.replace('.json', '_mask.png')
            mask_path = os.path.join(masks_dir, mask_filename)
            Image.fromarray(mask).convert('L').save(mask_path)  # 确保以灰度图像保存
            print(f"Saved mask to: {mask_path}")

# 调用函数时，传递实际的路径
json_directory = r"D:\Projects\pythonProject\unet\data\json"
masks_directory = r"D:\Projects\pythonProject\unet\data\masks"
convert_labelme_json_to_masks(json_directory, masks_directory)