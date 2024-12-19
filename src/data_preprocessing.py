import os
import json
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm

def convert_labelme_json_to_masks(json_dir, masks_dir):
    files = os.listdir(json_dir)
    print(f"文件目录中的文件：{files}")

    # 定义一个标签值，用于表示病害区域
    damaged = 255  # 或者其他您选择的值

    for file in tqdm(files):
        if file.endswith(".json"):
            json_path = os.path.join(json_dir, file)
            print(f"正在处理 JSON 文件：{json_path}")

            with open(json_path, 'r') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError as e:
                    print(f"读取 JSON 文件时出错 {json_path}: {e}")
                    continue

            # 检查 JSON 文件结构
            if 'shapes' not in data or 'imageHeight' not in data or 'imageWidth' not in data:
                print(f"{json_path} 中的 JSON 结构无效")
                continue

            img_shape = (data['imageHeight'], data['imageWidth'])
            mask = np.zeros(img_shape, dtype=np.uint8)

            for shape in data['shapes']:
                shape_type = shape.get('shape_type', None)
                print(f"形状类型：{shape_type}")

                if shape_type == 'polygon':
                    points = np.array(shape['points'], dtype=np.int32)
                    points = points.reshape((-1, 1, 2))  # 将点转换为多边形格式
                    cv2.fillPoly(mask, [points], color=damaged)

                elif shape_type == 'circle':
                    # 计算圆心和半径
                    if len(shape['points']) == 2:
                        p1, p2 = shape['points']
                        p1 = np.array(p1)
                        p2 = np.array(p2)

                        # 圆心是两个点的中点
                        cx, cy = (p1 + p2) / 2
                        # 半径是两个点之间的距离的一半
                        r = int(np.linalg.norm(p1 - p2) / 2)

                        cx, cy = int(cx), int(cy)  # 转换为整数
                        cv2.circle(mask, (cx, cy), r, color=damaged, thickness=-1)
                    else:
                        print(f"圆形数据不完整，跳过 {json_path}")

                elif shape_type == 'rectangle':  # 处理矩形
                    x, y, w, h = map(int, shape['points'][0] + shape['points'][1])
                    cv2.rectangle(mask, (x, y), (x + w, y + h), color=damaged, thickness=-1)

                elif shape_type == 'line' or shape_type == 'point':
                    # 如果不支持的形状，选择跳过
                    print(f"跳过不支持的形状类型：{shape_type}")
                else:
                    print(f"不支持的形状类型：{shape_type}")

            # 打印掩码的唯一值
            print(f"掩码的唯一值：{np.unique(mask)}")

            # 保存掩码为图像
            mask_filename = file.replace('.json', '_mask.png')
            mask_path = os.path.join(masks_dir, mask_filename)
            Image.fromarray(mask).convert('L').save(mask_path)
            print(f"已保存掩码：{mask_path}")

# 调用函数时，传递实际路径
json_directory = r"D:\Projects\pythonProject\unet\data\json"
masks_directory = r"D:\Projects\pythonProject\unet\data\masks"
convert_labelme_json_to_masks(json_directory, masks_directory)
