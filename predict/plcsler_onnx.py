'''
Author: liu_0000 1360668195@qq.com
Date: 2023-12-07 21:03:00
LastEditors: liu_0000 1360668195@qq.com
LastEditTime: 2023-12-08 14:16:33
FilePath: /plcs-onnx/predict/plcsler_onnx.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os.path
import pathlib
from time import time
from typing import Tuple
import onnxruntime as ort
from openpyxl import Workbook
import torch
from tqdm import tqdm

from tools.coco import COCO
from .onnx import OnnxPredict

class PlcsLerOnnxPredict(OnnxPredict):
    def __init__(self,coco_root:str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.coco_root = coco_root
        self.coco = COCO(coco_root)
    def convert2coco(self, points,image_path):
        """
        将预测数据保存为coco格式
        points: [B,K,N,2],N不相等
        """
        assert len(points)==1, "batch_size must be 1"
        points = points[0]
        self.coco.convert(points,image_path)
    def restore_keypoint_origin(self,keypoints,preresults):
        """
        回复1024热力图坐标到原图大小
        """
        assert len(keypoints)==1, "batch_size must be 1"
        keypoints = keypoints[0]
        input_size = preresults['input_size']
        input_center = preresults['input_center']
        input_scale = preresults['input_scale']
        for k in range(len(keypoints)):
            keypoints[k] = keypoints[k] / input_size * input_scale
            keypoints[k] += input_center - 0.5 * input_scale
        return [keypoints]
    def batch_coco(self, input_dir: str, save_dir: str, is_visualize=False,is_coco=False):
        image_list = list(pathlib.Path(input_dir).rglob("*.jpeg"))
        for image_path in tqdm(image_list):
            preresults = self.preprocess(str(image_path))
            inputs = preresults['inputs']
            vis = preresults['vis']
            outputs = self.predict(inputs)
            points,scores = self.post(inputs,outputs)
            pred = self.get_pred(scores) // self.num_keypoints
            pathlib.Path(save_dir).mkdir(parents=True,exist_ok=True)

            if is_visualize:
                save_path = pathlib.Path(save_dir) / image_path.parent.name / image_path.name
                save_path.parent.mkdir(parents=True,exist_ok=True)
                self.visualize(vis,str(save_path),points,texts=f" pred:{pred}")
            if is_coco:
                points = self.restore_keypoint_origin(points,preresults)
                self.convert2coco(points,image_path)
        self.coco.to_coco()