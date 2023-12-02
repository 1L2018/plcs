'''
Author: liu_0000 1360668195@qq.com
Date: 2023-12-02 11:52:43
LastEditors: liu_0000 1360668195@qq.com
LastEditTime: 2023-12-02 12:05:25
FilePath: /plcs-onnx/test/base.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from abc import abstractmethod
import os.path
import pathlib
from time import time
from typing import Tuple
import cv2
import numpy as np

from openpyxl import Workbook
import torch
from tqdm import tqdm

from tools.pipeline import TestPipeline
from tools.post_procecss import PostProcessing


class BasePredict:
    def __init__(
            self,
            model_path: str, 
            input_size: Tuple[int, int]=(1024,1024),
            device: str = "cuda:0",
            score_threshold: int = 0.1,
            num_keypoints: int = 3
            ) -> None:
        self.model_path = model_path
        self.input_size = input_size
        self.device = device
        self.score_threshold = score_threshold
        self.num_keypoints = num_keypoints
        self.trans = TestPipeline(input_size,device)
        self.post = PostProcessing(score_threshold)
    
    def preprocess(self, image_path: str):
        """
        返回网络输入，以及待可视化的图像
        """
        # 图像预处理 读取图像->resize->Tensor->Normalize
        inputs,vis = self.trans(image_path)
        return inputs,vis

    def visualize(self,image: np.ndarray,save_path: str,points: np.ndarray,texts: str="") -> None:
        vis_image = image.copy()
        for x,y in points[0]:
            vis_image = cv2.circle(vis_image,(int(x),int(y)),1,(0,0,255),-1)
        vis_image = cv2.putText(vis_image, texts, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imwrite(save_path, vis_image)

    @abstractmethod
    def predict(self, inputs:torch.Tensor):
        pass
    def calculate_accuracy(self,gt,pred):
        diff = abs(pred-gt)
        return (1 - abs(pred-gt)/gt) if diff<gt else 0
    def batch_predict(self, input_dir: str, save_dir:str, excel_name:str,is_visualize=False):
        wb = Workbook()
        ws = wb.create_sheet("plcs")
        ws.append(['filename','time_preprocess','time_predict','time_visualize','pred', 'gt','diff','accuracy'])
        

        image_list = list(pathlib.Path(input_dir).rglob("*.jpeg"))
        for image_path in tqdm(image_list):
            t1 = time()
            inputs,vis = self.preprocess(str(image_path))
            t2 = time()
            points,scores = self.predict(inputs)
            t3 = time()
            gt = int(image_path.parent.name)
            pred = scores[0].shape[0] // self.num_keypoints
            pathlib.Path(save_dir).mkdir(parents=True,exist_ok=True) 
            if is_visualize:
                save_path = pathlib.Path(save_dir) / image_path.parent.name / image_path.name
                save_path.parent.mkdir(parents=True,exist_ok=True)
                self.visualize(vis,str(save_path),points,texts=f"gt:{gt}, pred:{pred}")
            t4 = time()
            ws.append([image_path.name,round(t2-t1,3),round(t3-t2,3),round(t4-t3,3),pred,gt,pred-gt,self.calculate_accuracy(gt,pred)])
        wb.save(os.path.join(save_dir ,excel_name))