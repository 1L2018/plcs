'''
Author: William Wang 1309508438@qq.com
Date: 2023-12-01 11:17:54
LastEditors: William Wang 1309508438@qq.com
LastEditTime: 2023-12-01 16:39:06
FilePath: /plcs-onnx/main.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import pathlib
import os.path
from time import time
from typing import Tuple
import cv2
import numpy as np
import onnxruntime as ort
from openpyxl import Workbook
import torch
from tqdm import tqdm

from tools.pipeline import TestPipeline
from tools.post_procecss import PostProcessing
class Predict:
    def __init__(
            self,
            input_size:Tuple[int, int],
            onnx_path: str ,
            score_threshold: int = 0.1,
            device: str = "cpu",
            ) -> None:
        assert onnx_path is not None
        self.device = device
        self.input_size = input_size
        self.onnx_path = onnx_path
        self.trans = TestPipeline(input_size,device)
        self.post = PostProcessing(score_threshold)
        self.ort_session = ort.InferenceSession(onnx_path,providers=['CUDAExecutionProvider'])
        self.iobinding = self.ort_session.io_binding()

    def preprocess(self, image_path: str):
        """
        返回网络输入，以及待可视化的图像
        """
        # 图像预处理 读取图像->resize->Tensor->Normalize
        inputs,vis = self.trans(image_path)
        return inputs,vis

    def predict(self, inputs:torch.Tensor):
        # 绑定onnx的输入
        element_type = inputs.new_zeros(
                        1, device='cpu').numpy().dtype
        self.iobinding.bind_input(
            name="input",
            device_type="cuda",
            device_id=0,
            element_type=element_type,
            shape=inputs.shape,
            buffer_ptr=inputs.data_ptr())
        # 规定输出空间
        self.iobinding.bind_output("output")
        # 推理
        self.ort_session.run_with_iobinding(self.iobinding)
        # 获取预测结果
        output_data = self.iobinding.copy_outputs_to_cpu()
        outputs = torch.from_numpy(output_data[0])
        # 预测结果后处理
        points,scores = self.post(inputs,outputs)
        
        return points,scores
    def visualize(self,image: np.ndarray,save_path: str,points: np.ndarray,texts: str="") -> None:
        vis_image = image.copy()
        for x,y in points[0]:
            vis_image = cv2.circle(vis_image,(int(x),int(y)),1,(0,0,255),-1)
        vis_image = cv2.putText(vis_image, texts, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imwrite(save_path, vis_image)

def calculate_accuracy(gt,pred):
    diff = abs(pred-gt)
    return (1 - abs(pred-gt)/gt) if diff<gt else 0
def batch_predict(
        input_dir: str,
        save_dir:str, 
        onnx_path: str, 
        num_keypoints: int = 3,
        input_size: Tuple[int, int] = (1024,1024), 
        device: str = "cuda:0"):
        wb = Workbook()
        ws = wb.create_sheet("plcs")
        ws.append(['filename','time_preprocess','time_predict','time_visualize','pred', 'gt','diff','accuracy'])
        
        P = Predict(input_size,onnx_path=onnx_path,device=device)

        image_list = list(pathlib.Path(input_dir).rglob("*.jpeg"))
        for image_path in tqdm(image_list):
            t1 = time()
            inputs,vis = P.preprocess(str(image_path))
            t2 = time()
            points,scores = P.predict(inputs)
            t3 = time()
            gt = int(image_path.parent.name)
            pred = scores[0].shape[0] // num_keypoints
            save_path = pathlib.Path(save_dir) / image_path.parent.name / image_path.name
            save_path.parent.mkdir(parents=True,exist_ok=True)
            P.visualize(vis,str(save_path),points,texts=f"gt:{gt}, pred:{pred}")
            t4 = time()
            ws.append([image_path.name,round(t2-t1,3),round(t3-t2,3),round(t4-t3,3),pred,gt,pred-gt,calculate_accuracy(gt,pred)])
        wb.save(os.path.join(save_dir ,'plcs.xlsx'))


if __name__ == "__main__":
    input_dir = "/home/chuanzhi/mnt_2T/lrx/dataset/penaeus_1k/test/p10/"
    save_dir = "./result"
    onnx_path = "weights/plcs_hrnet-w32_8xb24-300e_coco-1024x1024.onnx"
    batch_predict(input_dir,save_dir,onnx_path)
