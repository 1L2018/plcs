'''
Author: William Wang 1309508438@qq.com
Date: 2023-12-01 11:17:54
LastEditors: liu_0000 1360668195@qq.com
LastEditTime: 2023-12-02 12:20:04
FilePath: /plcs-onnx/main.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from typing import Tuple
import onnxruntime as ort
import torch
from .base import BasePredict

class OnnxPredict(BasePredict):
    def __init__(
            self,
            input_size:Tuple[int, int],
            onnx_path: str ,
            score_threshold: int = 0.1,
            device: str = "cpu",
            ) -> None:
        super().__init__(onnx_path, input_size, device, score_threshold)
        if self.device == "cpu":
            self.ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        else:
            self.ort_session = ort.InferenceSession(onnx_path,providers=['CUDAExecutionProvider'])
        self.iobinding = self.ort_session.io_binding()

    def predict(self, inputs:torch.Tensor):
        # 绑定onnx的输入
        element_type = inputs.new_zeros(
                        1, device='cpu').numpy().dtype
        self.iobinding.bind_input(
            name="input",
            device_type=self.device,
            device_id=-1 if self.device == 'cpu' else 0,
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


if __name__ == "__main__":
    input_dir = "/home/chuanzhi/mnt_2T/lrx/dataset/penaeus_1k/test/p10/"
    save_dir = "./result"
    onnx_path = "/home/chuanzhi/mnt_2T/lrx/code/plcs-onnx/weights/plcs_hrnet-w32_cpu_8xb24-300e_coco-1024x1024.onnx"
    P = OnnxPredict((1024,1024),onnx_path)
    P.batch_predict(input_dir,save_dir,"plcs_onnx.xlsx",True)