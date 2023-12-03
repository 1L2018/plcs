'''
Author: liu_0000 1360668195@qq.com
Date: 2023-12-02 10:45:07
LastEditors: liu_0000 1360668195@qq.com
LastEditTime: 2023-12-03 17:17:09
FilePath: /plcs-onnx/main.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from predict.onnx import OnnxPredict
from predict.tensort import TensorrtPredict


input_dir = "/home/chuanzhi/mnt_2T/lrx/dataset/penaeus_1k/test/p10/"
save_dir = "./result/litehrnet/"
engine_path = "weights/tensorrt/plcs_hrnet-w32_tensorrt_8xb24-300e_coco-1024x1024.engine"
onnx_path = "weights/litehrnet/plcs_litehrnet-18_gpu_8xb24-300e_coco-1024x1024.onnx"
# PT = TensorrtPredict((1024,1024),engine_path)
# PT.batch_predict(input_dir,save_dir,"plcs_tensorrt.xlsx",False)
PO = OnnxPredict((1024,1024),onnx_path,device="cuda")
PO.batch_predict(input_dir,save_dir,"plcs_litehrnet_18_gpu_onnx.xlsx",False)