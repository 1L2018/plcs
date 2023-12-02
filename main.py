from predict.onnx import OnnxPredict
from predict.tensort import TensorrtPredict


input_dir = "/home/chuanzhi/mnt_2T/lrx/dataset/penaeus_1k/test/p10/"
save_dir = "./result"
engine_path = "weights/tensorrt/plcs_hrnet-w32_tensorrt_8xb24-300e_coco-1024x1024.engine"
onnx_path = "weights/unquant/plcs_hrnet-w32_gpu_8xb24-300e_coco-1024x1024.onnx"
PT = TensorrtPredict((1024,1024),engine_path)
PT.batch_predict(input_dir,save_dir,"plcs_tensorrt.xlsx",False)
# PO = OnnxPredict((1024,1024),onnx_path)
# PO.batch_predict(input_dir,save_dir,"plcs_onnx.xlsx",False)