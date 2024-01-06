'''
Author: liu_0000 1360668195@qq.com
Date: 2023-12-02 10:45:07
LastEditors: liu_0000 1360668195@qq.com
LastEditTime: 2023-12-25 21:40:34
FilePath: /plcs-onnx/main.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from predict.onnx import OnnxPredict
from predict.plcsler_onnx import PlcsLerOnnxPredict
# from predict.tensort import TensorrtPredict

def visualize():
    input_dir = "/home/chuanzhi/mnt_2T/lrx/dataset/penaeus_1k/test/p10/"
    xlsxName = "round{}_mask{}_dialate{}_score{}-round{}_mask{}_dialate{}_score{}"\
        .format(nowRound,nowMaskWeights,nowMaskDialate,nowScoreThreshold,
                preRound,preMaskWeights,preMaskDialate,preScoreThreshold)
    save_dir = save_dir + "/vis"
    PO = OnnxPredict((1024,1024),onnx_path,device="cuda",score_threshold=nowScoreThreshold,num_keypoints=1)
    PO.batch_predict(input_dir,
                     save_dir,
                    xlsxName,
                    is_visualize
                    )

def coco(save_dir):
    input_dir = "/home/chuanzhi/mnt_2T/lrx/dataset/penaeus_ler/20231218/images"
    # roundNow/nowParam/RoundPre/preParam
    
    save_dir = save_dir + "/coco"
    plcsler = PlcsLerOnnxPredict(save_dir,
                       onnx_path=onnx_path,
                       score_threshold=nowScoreThreshold,
                       device=device,
                       num_keypoints=num_keypoints
                       )
    plcsler.batch_coco(
        input_dir,
        save_dir,
        is_visualize=is_visualize,
        is_coco=is_coco
    )

# engine_path = "weights/hrnet/tensorrt/plcs_hrnet-w32_tensorrt_8xb24-300e_coco-1024x1024.engine"
# onnx_path = "weights/hrnet/unquant/plcs_hrnet-w32_gpu_8xb24-300e_coco-1024x1024.onnx"
# PT = TensorrtPredict((1024,1024),engine_path)
# PT.batch_predict(input_dir,save_dir,"plcs_hrnet-w32_tensorrt.xlsx",False)
# PO = OnnxPredict((1024,1024),onnx_path,device="cuda",score_threshold=0.1,num_keypoints=3)
# PO.batch_predict(input_dir,save_dir,"plcs-hrnet-w32-test.xlsx",True)
# coco_root = "/home/chuanzhi/mnt_2T/lrx/code/plcs-onnx/result/plcsler-hrnet-round1"
# input_dir = "/home/chuanzhi/mnt_2T/lrx/dataset/penaeus_ler/images"

nowRound=3
nowMaskWeights=0.0
nowMaskDialate=0
nowScoreThreshold=0.7
preRound=2
preMaskWeights=0.0
preMaskDialate=0
preScoreThreshold=0.5
onnx_path = "weights/plcser/20231224/l1loss/round3/mask0.0dialate0/round2/mask0.0dialate0score0.5/round3.onnx"
device="cuda"
is_coco=True
is_visualize=True
num_keypoints=1
save_dir = "./result/20231224/l1loss/round{}/mask{}_dialate{}_score{}/round{}/mask{}_dialate{}_score{}"\
        .format(nowRound,nowMaskWeights,nowMaskDialate,nowScoreThreshold,
                preRound,preMaskWeights,preMaskDialate,preScoreThreshold)
coco(save_dir)
# visualize()

