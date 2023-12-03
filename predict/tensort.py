'''
Author: William Wang 1309508438@qq.com
Date: 2023-12-01 11:17:54
LastEditors: liu_0000 1360668195@qq.com
LastEditTime: 2023-12-03 17:27:15
FilePath: /plcs-onnx/main.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import tensorrt as trt

from typing import Sequence, Tuple
import torch
from .base import BasePredict

class TensorrtPredict(BasePredict):
    def __init__(
            self,
            input_size:Tuple[int, int],
            engine_path: str ,
            score_threshold: int = 0.1,
            device: str = "cuda:0",
            ) -> None:
        super().__init__(engine_path, input_size, device, score_threshold)
        
        self._input_names = None
        self._output_names = None
        self.engine = self.loadEngine(engine_path)
        self.context = self.engine.create_execution_context()
        self.__load_io_names()

    @staticmethod
    def loadEngine(engine_path: str):
        assert engine_path
        with trt.Logger() as logger, trt.Runtime(logger) as runtime:     
            with open(engine_path, mode='rb') as f:
                engine_bytes = f.read()
            trt.init_libnvinfer_plugins(logger, namespace='')
            engine = runtime.deserialize_cuda_engine(engine_bytes)
            return engine
    
    @staticmethod
    def torch_device_from_trt(device: trt.TensorLocation):
        """Convert pytorch device to TensorRT device.

        Args:
            device (trt.TensorLocation): The device in tensorrt.
        Returns:
            torch.device: The corresponding device in torch.
        """
        if device == trt.TensorLocation.DEVICE:
            return torch.device('cuda')
        elif device == trt.TensorLocation.HOST:
            return torch.device('cpu')
        else:
            return TypeError(f'{device} is not supported by torch')


    @staticmethod
    def torch_dtype_from_trt(dtype: trt.DataType) -> torch.dtype:
        """Convert pytorch dtype to TensorRT dtype.

        Args:
            dtype (str.DataType): The data type in tensorrt.

        Returns:
            torch.dtype: The corresponding data type in torch.
        """

        if dtype == trt.bool:
            return torch.bool
        elif dtype == trt.int8:
            return torch.int8
        elif dtype == trt.int32:
            return torch.int32
        elif dtype == trt.float16:
            return torch.float16
        elif dtype == trt.float32:
            return torch.float32
        else:
            raise TypeError(f'{dtype} is not supported by torch')

    def __load_io_names(self):
        """Load input/output names from engine."""
        names = [_ for _ in self.engine]
        input_names = list(filter(self.engine.binding_is_input, names))
        self._input_names = input_names

        if self._output_names is None:
            output_names = list(set(names) - set(input_names))
            self._output_names = output_names

    def __trt_execute(self, bindings: Sequence[int]):
        """Run inference with TensorRT.

        Args:
            bindings (list[int]): A list of integer binding the input/output.
        """
        self.context.execute_async_v2(bindings,
                                      torch.cuda.current_stream().cuda_stream)
        

    def predict(self, input_tensor:torch.Tensor):
        bindings = [None] * (len(self._input_names) + len(self._output_names))
        profile = self.engine.get_profile_shape(0, self._input_names[0])
        # 输出尺寸检查
        assert input_tensor.dim() == len(
                profile[0]), 'Input dim is different from engine profile.'
        for s_min, s_input, s_max in zip(profile[0], input_tensor.shape,
                                             profile[2]):
                assert s_min <= s_input <= s_max, \
                    'Input shape should be between ' \
                    + f'{profile[0]} and {profile[2]}' \
                    + f' but get {tuple(input_tensor.shape)}.'
        
        idx = self.engine.get_binding_index(self._input_names[0])
        assert 'cuda' in input_tensor.device.type
        input_tensor = input_tensor.contiguous()
        if input_tensor.dtype == torch.long:
            input_tensor = input_tensor.int()
        self.context.set_binding_shape(idx, tuple(input_tensor.shape))
        bindings[idx] = input_tensor.contiguous().data_ptr()
        idx = self.engine.get_binding_index(self._output_names[0])
        dtype = self.torch_dtype_from_trt(self.engine.get_binding_dtype(idx))
        shape = tuple(self.context.get_binding_shape(idx))
    
        device = self.torch_device_from_trt(self.engine.get_location(idx))
        outputs = torch.empty(size=shape, dtype=dtype, device=device)
        bindings[idx] = outputs.data_ptr()

        self.__trt_execute(bindings=bindings)

        # 预测结果后处理
        # points,scores = self.post(input_tensor,outputs)
        outputs = outputs.to(self.device)
        return outputs


if __name__ == "__main__":
    input_dir = "/home/chuanzhi/mnt_2T/lrx/dataset/penaeus_1k/test/p10/"
    save_dir = "./result"
    onnx_path = "weights/tensorrt/plcs_hrnet-w32_tensorrt_8xb24-300e_coco-1024x1024.engine"
    P = TensorrtPredict((1024,1024),onnx_path)
    P.batch_predict(input_dir,save_dir,"plcs_tensorrt.xlsx",True)