<!--
 * @Author: liu_0000 1360668195@qq.com
 * @Date: 2023-12-01 21:51:16
 * @LastEditors: liu_0000 1360668195@qq.com
 * @LastEditTime: 2023-12-03 18:17:14
 * @FilePath: /plcs-onnx/readme.md
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
-->
## 使用说明
+ 这是plcs_onnx测试代码项目,`onnx`文件通过`mmdeploy`项目将`mmpose`代码转换而来，`tools`文件夹中包含的是图像预处理代码、模型后处理代码；
+ 使用CPU时需要切换环境为`cutlerByLRX`，GPU切换环境为`mmdeploy`,并在`main.py`中切换`device`；以下报错既是环境错误：
```
/opt/rh/devtoolset-8/root/usr/include/c++/8/bits/stl_vector.h:932: std::vector<_Tp, _Alloc>::reference std::vector<_Tp, _Alloc>::operator[](std::vector<_Tp, _Alloc>::size_type) [with _Tp = int; _Alloc = std::allocator<int>; std::vector<_Tp, _Alloc>::reference = int&; std::vector<_Tp, _Alloc>::size_type = long unsigned int]: Assertion '__builtin_expect(__n < this->size(), true)' failed.

```
+ `weights/unquant`中保存的是 **非量化onnx权重**,hrnet_w32, gpu/cpu；
+ 推理代码全部放在predict中，继承自 **BasePredict**；
+ `onnx/tensorrt`推理结束后得到的结果会重新放到GPU中，相比于*cpu后处理*，gpu的处理速度下降了2个量级，从0.03->0.0001,目前处理速度瓶颈主要在预处理部分，后续想办法优化。
+ 预处理阶段耗时：总时长0.08，`cv2.imread()`读取图片时长接近于*0.08*，warpAffine操作仅耗时0.001