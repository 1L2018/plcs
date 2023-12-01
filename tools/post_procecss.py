'''
Author: William Wang 1309508438@qq.com
Date: 2023-12-01 14:47:27
LastEditors: William Wang 1309508438@qq.com
LastEditTime: 2023-12-01 15:20:54
FilePath: /plcs-onnx/post_procecss.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from typing import Any, Tuple
import numpy as np
import torch.nn.functional as F

import torch


class PostProcessing:
    def __init__(self,score_threshold: int = 0.1) -> None:
        self.score_threshold = score_threshold
    def batch_heatmap_nms(self,batch_heatmaps: torch.Tensor, kernel_size: int = 5):
        """Apply NMS on a batch of heatmaps.

        Args:
            batch_heatmaps (Tensor): batch heatmaps in shape (B, K, H, W)
            kernel_size (int): The kernel size of the NMS which should be
                a odd integer. Defaults to 5

        Returns:
            Tensor: The batch heatmaps after NMS.
        """

        assert isinstance(kernel_size, int) and kernel_size % 2 == 1, \
            f'The kernel_size should be an odd integer, got {kernel_size}'

        padding = (kernel_size - 1) // 2

        maximum = F.max_pool2d(
            batch_heatmaps, kernel_size, stride=1, padding=padding)
        maximum_indicator = torch.eq(batch_heatmaps, maximum)
        batch_heatmaps = batch_heatmaps * maximum_indicator.float()

        return batch_heatmaps


    def __call__(self,inputs: torch.Tensor, outputs: torch.Tensor) -> Any:
        h,w = inputs.shape[2:]
        _heatmaps = F.interpolate(
            outputs,
            size=(h, w),
            mode='bilinear')
        B,K,H,W = _heatmaps.shape
        # Heatmap NMS
        _heatmaps = self.batch_heatmap_nms(_heatmaps)
        _heatmaps[_heatmaps < self.score_threshold] = 0

        locations = _heatmaps.nonzero()
        scores = _heatmaps[locations[...,0],locations[...,1],locations[...,2],locations[...,3]]
        locations = locations.numpy()
        scores = scores.numpy()
        batch_locations = []
        batch_scores = []
        for b in range(B):
            condi = locations[:,0]==b
            batch_locations.append(locations[condi][...,-2:][:,::-1])
            batch_scores.append(scores[condi])
        return batch_locations,batch_scores