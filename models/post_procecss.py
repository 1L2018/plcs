'''
Author: William Wang 1309508438@qq.com
Date: 2023-12-01 14:47:27
LastEditors: liu_0000 1360668195@qq.com
LastEditTime: 2023-12-07 22:07:27
FilePath: /plcs-onnx/post_procecss.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from typing import Any, Tuple
import numpy as np
import torch.nn.functional as F

import torch

from tools.tensor_utils import to_numpy


class PostProcessing:
    def __init__(self,score_threshold: int = 0.3,decode_topk=3000) -> None:
        self.score_threshold = score_threshold
        self.decode_topk = decode_topk
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
        """
        @return: batch_locs: [B,K,N,2] batch_scores: [B,K,N,1]
        """
        h,w = inputs.shape[2:]
        _heatmaps = F.interpolate(
            outputs,
            size=(h, w),
            mode='bilinear')
        B,K,H,W = _heatmaps.shape
        # Heatmap NMS
        _heatmaps = self.batch_heatmap_nms(_heatmaps)

        topk_vals, topk_indices = _heatmaps.flatten(-2, -1).topk(self.decode_topk, dim=-1)
        topk_locs = torch.stack([topk_indices % W, topk_indices // W],dim=-1)  # (B, K, TopK, 2)
        topk_vals = to_numpy(topk_vals)
        topk_locs = to_numpy(topk_locs)
        batch_locs = []
        batch_scores = []

        
        for b in range(B):
            channel_vals = topk_vals[b,0]
            channel_scores = []
            channel_locs = []
            for k in range(K):
                channel_vals = topk_vals[b,k]
                scores = channel_vals[channel_vals>self.score_threshold]
                keypoints = topk_locs[b,k,channel_vals>self.score_threshold]
                channel_scores.append(scores)
                channel_locs.append(keypoints)
            batch_scores.append(channel_scores)
            batch_locs.append(channel_locs)   

        return batch_locs,batch_scores