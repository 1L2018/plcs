
import math
from typing import Any, Sequence, Tuple, Union
import cv2
import torch.nn.functional as F

import numpy as np
import torch


def _rotate_point(pt: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate a point by an angle.

    Args:
        pt (np.ndarray): 2D point coordinates (x, y) in shape (2, )
        angle_rad (float): rotation angle in radian

    Returns:
        np.ndarray: Rotated point in shape (2, )
    """

    sn, cs = np.sin(angle_rad), np.cos(angle_rad)
    rot_mat = np.array([[cs, -sn], [sn, cs]])
    return rot_mat @ pt


def _get_3rd_point(a: np.ndarray, b: np.ndarray):
    """To calculate the affine matrix, three pairs of points are required. This
    function is used to get the 3rd point, given 2D points a & b.

    The 3rd point is defined by rotating vector `a - b` by 90 degrees
    anticlockwise, using b as the rotation center.

    Args:
        a (np.ndarray): The 1st point (x,y) in shape (2, )
        b (np.ndarray): The 2nd point (x,y) in shape (2, )

    Returns:
        np.ndarray: The 3rd point.
    """
    direction = a - b
    c = b + np.r_[-direction[1], direction[0]]
    return c

def get_warp_matrix(center: np.ndarray,
                    scale: np.ndarray,
                    rot: float,
                    output_size: Tuple[int, int],
                    shift: Tuple[float, float] = (0., 0.),
                    inv: bool = False) -> np.ndarray:
    """Calculate the affine transformation matrix that can warp the bbox area
    in the input image to the output size.

    Args:
        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        rot (float): Rotation angle (degree).
        output_size (np.ndarray[2, ] | list(2,)): Size of the
            destination heatmaps.
        shift (0-100%): Shift translation ratio wrt the width/height.
            Default (0., 0.).
        inv (bool): Option to inverse the affine transform direction.
            (inv=False: src->dst or inv=True: dst->src)

    Returns:
        np.ndarray: A 2x3 transformation matrix
    """
    assert len(center) == 2
    assert len(scale) == 2
    assert len(output_size) == 2
    assert len(shift) == 2

    shift = np.array(shift)
    src_w = scale[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.deg2rad(rot)
    src_dir = _rotate_point(np.array([0., src_w * -0.5]), rot_rad)
    dst_dir = np.array([0., dst_w * -0.5])

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale * shift
    src[1, :] = center + src_dir + scale * shift
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        warp_mat = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        warp_mat = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    return warp_mat


class TestPipeline:
    def __init__(self,
                 input_size:Tuple[int, int],
                 device: str = "cpu",
                 size_factor:int=32,
                 resize_mode: str="expand"
                  ) -> None:
        self.device = device
        self.input_size = input_size
        self.resize_mode = resize_mode
        self.size_factor = size_factor
        # normalize
        self.mean=torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        self.std=torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
        self.pad_size_divisor = 1
        self.pad_value = 0

    def image_to_tensor(self,img: Union[np.ndarray,
                               Sequence[np.ndarray]]) -> torch.torch.Tensor:


        if isinstance(img, np.ndarray):
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)

            img = np.ascontiguousarray(img)
            tensor = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        else:
            tensor = torch.stack([self.image_to_tensor(_img) for _img in img])

        return tensor

    def normalize(self,inputs: torch.Tensor):
        inputs = inputs.float()
        inputs = (inputs - self.mean) / self.std
        h, w = inputs.shape[2:]
        target_h = math.ceil(
            h / self.pad_size_divisor) * self.pad_size_divisor
        target_w = math.ceil(
            w / self.pad_size_divisor) * self.pad_size_divisor
        pad_h = target_h - h
        pad_w = target_w - w
        inputs = F.pad(inputs, (0, pad_w, 0, pad_h),
                                'constant', self.pad_value)
        return inputs
        
    
    @staticmethod
    def _ceil_to_multiple(size: Tuple[int, int], base: int):
        """Ceil the given size (tuple of [w, h]) to a multiple of the base."""
        return tuple(int(np.ceil(s / base) * base) for s in size)
    
    def _get_input_size(self, img_size: Tuple[int, int]) -> Tuple:
        """
        Calculate the actual input size and the padded input size based on the given image size and input size.

        Parameters:
            img_size (Tuple[int, int]): The size of the input image as a tuple of width and height.

        Returns:
            Tuple: A tuple containing the actual input size and the padded input size.
        """
        img_w, img_h = img_size
        ratio = img_w / img_h

        if self.resize_mode == 'fit':
            padded_input_size = self._ceil_to_multiple(self.input_size,
                                                       self.size_factor)
            if padded_input_size != self.input_size:
                raise ValueError(
                    'When ``resize_mode==\'fit\', the input size (height and'
                    ' width) should be mulitples of the size_factor('
                    f'{self.size_factor}) at all scales. Got invalid input '
                    f'size {self.input_size}.')

            pad_w, pad_h = padded_input_size
            rsz_w = min(pad_w, pad_h * ratio)
            rsz_h = min(pad_h, pad_w / ratio)
            actual_input_size = (rsz_w, rsz_h)

        elif self.resize_mode == 'expand':
            _padded_input_size = self._ceil_to_multiple(
                self.input_size, self.size_factor)
            pad_w, pad_h = _padded_input_size
            rsz_w = max(pad_w, pad_h * ratio)
            rsz_h = max(pad_h, pad_w / ratio)

            actual_input_size = (rsz_w, rsz_h)
            padded_input_size = self._ceil_to_multiple(actual_input_size,
                                                       self.size_factor)

        else:
            raise ValueError(f'Invalid resize mode {self.resize_mode}')

        return actual_input_size, padded_input_size
    
    def __call__(self, image: Union[np.ndarray, str],) -> Any:
        if  isinstance(image,str):
            img = cv2.imread(image)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = image
        h,w,_ = img.shape
        
        actual_input_size, padded_input_size = self._get_input_size((w,h))
        center = np.array([w / 2, h / 2], dtype=np.float32)
        scale = np.array([
                    w * padded_input_size[0] / actual_input_size[0],
                    h * padded_input_size[1] / actual_input_size[1]
                ],
                                 dtype=np.float32)
        warp_mat = get_warp_matrix(
                    center=center,
                    scale=scale,
                    rot=0,
                    output_size=padded_input_size)
        img = cv2.warpAffine(
                img, warp_mat, padded_input_size, flags=cv2.INTER_LINEAR)
        vis = img.copy()
        img = self.image_to_tensor(img).unsqueeze(dim=0)
        inputs = self.normalize(img).contiguous().to(self.device)

        return inputs,vis

