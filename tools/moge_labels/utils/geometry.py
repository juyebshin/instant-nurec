# closed_form_inverse_se3 copied unchanged from DVGT dvgt/utils/geometry.py
from typing import Union

import numpy as np
import torch


def closed_form_inverse_se3(se3: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """
    以闭式解计算一批SE(3)矩阵的逆。
    这个函数支持任意批次维度 (...) 以及单个矩阵。

    Args:
        se3: SE(3) 变换矩阵，形状为 (..., 4, 4) 或 (..., 3, 4)。

    Returns:
        逆变换矩阵，与输入 se3 的类型、设备和批次维度相同，形状为 (..., 4, 4)。
        
    公式:
        如果 T_mat = [[R, t], [0, 1]]
        则 T_mat_inv = [[R.T, -R.T @ t], [0, 1]]
    """
    # 检查输入是 NumPy 数组还是 PyTorch 张量
    is_numpy = isinstance(se3, np.ndarray)
    
    # 为单个矩阵 (ndim=2) 临时增加一个批次维度，以便统一处理
    was_single = se3.ndim == 2
    if was_single:
        se3 = se3[None, ...]  # works for both numpy and torch

    # 验证最后两个维度
    if se3.shape[-2:] not in [(4, 4), (3, 4)]:
        raise ValueError(f"输入矩阵的最后两个维度必须是 (4, 4) 或 (3, 4)，但得到的是 {se3.shape}")

    R = se3[..., :3, :3]
    t = se3[..., :3, 3:4]

    if is_numpy:
        R_transposed = np.swapaxes(R, -2, -1)
    else:
        R_transposed = R.transpose(-2, -1)
    
    t_inverted = -R_transposed @ t

    batch_shape = se3.shape[:-2]
    
    if is_numpy:
        inverted_matrix = np.zeros((*batch_shape, 4, 4), dtype=se3.dtype)
    else:
        inverted_matrix = torch.zeros((*batch_shape, 4, 4), device=se3.device, dtype=se3.dtype)
    
    inverted_matrix[..., :3, :3] = R_transposed
    inverted_matrix[..., :3, 3:4] = t_inverted
    inverted_matrix[..., 3, 3] = 1.0

    # 5. 如果输入是单个矩阵，移除之前添加的批次维度
    if was_single:
        inverted_matrix = inverted_matrix[0]

    return inverted_matrix
