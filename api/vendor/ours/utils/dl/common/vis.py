import os
import torch
import numpy as np
from PIL import Image


def save_tensor_image(
    tensor: torch.Tensor,
    save_path: str,
    normalize: bool = True,
    depth_clip: tuple | None = None,
):
    """
    保存 (C,H,W) 或 (B,C,H,W) 的 tensor 图像到本地

    Args:
        tensor: torch.Tensor
            shape = (C,H,W) or (B,C,H,W)
            C=1 (depth / gray) or C=3 (RGB)
        save_path: str
            保存路径，若为 batch，会自动加 index
        normalize: bool
            是否对 float tensor 做归一化到 [0,255]
        depth_clip: (min, max) or None
            仅对 C=1 时有效，深度裁剪范围
    """

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    print(f'image min: {tensor.min()}, max: {tensor.max()}')

    if tensor.dim() == 3:
        _save_single(tensor, save_path, normalize, depth_clip)
    elif tensor.dim() == 4:
        base, ext = os.path.splitext(save_path)
        for i in range(tensor.shape[0]):
            _save_single(
                tensor[i],
                f"{base}_{i}{ext}",
                normalize,
                depth_clip,
            )
    else:
        raise ValueError(f"Unsupported tensor shape: {tensor.shape}")


def _save_single(
    tensor: torch.Tensor,
    save_path: str,
    normalize: bool,
    depth_clip: tuple | None,
):
    """
    保存单张 (C,H,W)
    """
    assert tensor.dim() == 3, tensor.shape
    C, H, W = tensor.shape

    img = tensor.detach().cpu()

    if img.dtype != torch.uint8:
        img = img.float()

        if C == 1 and depth_clip is not None:
            img = torch.clamp(img, depth_clip[0], depth_clip[1])

        if normalize:
            min_val = img.min()
            max_val = img.max()
            img = (img - min_val) / (max_val - min_val + 1e-6)
            img = img * 255.0

        img = img.to(torch.uint8)

    img = img.numpy()

    if C == 1:
        img = img.squeeze(0)  # (H,W)
        pil_img = Image.fromarray(img, mode="L")
    elif C == 3:
        img = img.transpose(1, 2, 0)  # (H,W,C)
        pil_img = Image.fromarray(img, mode="RGB")
    else:
        raise ValueError(f"Unsupported channel number: {C}")

    pil_img.save(save_path)