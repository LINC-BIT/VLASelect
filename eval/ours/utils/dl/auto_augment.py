import numpy as np

from PIL import Image, ImageEnhance, ImageOps
import random


class ShearX(object):
    def __init__(self, fillcolor=(128, 128, 128)):
        self.fillcolor = fillcolor

    def __call__(self, x, magnitude):
        return x.transform(
            x.size, Image.AFFINE, (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
            Image.BICUBIC, fillcolor=self.fillcolor)


class ShearY(object):
    def __init__(self, fillcolor=(128, 128, 128)):
        self.fillcolor = fillcolor

    def __call__(self, x, magnitude):
        return x.transform(
            x.size, Image.AFFINE, (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
            Image.BICUBIC, fillcolor=self.fillcolor)


class TranslateX(object):
    def __init__(self, fillcolor=(128, 128, 128)):
        self.fillcolor = fillcolor

    def __call__(self, x, magnitude):
        return x.transform(
            x.size, Image.AFFINE, (1, 0, magnitude * x.size[0] * random.choice([-1, 1]), 0, 1, 0),
            fillcolor=self.fillcolor)


class TranslateY(object):
    def __init__(self, fillcolor=(128, 128, 128)):
        self.fillcolor = fillcolor

    def __call__(self, x, magnitude):
        return x.transform(
            x.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * x.size[1] * random.choice([-1, 1])),
            fillcolor=self.fillcolor)


class Rotate(object):
    # from https://stackoverflow.com/questions/
    # 5252170/specify-image-filling-color-when-rotating-in-python-with-pil-and-setting-expand
    def __call__(self, x, magnitude):
        rot = x.convert("RGBA").rotate(magnitude * random.choice([-1, 1]))
        return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(x.mode)


class Color(object):
    def __call__(self, x, magnitude):
        return ImageEnhance.Color(x).enhance(1 + magnitude * random.choice([-1, 1]))
    
    
class Grey(object):
    def __call__(self, x, magnitude):
        return ImageEnhance.Color(x).enhance(max(0.001, 1 - magnitude * 5)) # if mag > 0.2 the image will be grey-scale


class Posterize(object):
    def __call__(self, x, magnitude):
        return ImageOps.posterize(x, magnitude)


class Solarize(object):
    def __call__(self, x, magnitude):
        return ImageOps.solarize(x, magnitude)


class Contrast(object):
    def __call__(self, x, magnitude):
        return ImageEnhance.Contrast(x).enhance(1 + magnitude * random.choice([-1, 1]))


class Sharpness(object):
    def __call__(self, x, magnitude):
        return ImageEnhance.Sharpness(x).enhance(1 + magnitude * random.choice([-1, 1]))


class Brightness(object):
    def __call__(self, x, magnitude):
        return ImageEnhance.Brightness(x).enhance(1 + magnitude * random.choice([-1, 1]))


class AutoContrast(object):
    def __call__(self, x, magnitude):
        return ImageOps.autocontrast(x)


class Equalize(object):
    def __call__(self, x, magnitude):
        return ImageOps.equalize(x)


class Invert(object):
    def __call__(self, x, magnitude):
        return ImageOps.invert(x)


policy_names = list({
    "shearX": np.linspace(0, 0.3, 10),
    "shearY": np.linspace(0, 0.3, 10),
    "translateX": np.linspace(0, 150 / 331, 10),
    "translateY": np.linspace(0, 150 / 331, 10),
    "rotate": np.linspace(0, 30, 10),
    "color": np.linspace(0.0, 0.9, 10),
    "grey": np.linspace(0.0, 0.9, 10),
    "posterize": np.round(np.linspace(8, 4, 10), 0).astype(int),
    "solarize": np.linspace(256, 0, 10),
    "contrast": np.linspace(0.0, 0.9, 10),
    "sharpness": np.linspace(0.0, 0.9, 10),
    "brightness": np.linspace(0.0, 0.9, 10),
    "autocontrast": [0] * 10,
    "equalize": [0] * 10,
    "invert": [0] * 10
}.keys())

class SimTargetDomainPolicy:
    def __init__(self, funcs, magnitudes, fillcolor=(128, 128, 128)):

        self.ranges = {
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 150 / 331, 10),
            "translateY": np.linspace(0, 150 / 331, 10),
            "rotate": np.linspace(0, 30, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "grey": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(int),
            "solarize": np.linspace(256, 0, 10),
            "contrast": np.linspace(0.0, 0.9, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.9, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10
        }

        self.funcs = {
            "shearX": ShearX(fillcolor=fillcolor),
            "shearY": ShearY(fillcolor=fillcolor),
            "translateX": TranslateX(fillcolor=fillcolor),
            "translateY": TranslateY(fillcolor=fillcolor),
            "rotate": Rotate(),
            "color": Color(),
            "grey": Grey(),
            "posterize": Posterize(),
            "solarize": Solarize(),
            "contrast": Contrast(),
            "sharpness": Sharpness(),
            "brightness": Brightness(),
            "autocontrast": AutoContrast(),
            "equalize": Equalize(),
            "invert": Invert()
        }
        
        assert len(magnitudes) == len(funcs)
        for i in magnitudes:
            assert isinstance(i, int)
            assert i >= 0 and i < 10
        
        self.magnitudes = magnitudes # 0~10
        self.funcs_name = funcs
        self.funcs = {k: v for k, v in self.funcs.items() if k in funcs}

    def __call__(self, img):
        for (op_name, op), mag in zip(self.funcs.items(), self.magnitudes):
            img = op(img, self.ranges[op_name][mag])
        return img
    
    def __repr__(self) -> str:
        return f'SimTargetDomainPolicy(funcs={list(self.funcs.keys())}, magnitudes={self.magnitudes})'
    
    def to_json(self):
        return {
            'funcs': self.funcs_name,
            'magnitudes': self.magnitudes
        }
    
    
only_color_policy_names = list({
    # "shearX": np.linspace(0, 0.3, 10),
    # "shearY": np.linspace(0, 0.3, 10),
    # "translateX": np.linspace(0, 150 / 331, 10),
    # "translateY": np.linspace(0, 150 / 331, 10),
    # "rotate": np.linspace(0, 30, 10),
    "color": np.linspace(0.0, 0.9, 10),
    # "grey": np.linspace(0.0, 0.9, 10),
    "posterize": np.round(np.linspace(8, 4, 10), 0).astype(int),
    "solarize": np.linspace(256, 0, 10),
    "contrast": np.linspace(0.0, 0.9, 10),
    "sharpness": np.linspace(0.0, 0.9, 10),
    "brightness": np.linspace(0.0, 0.9, 10),
    "autocontrast": [0] * 10,
    "equalize": [0] * 10,
    "invert": [0] * 10
}.keys())

class SimTargetDomainOnlyColorPolicy:
    def __init__(self, funcs, magnitudes, fillcolor=(128, 128, 128)):

        self.ranges = {
            # "shearX": np.linspace(0, 0.3, 10),
            # "shearY": np.linspace(0, 0.3, 10),
            # "translateX": np.linspace(0, 150 / 331, 10),
            # "translateY": np.linspace(0, 150 / 331, 10),
            # "rotate": np.linspace(0, 30, 10),
            "color": np.linspace(0.0, 0.9, 10),
            # "grey": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(int),
            "solarize": np.linspace(256, 0, 10),
            "contrast": np.linspace(0.0, 0.9, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.9, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10
        }

        self.funcs = {
            # "shearX": ShearX(fillcolor=fillcolor),
            # "shearY": ShearY(fillcolor=fillcolor),
            # "translateX": TranslateX(fillcolor=fillcolor),
            # "translateY": TranslateY(fillcolor=fillcolor),
            # "rotate": Rotate(),
            "color": Color(),
            # "grey": Grey(),
            "posterize": Posterize(),
            "solarize": Solarize(),
            "contrast": Contrast(),
            "sharpness": Sharpness(),
            "brightness": Brightness(),
            "autocontrast": AutoContrast(),
            "equalize": Equalize(),
            "invert": Invert()
        }
        
        assert len(magnitudes) == len(funcs)
        for i in magnitudes:
            assert isinstance(i, int)
            assert i >= 0 and i < 10
        
        self.magnitudes = magnitudes # 0~10
        self.funcs_name = funcs
        self.funcs = {k: v for k, v in self.funcs.items() if k in funcs}

    def __call__(self, img):
        for (op_name, op), mag in zip(self.funcs.items(), self.magnitudes):
            img = op(img, self.ranges[op_name][mag])
        return img
    
    def __repr__(self) -> str:
        return f'SimTargetDomainOnlyColorPolicy(funcs={list(self.funcs.keys())}, magnitudes={self.magnitudes})'
    
    def to_json(self):
        return {
            'funcs': self.funcs_name,
            'magnitudes': self.magnitudes
        }
    

import math
def gen_random_sim_policy(target_shift_mag, only_color_policy=False):
    # choose random number of funcs with random mag
    
    # print(num_funcs)
    # num_funcs = len(policy_names)
    # target_sum_mags = random.randint(1, num_funcs * 10)
    _policy_names = policy_names if not only_color_policy else only_color_policy_names
    
    if target_shift_mag is None:
        num_funcs = random.randint(1, int(len(_policy_names) * 1.0))
        num_funcs = min(num_funcs, len(_policy_names))
        rand_mags = np.random.randint(low=1, high=10, size=num_funcs).tolist()
    else:
        num_funcs = len(_policy_names) // 2 if not only_color_policy else len(_policy_names) // 3 * 2
        rand_mags = np.full(num_funcs, max(1, target_shift_mag // num_funcs), dtype=int).tolist()
    
    # make sure sum(rand_mags) == target_sum_mags
    # rand_mags = target_sum_mags / sum(rand_mags) * rand_mags
    # rand_mags = [min(i, 9) for i in rand_mags] # I don't know why but this generates uniform distribution on sum(rand_mags)
    
    return SimTargetDomainPolicy(
        random.sample(policy_names, k=num_funcs),
        rand_mags
    ) if not only_color_policy else SimTargetDomainOnlyColorPolicy(
        random.sample(only_color_policy_names, k=num_funcs),
        rand_mags
    )
    
    
import torchvision.transforms as trn
    
def apply_sim_policy_on_pre_transform(pre_transform, random_policy):
    # NOTE: can't apply geo transform on images of seg / detection task!

    # random_policy = gen_random_sim_policy(target_shift_mag, only_color_transform)

    if pre_transform is None:
        return trn.Compose([
        # *[t for t in pre_transform.transforms if not isinstance(t, trn.Normalize)], # output Tensor
        trn.ToPILImage(),
        random_policy,
        lambda img: np.asarray(img)
        # trn.ToTensor(),
        # trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # *[t for t in pre_transform.transforms if isinstance(t, trn.Normalize)]
    ]), str(random_policy)
    
    return trn.Compose([
        *[t for t in pre_transform.transforms if not isinstance(t, trn.Normalize)], # output Tensor
        trn.ToPILImage(),
        random_policy,
        trn.ToTensor(),
        *[t for t in pre_transform.transforms if isinstance(t, trn.Normalize)]
    ]), str(random_policy)
    

from data import ABDataset, get_dataset
from typing import List


def generate_sim_datasets_with_same_aug(raw_datasets: List[ABDataset], random_sim_policy):
    from .auto_augment import apply_sim_policy_on_pre_transform, gen_random_sim_policy
    
    # target_shift_mag = random.randint(1, 50)
    # random_sim_policy = gen_random_sim_policy(target_shift_mag, False)
    
    res = []
    
    for raw_dataset in raw_datasets:
        sim_transform, _ = apply_sim_policy_on_pre_transform(raw_dataset.transform, random_sim_policy)
        
        if 'COCO' in raw_dataset.name:
            # return _AugWrapperForDataset(raw_dataset, sim_transform), sim_transform_desc
            raise NotImplementedError
            
        dataset = get_dataset(
            raw_dataset.name,
            raw_dataset.root_dir,
            raw_dataset.split,
            sim_transform,
            raw_dataset.ignore_classes,
            raw_dataset.idx_map
        )
        res += [dataset]
    
    return res
