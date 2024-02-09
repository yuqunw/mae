# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import torch
from torch.utils.data import Dataset
from pathlib import Path
from colmap import qvec2rotmat
import json
import numpy as np

import torch.nn.functional as NF
import torchvision.transforms.functional as F
import cv2
from PIL import Image

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)

    print(dataset)

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


class Toybox5(Dataset):
    def __init__(self, scene_root, split, region_num=200, resample_rate=2, transform=None):
        super().__init__()
        self.scene_root = Path(scene_root)
        self.region_num = region_num
        self.split = split
        self.resample_rate = resample_rate

        # Load data
        self.meta = self.load_meta()
        self.K, self.poses, self.h, self.w = self.load_camera() # 3x3, nx4x4 (camera to world)
        self.split_list = self.load_split()
        self.images_path, self.depths_path, self.segments_path, self.features_path = self.load_paths()
        self.rays, self.ray_scales = self.load_rays(self.h, self.w)

        # Cached
        self.images = self.load_images()
        self.depths = self.load_depths()
        self.origins, self.dirs, self.moments = self.load_ray_embedding()

    def load_ray_embedding(self):
        # (H, W, 1, 1, 3) @ (1, 1, N, 3, 3).T = (H, W, N, 1, 3) -> (H, W, N, 3)
        ray_d = (self.rays[:, :, None, None] @ self.poses[:, :3, :3].permute(0, 2, 1)[None][None])[..., 0, :]
        ray_o = self.poses[:, :3, 3].expand(ray_d.shape) # (N, 3) -> (H, W, N, 3)
        ray_moment = torch.cross(ray_o, ray_d, dim=-1) # (H, W, N, 3)
        return ray_o, ray_d, ray_moment

    def load_depths(self):
        depths = []
        for depth_file in self.depths_path:
            depth = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED)
            depth = torch.from_numpy(depth).unsqueeze(0).float()
            depths.append(depth)
        depths = torch.stack(depths, dim=0)
        return depths

    def load_images(self):
        images = []
        for image_file in self.images_path:
            image = Image.open(image_file)
            image = F.to_tensor(image)
            if image.shape[0] == 4:
                image = image[:3]
            images.append(image)
        images = torch.stack(images, dim=0)
        return images

    def load_split(self):
        split = sorted(self.meta['split_ids'][self.split])
        return split

    def load_meta(self):
        meta_file = self.scene_root / 'metadata.json'
        with open(meta_file, 'r') as f:
            meta = json.load(f)
        return meta

    def load_camera(self):
        poses_list = []
        cameras = self.meta['camera']

        for qvec, tvec in zip(cameras['quaternions'], cameras['positions']):
            pose = np.ones((4, 4))
            pose[:3, :3] = qvec2rotmat(qvec)
            pose[:3, 3] = tvec

            # Transform from opengl to opencv
            pose[:, 1] *= -1
            pose[:, 2] *= -1
            poses_list.append(pose)

        poses = np.stack(poses_list, axis=0)

        focal_length = cameras['focal_length']
        h, w = cameras['height'], cameras['width']
        K = np.eye(3)
        K[0, 0] = focal_length
        K[1, 1] = focal_length
        K[0, 2] = w / 2
        K[1, 2] = h / 2
        K = torch.from_numpy(K).float()
        poses = torch.from_numpy(poses).float()

        return K, poses, h, w

    def load_rays(self, h, w, O=0.5):
        O = 0.5
        x_coords = torch.linspace(O, w - 1 + O, w)
        y_coords = torch.linspace(O, h - 1 + O, h)

        # HxW grids
        y_grid_coords, x_grid_coords = torch.meshgrid([y_coords, x_coords])

        # HxWx3
        h_coords = torch.stack([x_grid_coords, y_grid_coords, torch.ones_like(x_grid_coords)], -1)
        rays = h_coords @ self.K.inverse().T
        ray_scale = rays.norm(p=2, dim=-1)

        return NF.normalize(rays, p=2, dim=-1), ray_scale

    def load_paths(self):
        images_path = [str(self.scene_root / 'images' / f'rgba_{i:05d}.png') for i in self.split_list]
        depths_path = [str(self.scene_root / 'depths' / f'depth_{i:05d}.diff') for i in self.split_list]
        segments_path = [str(self.scene_root / 'segments' / f'seg_{i:05d}.png') for i in self.split_list]
        features_path = [str(self.scene_root / 'features' / f'feature_{i:05d}.pth') for i in self.split_list]

        return images_path, depths_path, segments_path, features_path
    
    def __getitem__(self, index):
        sample = {}

        sample['image'] = self.images[index]
        sample['depth'] = self.depths[index]
        sample['origin'] = self.origins
        sample['dir'] = self.dirs
        sample['moment'] = self.moments

        return sample

    def __len__(self):
        return len(self.images_path)