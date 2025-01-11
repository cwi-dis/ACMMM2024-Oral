import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
import random
from torchvision import transforms
from torch.utils import data
from PIL import Image
import cv2
import itertools
from torchvision.transforms import InterpolationMode


class MMDataset(data.Dataset):
    """Read data from the original dataset for feature extraction"""

    def __init__(
        self,
        data_dir_texture,
        data_dir_depth,
        data_dir_normal,
        data_dir_texture_pc,
        data_dir_position_pc,
        data_dir_normal_pc,
        datainfo_path,
        transform,
        crop_size=224,
        img_length_read=6,
        patch_length_read=6,
        npoint=2048,
        is_train=True,
    ):
        super(MMDataset, self).__init__()
        dataInfo = pd.read_csv(
            datainfo_path, header=0, sep=",", index_col=False, encoding="utf-8-sig"
        )
        self.ply_name = dataInfo[["name"]]
        self.ply_mos = dataInfo["mos"]
        self.ply_disTypes = dataInfo["DT"]
        self.crop_size = crop_size
        self.data_dir_texture = data_dir_texture
        self.data_dir_depth = data_dir_depth
        self.data_dir_normal = data_dir_normal
        self.transform = transform
        self.img_length_read = img_length_read
        self.patch_length_read = patch_length_read
        self.npoint = npoint
        self.data_dir_texture_pc = data_dir_texture_pc
        self.data_dir_position_pc = data_dir_position_pc
        self.data_dir_normal_pc = data_dir_normal_pc
        self.length = len(self.ply_name)
        self.is_train = is_train

    def __len__(self):
        return self.length

    def random_crop(self, img, depth, normal):
        # before random crop, make sure the img size is larger than crop size
        # img_width = img.size[1]
        # print("img width is:", img_width)
        # img_height = img.size[0]
        # print("img height is:", img_height)
        if img.size[1] < self.crop_size or img.size[0] < self.crop_size:
            # print("img size is smaller than crop size:", imge_name)
            temp_cropsize = min(img.size[0], img.size[1])
            # random crop
            i, j, h, w = transforms.RandomCrop.get_params(
                img, output_size=(temp_cropsize, temp_cropsize)
            )
            img = transforms.functional.crop(img, i, j, h, w)
            depth = transforms.functional.crop(depth, i, j, h, w)
            normal = transforms.functional.crop(normal, i, j, h, w)
            # self.crop_size = 224
            img = transforms.functional.resize(
                img, (self.crop_size, self.crop_size)
            )  # xm: resize to crop_size
            depth = transforms.functional.resize(
                depth, (self.crop_size, self.crop_size)
            )
            normal = transforms.functional.resize(
                normal, (self.crop_size, self.crop_size)
            )
            return img, depth, normal
        else:
            # print("img size is larger than crop size:", imge_name)
            # random crop
            i, j, h, w = transforms.RandomCrop.get_params(
                img, output_size=(self.crop_size, self.crop_size)
            )
            img = transforms.functional.crop(img, i, j, h, w)
            depth = transforms.functional.crop(depth, i, j, h, w)
            normal = transforms.functional.crop(normal, i, j, h, w)

            return img, depth, normal

    def set_rand_seed(seed=1998):
        print("Random Seed: ", seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def __getitem__(self, idx):
        img_name = self.ply_name.iloc[idx, 0]
        texture_dir = os.path.join(self.data_dir_texture, img_name)
        depth_dir = os.path.join(self.data_dir_depth, img_name)
        normal_dir = os.path.join(self.data_dir_normal, img_name)

        img_channel = 3
        img_height_crop = self.crop_size
        img_width_crop = self.crop_size

        img_length_read = self.img_length_read
        transformed_img = torch.zeros(
            [img_length_read, img_channel, img_height_crop, img_width_crop]
        )
        transformed_depth = torch.zeros(
            [img_length_read, img_channel, img_height_crop, img_width_crop]
        )
        transformed_normal = torch.zeros(
            [img_length_read, img_channel, img_height_crop, img_width_crop]
        )
        # read images : only texture
        img_read_index = 0
        for i in range(img_length_read):
            # load images
            imge_name = os.path.join(texture_dir, str(i) + ".png")
            normal_name = os.path.join(normal_dir, str(i) + ".png")
            depth_name = os.path.join(depth_dir, str(i) + ".png")
            assert os.path.exists(imge_name), f"{imge_name}, Image do not exist!"
            assert os.path.exists(normal_name), f"{normal_name}, Normal do not exist!"
            assert os.path.exists(depth_name), f"{depth_name}, Depth do not exist!"

            read_frame = Image.open(imge_name)
            read_depth = Image.open(depth_name)
            read_normal = Image.open(normal_name)

            read_frame = read_frame.convert("RGB")
            read_depth = read_depth.convert("RGB")
            read_normal = read_normal.convert("RGB")
            try:
                read_frame, read_depth, read_normal = self.random_crop(
                    read_frame, read_depth, read_normal
                )
            except:
                raise
            read_frame = self.transform(read_frame)
            read_depth = self.transform(read_depth)
            read_normal = self.transform(read_normal)

            transformed_img[i] = read_frame
            transformed_depth[i] = read_depth
            transformed_normal[i] = read_normal

            img_read_index += 1

        if img_read_index < img_length_read:
            for j in range(img_read_index, img_length_read):
                transformed_img[j] = transformed_img[img_read_index - 1]
                transformed_depth[j] = transformed_depth[img_read_index - 1]
                transformed_normal[j] = transformed_normal[img_read_index - 1]

        # read pc
        patch_length_read = self.patch_length_read
        npoint = self.npoint
        selected_patches_texture = torch.zeros([patch_length_read, 6, npoint])
        selected_patches_normal = torch.zeros([patch_length_read, 6, npoint])
        selected_patches_position = torch.zeros([patch_length_read, 6, npoint])
        texture_pc_path = os.path.join(
            self.data_dir_texture_pc, self.ply_name.iloc[idx, 0].split(".")[0] + ".npy"
        )
        texture_points = list(np.load(texture_pc_path))

        normal_pc_path = os.path.join(
            self.data_dir_normal_pc, self.ply_name.iloc[idx, 0].split(".")[0] + ".npy"
        )
        normal_points = list(np.load(normal_pc_path))

        position_pc_path = os.path.join(
            self.data_dir_position_pc, self.ply_name.iloc[idx, 0].split(".")[0] + ".npy"
        )
        position_points = list(np.load(position_pc_path))

        # randomly select patches during the training stage
        if self.is_train:
            sampled_index = random.sample(range(len(texture_points)), patch_length_read)
            assert len(texture_points) == len(
                normal_points
            ), "texture length should equal to normal length"
            assert len(texture_points) == len(
                position_points
            ), "texture length should equal to position length"
            assert len(normal_points) == len(
                position_points
            ), "normal length should equal to position length"

            for i in range(patch_length_read):
                selected_patches_texture[i] = torch.from_numpy(
                    texture_points[sampled_index[i]]
                ).transpose(0, 1)
                selected_patches_normal[i] = torch.from_numpy(
                    normal_points[sampled_index[i]]
                ).transpose(0, 1)
                selected_patches_position[i] = torch.cat(
                    [
                        torch.from_numpy(position_points[sampled_index[i]]),
                        torch.zeros(2048, 3),
                    ],
                    dim=1,
                ).transpose(0, 1)

        else:
            if len(texture_points) < patch_length_read:
                indices = itertools.cycle(range(len(texture_points)))
                for i in range(patch_length_read):
                    index = next(indices)
                    selected_patches_texture[i] = torch.from_numpy(
                        texture_points[index]
                    ).transpose(0, 1)
                    selected_patches_normal[i] = torch.from_numpy(
                        normal_points[index]
                    ).transpose(0, 1)
                    selected_patches_position[i] = torch.cat(
                        [
                            torch.from_numpy(position_points[index]),
                            torch.zeros(2048, 3),
                        ],
                        dim=1,
                    ).transpose(0, 1)
            else:
                for i in range(patch_length_read):
                    selected_patches_texture[i] = torch.from_numpy(
                        texture_points[i]
                    ).transpose(0, 1)
                    selected_patches_normal[i] = torch.from_numpy(
                        normal_points[i]
                    ).transpose(0, 1)
                    selected_patches_position[i] = torch.cat(
                        [
                            torch.from_numpy(position_points[i]),
                            torch.zeros(2048, 3),
                        ],
                        dim=1,
                    ).transpose(0, 1)

        # read gt
        y_mos = self.ply_mos.iloc[idx]
        y_label = torch.FloatTensor(np.array(y_mos))

        disType = self.ply_disTypes.iloc[idx]
        dis_label = torch.tensor(np.array(disType))

        return (
            img_name,
            transformed_img,
            transformed_depth,
            transformed_normal,
            selected_patches_texture,
            selected_patches_normal,
            selected_patches_position,
            y_label,
            dis_label,
        )
