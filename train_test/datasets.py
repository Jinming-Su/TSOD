import os
import os.path

import torch.utils.data as data
from PIL import Image


def make_dataset(root):
    img_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, 'image')) if f.endswith('.png')]
    return [(os.path.join(root, 'image/' + img_name + '.png'), os.path.join(root, 'groundtruth/' + img_name + '.png'), \
            os.path.join(root, 'groundtruth_boundary/' + img_name + '.png'), os.path.join(root, 'groundtruth_interior/' + img_name + '.png')\
             ) for img_name in img_list]


class ImageFolder(data.Dataset):
    # image and gt should be in the same folder and have same filename except extended name (jpg and png respectively)
    def __init__(self, root, joint_transform=None, transform=None, target_transform=None):
        self.root = root
        self.imgs = make_dataset(root)
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img_path, gt_path, gt_b_path, gt_i_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        target = Image.open(gt_path).convert('L')
        target_b = Image.open(gt_b_path).convert('L')
        target_i = Image.open(gt_i_path).convert('L')

        if self.joint_transform is not None:
            img, target, target_b, target_i = self.joint_transform(img, target, target_b, target_i)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
            target_b = self.target_transform(target_b)
            target_i = self.target_transform(target_i)

        return img, target, target_b, target_i

    def __len__(self):
        return len(self.imgs)
