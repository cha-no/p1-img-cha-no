import os
import glob
import random
from collections import defaultdict

import numpy as np
import torch
import torch.utils.data as data
import cv2
from PIL import Image
from torch.utils.data import Subset
from torchvision import transforms
from torchvision.transforms import *

import albumentations as A
from albumentations.pytorch import ToTensorV2

IMG_EXTENSIONS = [
    ".jpg", ".JPG", ".jpeg", ".JPEG", ".png",
    ".PNG", ".ppm", ".PPM", ".bmp", ".BMP",
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


class BaseAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = transforms.Compose([
            Resize(resize, Image.BILINEAR),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])

    def __call__(self, image):
        return self.transform(image)


class AddGaussianNoise(object):
    """
        transform 에 없는 기능들은 이런식으로 __init__, __call__, __repr__ 부분을
        직접 구현하여 사용할 수 있습니다.
    """
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class CustomAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = transforms.Compose([
            CenterCrop((320, 256)),
            Resize(resize, Image.BILINEAR),
            ColorJitter(0.1, 0.1, 0.1, 0.1),
            ToTensor(),
            Normalize(mean=mean, std=std),
            AddGaussianNoise()
        ])

    def __call__(self, image):
        return self.transform(image)

class BasicAugmentation(object):
    def __init__(self, resize, mean, std, **args):
        self.transform = A.Compose([
            A.Resize(resize, resize, p=1.0),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(p = 1.0),
        ])
    def __call__(self, image):
        return self.transform(image = image)['image']

class Augmentation1(object):
    def __init__(self, resize, mean, std, **args):
        self.transform = A.Compose([
            A.Resize(resize, resize, p=1.0),
            A.HorizontalFlip(p = 0.5),
            A.ShiftScaleRotate(p = 0.5),
            A.IAAAffine(scale = [0.8, 1.2], rotate = [-10, 10], shear = 10, p = 0.5),
            A.OneOf([
                A.RandomBrightness(limit = 0.2, p = 0.5),
                A.RandomBrightnessContrast(brightness_limit = 0.2, contrast_limit = 0.1, p = 0.5),
            ], p = 0.5),
            A.Normalize(mean = mean, std = std, max_pixel_value = 255.0, p = 1.0),
            ToTensorV2(p = 1.0),
        ])

    def __call__(self, image):
        return self.transform(image = image)['image']

class Augmentation2(object):
    def __init__(self, resize, mean, std, **args):
        self.transform = A.Compose([
            A.CenterCrop(384, 320, p = 0.5),
            A.Resize(resize, resize, p=1.0),
            A.Normalize(mean = mean, std = std, max_pixel_value = 255.0, p = 1.0),
            ToTensorV2(p = 1.0),
        ])

    def __call__(self, image):
        return self.transform(image = image)['image']


class Augmentation3(object):
    def __init__(self, resize, mean, std, **args):
        self.transform = A.Compose([
            A.Resize(resize, resize, p=1.0),
            A.ShiftScaleRotate(p = 0.5),
            A.IAAAffine(scale = [0.8, 1.2], rotate = [-10, 10], shear = 10, p = 0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), mean=0, always_apply=False, p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), sigma_limit=0, always_apply=False, p=0.5),
            A.OneOf([
                A.MedianBlur(blur_limit=7, always_apply=False, p=1.0),
                A.MotionBlur(blur_limit=7, always_apply=False, p=1.0),
            ], p = 0.5),
            A.OneOf([
                A.GlassBlur(sigma=0.7, max_delta=4, iterations=2, always_apply=False, mode='fast', p=1.0),
                A.Blur(blur_limit=7, always_apply=False, p=1.0),
            ], p = 0.5),
            A.Normalize(mean = mean, std = std, max_pixel_value = 255.0, p = 1.0),
            ToTensorV2(p = 1.0),
        ])

    def __call__(self, image):
        return self.transform(image = image)['image']

class Augmentation4(object):
    def __init__(self, resize, mean, std, **args):
        self.transform = A.Compose([
            A.Resize(256, 256, p=1.0),
            A.Compose([
                A.HorizontalFlip(p = 0.5),
                A.ShiftScaleRotate(p = 0.5),
                A.IAAAffine(scale = [0.8, 1.2], rotate = [-10, 10], shear = 10, p = 0.5),
                A.OneOf([
                    A.RandomBrightness(limit = 0.2, p = 0.5),
                    A.RandomBrightnessContrast(brightness_limit = 0.2, contrast_limit = 0.1, p = 0.5),
                ]),
            ], p = 0.5),
            A.Normalize(mean = mean, std = std, max_pixel_value = 255.0, p = 1.0),
            ToTensorV2(p = 1.0),
        ])

    def __call__(self, image):
        return self.transform(image = image)['image']

class Augmentation5(object):
    def __init__(self, resize, mean, std, **args):
        self.transform = A.Compose([
            A.CenterCrop(350, 300, p = 1.0),
            A.Resize(resize, resize, p=1.0),
            A.Normalize(mean = mean, std = std, max_pixel_value = 255.0, p = 1.0),
            ToTensorV2(p = 1.0),
        ])

    def __call__(self, image):
        return self.transform(image = image)['image']

class Augmentation6(object):
    def __init__(self, resize, mean, std, **args):
        self.transform = A.Compose([
            A.CenterCrop(350, 300, p = 1.0),
            A.Resize(256, 256, p=1.0),
            A.Compose([
                A.HorizontalFlip(p = 0.5),
                A.ShiftScaleRotate(p = 0.5),
                A.IAAAffine(scale = [0.8, 1.2], rotate = [-10, 10], shear = 10, p = 0.5),
                A.OneOf([
                    A.RandomBrightness(limit = 0.2, p = 0.5),
                    A.RandomBrightnessContrast(brightness_limit = 0.2, contrast_limit = 0.1, p = 0.5),
                ]),
            ], p = 0.5),
            A.Normalize(mean = mean, std = std, max_pixel_value = 255.0, p = 1.0),
            ToTensorV2(p = 1.0),
        ])

    def __call__(self, image):
        return self.transform(image = image)['image']

class MaskBaseDataset(data.Dataset):
    num_classes = 3 * 2 * 3

    class MaskLabels:
        mask = 0
        incorrect = 1
        normal = 2

    class GenderLabels:
        male = 0
        female = 1

    class AgeGroup:
        map_label = lambda x: 0 if int(x) < 30 else 1 if int(x) < 58 else 2

    _file_names = {
        "mask1": MaskLabels.mask,
        "mask2": MaskLabels.mask,
        "mask3": MaskLabels.mask,
        "mask4": MaskLabels.mask,
        "mask5": MaskLabels.mask,
        "incorrect_mask": MaskLabels.incorrect,
        "normal": MaskLabels.normal
    }

    image_paths = []
    mask_labels = []
    gender_labels = []
    age_labels = []

    def __init__(self, data_dir, mean=[0.56019358, 0.52410121, 0.501457], std=[0.23318603, 0.24300033, 0.24567522], val_ratio=0.2):
        self.data_dir = data_dir
        self.mean = mean
        self.std = std
        self.val_ratio = val_ratio

        self.transform = None
        self.setup()
        self.calc_statistics()

    def setup(self):
        profiles = os.listdir(self.data_dir)
        for profile in profiles:
            if profile.startswith("."):  # "." 로 시작하는 파일은 무시합니다
                continue

            img_folder = os.path.join(self.data_dir, profile)
            for file_name in os.listdir(img_folder):
                _file_name, ext = os.path.splitext(file_name)
                if _file_name not in self._file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                    continue

                img_path = os.path.join(self.data_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                mask_label = self._file_names[_file_name]

                id, gender, race, age = profile.split("_")
                gender_label = getattr(self.GenderLabels, gender)
                age_label = self.AgeGroup.map_label(age)

                self.image_paths.append(img_path)
                self.mask_labels.append(mask_label)
                self.gender_labels.append(gender_label)
                self.age_labels.append(age_label)

    def calc_statistics(self):
        has_statistics = self.mean is not None and self.std is not None
        if not has_statistics:
            print("[Warning] Calculating statistics... It can takes huge amounts of time depending on your CPU machine :(")
            sums = []
            squared = []
            for image_path in self.image_paths[:3000]:
                image = np.array(Image.open(image_path)).astype(np.int32)
                sums.append(image.mean(axis=(0, 1)))
                squared.append((image ** 2).mean(axis=(0, 1)))

            self.mean = np.mean(sums, axis=0) / 255
            self.std = (np.mean(squared, axis=0) - self.mean ** 2) ** 0.5 / 255

    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, index):
        image = self.read_image(index)
        mask_label = self.get_mask_label(index)
        gender_label = self.get_gender_label(index)
        age_label = self.get_age_label(index)
        multi_class_label = self.encode_multi_class(mask_label, gender_label, age_label)
        if self.transform:
            image = self.transform(image)
        return image, multi_class_label

    def __len__(self):
        return len(self.image_paths)

    def get_mask_label(self, index):
        return self.mask_labels[index]

    def get_gender_label(self, index):
        return self.gender_labels[index]

    def get_age_label(self, index):
        return self.age_labels[index]

    def read_image(self, index):
        image_path = self.image_paths[index]
        image = cv2.imread(image_path)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    @staticmethod
    def encode_multi_class(mask_label, gender_label, age_label):
        return mask_label * 6 + gender_label * 3 + age_label

    @staticmethod
    def decode_multi_class(multi_class_label):
        mask_label = (multi_class_label // 6) % 3
        gender_label = (multi_class_label // 3) % 2
        age_label = multi_class_label % 3
        return mask_label, gender_label, age_label

    @staticmethod
    def denormalize_image(image, mean, std):
        img_cp = image.copy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
        return img_cp

    def split_dataset(self):
        n_val = int(len(self) * self.val_ratio)
        n_train = len(self) - n_val
        train_set, val_set = torch.utils.data.random_split(self, [n_train, n_val])
        return train_set, val_set

class MaskSplitByProfileDataset(MaskBaseDataset):
    """
        train / val 나누는 기준을 이미지에 대해서 random 이 아닌
        사람(profile)을 기준으로 나눕니다.
        구현은 val_ratio 에 맞게 train / val 나누는 것을 이미지 전체가 아닌 사람(profile)에 대해서 진행하여 indexing 을 합니다
        이후 `split_dataset` 에서 index 에 맞게 Subset 으로 dataset 을 분기합니다.
    """
    def __init__(self, data_dir, mean=[0.56019358, 0.52410121, 0.501457], std=[0.23318603, 0.24300033, 0.24567522], val_ratio=0.2):
        self.indices = defaultdict(list)
        super().__init__(data_dir, mean, std, val_ratio)

    @staticmethod
    def _split_profile(profiles, val_ratio):
        length = len(profiles)
        n_val = int(length * val_ratio)
        # val_indices = set(random.choices(range(length), k=n_val))
        val_indices = set(random.sample(range(length), n_val))
        train_indices = set(range(length)) - val_indices
        
        return {
            "train": train_indices,
            "val": val_indices
        }

    def setup(self):
        profiles = os.listdir(self.data_dir)
        profiles = [profile for profile in profiles if not profile.startswith(".")]
        split_profiles = self._split_profile(profiles, self.val_ratio)

        cnt = 0
        for phase, indices in split_profiles.items():
            for _idx in indices:
                profile = profiles[_idx]
                img_folder = os.path.join(self.data_dir, profile)
                for file_name in os.listdir(img_folder):
                    _file_name, ext = os.path.splitext(file_name)
                    if _file_name not in self._file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                        continue

                    img_path = os.path.join(self.data_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                    mask_label = self._file_names[_file_name]

                    id, gender, race, age = profile.split("_")
                    gender_label = getattr(self.GenderLabels, gender)
                    age_label = self.AgeGroup.map_label(age)

                    self.image_paths.append(img_path)
                    self.mask_labels.append(mask_label)
                    self.gender_labels.append(gender_label)
                    self.age_labels.append(age_label)

                    self.indices[phase].append(cnt)
                    cnt += 1

    def split_dataset(self):
        return [Subset(self, indices) for phase, indices in self.indices.items()]

class AgeDataset(MaskSplitByProfileDataset):
    def __init__(self, data_dir, mean=[0.56019358, 0.52410121, 0.501457], std=[0.23318603, 0.24300033, 0.24567522], val_ratio=0.2):
        self.indices = defaultdict(list)
        super().__init__(data_dir, mean, std, val_ratio)

    def __getitem__(self, index):
        image = self.read_image(index)
        age_label = self.get_age_label(index)
        if self.transform:
            image = self.transform(image)
        return image, age_label

class GenderDataset(MaskSplitByProfileDataset):
    def __init__(self, data_dir, mean=[0.56019358, 0.52410121, 0.501457], std=[0.23318603, 0.24300033, 0.24567522], val_ratio=0.2):
        self.indices = defaultdict(list)
        super().__init__(data_dir, mean, std, val_ratio)

    def __getitem__(self, index):
        image = self.read_image(index)
        #mask_label = self.get_mask_label(index)
        gender_label = self.get_gender_label(index)
        #age_label = self.get_age_label(index)
        #multi_class_label = self.encode_multi_class(mask_label, gender_label, age_label)
        if self.transform:
            image = self.transform(image)
        return image, gender_label

class MaskCorrectDataset(MaskSplitByProfileDataset):
    def __init__(self, data_dir, mean=[0.56019358, 0.52410121, 0.501457], std=[0.23318603, 0.24300033, 0.24567522], val_ratio=0.2):
        self.indices = defaultdict(list)
        super().__init__(data_dir, mean, std, val_ratio)

    def __getitem__(self, index):
        image = self.read_image(index)
        mask_label = self.get_mask_label(index)
        #gender_label = self.get_gender_label(index)
        #age_label = self.get_age_label(index)
        #multi_class_label = self.encode_multi_class(mask_label, gender_label, age_label)
        if self.transform:
            image = self.transform(image)
        return image, mask_label

class DatasetFromSubset(MaskCorrectDataset, GenderDataset, AgeDataset, MaskSplitByProfileDataset):
    def __init__(self, subset, transform = None, old_transform = None):
        self.subset = subset
        self.transform = transform
        self.old_transform = old_transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.old_transform:
            if y % 3 == 2:
                x = self.old_transform(x)
            else:
                x = self.transform(x)
        else:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


class TestDataset(data.Dataset):
    def __init__(self, img_paths, resize, transform, mean=[0.56019358, 0.52410121, 0.501457], std=[0.23318603, 0.24300033, 0.24567522]):
        self.img_paths = img_paths
        self.transform = transform

    def __getitem__(self, index):
        image = cv2.imread(self.img_paths[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)


def get_mask_label(image_name):
    """
    이미지 파일 이름을 통해 mask label을 구합니다.

    :param image_name: 학습 이미지 파일 이름
    :return: mask label
    """
    if 'incorrect_mask' in image_name:
        return 1
    elif 'normal' in image_name:
        return 2
    elif 'mask' in image_name:
        return 0
    else:
        raise ValueError(f'No class for {image_name}')


def get_gender_label(gender):
    """
    gender label을 구하는 함수입니다.
    :param gender: `male` or `female`
    :return: gender label
    """
    return 0 if gender == 'male' else 1


def get_age_label(age):
    """
    age label을 구하는 함수입니다.
    :param age: 나이를 나타내는 int.
    :return: age label
    """
    # return 0 if int(age) < 30 else 1 if int(age) < 60 else 2
    return 0 if int(age) < 30 else 1 if int(age) < 58 else 2

def convert_gender_age(gender, age):
    """
    gender와 age label을 조합하여 고유한 레이블을 만듭니다.
    이를 구하는 이유는 train/val의 성별 및 연령 분포를 맞추기 위함입니다. (by Stratified K-Fold)
    :param gender: `male` or `female`
    :param age: 나이를 나타내는 int.
    :return: gender & age label을 조합한 레이블
    """
    gender_label = get_gender_label(gender)
    age_label = get_age_label(age)
    return gender_label * 3 + age_label


def convert_label(image_path, sep=False):
    """
    이미지의 label을 구하는 함수입니다.
    :param image_path: 이미지 경로를 나타내는 str
    :param sep: 마스크, 성별, 연령 label을 따로 반환할건지 합쳐서 할지 나타내는 bool 인수입니다. 참일 경우 따로 반환합니다.
    :return: 이미지의 label (int or list)
    """
    image_name = image_path.split('/')[-1]
    mask_label = get_mask_label(image_name)

    profile = image_path.split('/')[-2]
    image_id, gender, race, age = profile.split("_")
    gender_label = get_gender_label(gender)
    age_label = get_age_label(age)
    if sep:
        return mask_label, gender_label, age_label
    else:
        return mask_label * 6 + gender_label * 3 + age_label

class MaskDataset(data.Dataset):
    def __init__(self, image_dir, info, transform=None):
        self.image_dir = image_dir
        self.info = info
        self.transform = transform

        self.mean = [0.56019358, 0.52410121, 0.501457]
        self.std = [0.23318603, 0.24300033, 0.24567522]

        self.image_paths = [path for name in info.path.values for path in glob.glob(os.path.join(image_dir, name, '*'))]
        self.image_paths = list(filter(self.is_image_file, self.image_paths))
        self.image_paths = list(filter(self.remove_hidden_file, self.image_paths))

        self.labels = [convert_label(path, sep=False) for path in self.image_paths]

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = self.get_img(image_path)

        if self.transform:
            image = self.transform(image)
        label = torch.eye(18)[label]
        return image, label

    def __len__(self):
        return len(self.image_paths)
    
    def get_img(self, path):
        """
        이미지를 불러옵니다.
        """
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def is_image_file(self, filepath):
        """
        해당 파일이 이미지 파일인지 확인합니다.
        """
        return any(filepath.endswith(extension) for extension in IMG_EXTENSIONS)


    def remove_hidden_file(self, filepath):
        """
        `._`로 시작하는 숨김 파일일 경우 False를 반환합니다.
        """
        filename = filepath.split('/')[-1]
        return False if filename.startswith('._') else True
    
    @staticmethod
    def encode_multi_class(mask_label, gender_label, age_label):
        return mask_label * 6 + gender_label * 3 + age_label

    @staticmethod
    def decode_multi_class(multi_class_label):
        mask_label = (multi_class_label // 6) % 3
        gender_label = (multi_class_label // 3) % 2
        age_label = multi_class_label % 3
        return mask_label, gender_label, age_label

    @staticmethod
    def denormalize_image(image, mean, std):
        img_cp = image.copy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
        return img_cp

class MaskOldDataset(data.Dataset):
    def __init__(self, image_dir, info, transform = None, transform1 = None):
        self.image_dir = image_dir
        self.info = info
        self.transform = transform
        self.transform1 = transform1

        self.mean = [0.56019358, 0.52410121, 0.501457]
        self.std = [0.23318603, 0.24300033, 0.24567522]

        self.image_paths = [path for name in info.path.values for path in glob.glob(os.path.join(image_dir, name, '*'))]
        self.image_paths = list(filter(self.is_image_file, self.image_paths))
        self.image_paths = list(filter(self.remove_hidden_file, self.image_paths))

        self.labels = [convert_label(path, sep=False) for path in self.image_paths]

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = self.get_img(image_path)

        if self.transform1:
            if label % 3 == 2:
                image = self.transform1(image)
            else:
                image = self.transform(image)
        else:
            image = self.transform(image)
        label = torch.eye(18)[label]
        return image, label

    def __len__(self):
        return len(self.image_paths)
    
    def get_img(self, path):
        """
        이미지를 불러옵니다.
        """
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def is_image_file(self, filepath):
        """
        해당 파일이 이미지 파일인지 확인합니다.
        """
        return any(filepath.endswith(extension) for extension in IMG_EXTENSIONS)


    def remove_hidden_file(self, filepath):
        """
        `._`로 시작하는 숨김 파일일 경우 False를 반환합니다.
        """
        filename = filepath.split('/')[-1]
        return False if filename.startswith('._') else True
    
    @staticmethod
    def encode_multi_class(mask_label, gender_label, age_label):
        return mask_label * 6 + gender_label * 3 + age_label

    @staticmethod
    def decode_multi_class(multi_class_label):
        mask_label = (multi_class_label // 6) % 3
        gender_label = (multi_class_label // 3) % 2
        age_label = multi_class_label % 3
        return mask_label, gender_label, age_label

    @staticmethod
    def denormalize_image(image, mean, std):
        img_cp = image.copy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
        return img_cp