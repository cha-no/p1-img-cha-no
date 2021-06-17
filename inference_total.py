import argparse
import os
from importlib import import_module

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import TestDataset, MaskBaseDataset

MEAN = (0.56019358, 0.52410121, 0.501457)
STD = (0.23318603, 0.24300033, 0.24567522)

MEAN = (0.56019358, 0.52410121, 0.501457)
STD = (0.23318603, 0.24300033, 0.24567522)

def load_model(model_path, model_name, num_classes, device):
    model_cls = getattr(import_module("model"), model_name)
    model = model_cls(
        num_classes=num_classes
    )

    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)

    model.load_state_dict(torch.load(model_path, map_location=device))

    return model

def cal_label(gender, mask, age):
    return 6 * mask + 3 * gender + age


@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    """
    """

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    test_transform_module = getattr(import_module("dataset"), args.test_augmentation)  # default: BaseAugmentation

    test_transform = test_transform_module(
        resize=args.resize,
        mean=MEAN,
        std=STD,
    )

    dataset = TestDataset(img_paths, args.resize, test_transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers = 1,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    gender_model_dir = os.path.join(args.model_dir, args.gender_dir)
    mask_model_dir = os.path.join(args.model_dir, args.mask_dir)
    age_model_dir = os.path.join(args.model_dir, args.age_dir)
    
    gender_model = load_model(gender_model_dir, 'EfficientNet_b0', 2, device).to(device)
    mask_model = load_model(mask_model_dir, 'EfficientNet_b0', 3, device).to(device)
    age_model = load_model(age_model_dir, args.model, 3, device).to(device)

    print("Calculating inference results..")
    preds = []
    with torch.no_grad():
        gender_model.eval()
        mask_model.eval()
        age_model.eval()

        for idx, images in enumerate(loader):
            images = images.to(device)
            gender_outs = gender_model(images)
            mask_outs = mask_model(images)
            age_outs = age_model(images)
            
            gender_preds = torch.argmax(gender_outs, dim=-1)
            mask_preds = torch.argmax(mask_outs, dim=-1)
            age_preds = torch.argmax(age_outs, dim=-1)

            pred = cal_label(gender_preds, mask_preds, age_preds)
            preds.extend(pred.cpu().numpy())

    info['ans'] = preds
    info.to_csv(os.path.join(output_dir, f'output.csv'), index=False)
    print(f'Inference Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=2021, help='random seed (default: 2021)')
    parser.add_argument('--dataset', type=str, default='TestDataset', help='dataset augmentation type (default: MaskSplitByProfileDataset)')
    parser.add_argument('--batch_size', type=int, default=512, help='input batch size for validing (default: 1000)')
    parser.add_argument('--resize', type=int, default=256, help='resize size for image when you trained (default: 256)')
    parser.add_argument('--test_augmentation', type=str, default='BasicAugmentation', help='test augmentation (default: BasicAugmentation)')
    parser.add_argument('--model', type=str, default='EfficientNet_b0', help='model type (default: EfficientNet_b0)')
    parser.add_argument('--gender_dir', type=str, default='gender/BasicEpoch10SplitAge58/best.pth', help='gender model dir (default: gender/GenderBasicEpoch10Split/best.pth)')
    parser.add_argument('--mask_dir', type=str, default='mask/BasicEpoch10SplitAge58/best.pth', help='mask model dir (default: mask/MaskBasicEpoch10Split/best.pth)')
    parser.add_argument('--age_dir', type=str, default='age/BasicEpoch10SplitAge58/best.pth', help='age model dir (default: age/AgeBasicEpoch10Split/best.pth)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output/total'))

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    inference(data_dir, model_dir, output_dir, args)
