import argparse
import os
from importlib import import_module
from tqdm import tqdm

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import TestDataset, MaskBaseDataset

MEAN = (0.56019358, 0.52410121, 0.501457)
STD = (0.23318603, 0.24300033, 0.24567522)

def load_model(saved_model, num_classes, device):
    model_cls = getattr(import_module("model"), args.model)
    model = model_cls(
        num_classes=num_classes
    )

    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)

    model_path = os.path.join(saved_model, args.model_path)
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    """
    """
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = MaskBaseDataset.num_classes  # 18
    model = load_model(model_dir, num_classes, device).to(device)
    model.eval()

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
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers = 1,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("This notebook use [%s]."%(device))

    print("Calculating inference results..")
    preds = []
    with torch.no_grad():
        for idx, images in tqdm(enumerate(loader)):
            images = images.to(device)
            pred = model(images)
            pred = pred.argmax(dim=-1)
            preds.extend(pred.cpu().numpy())

    info['ans'] = preds
    info.to_csv(os.path.join(output_dir, f'output.csv'), index=False)
    print(f'Inference Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    parser.add_argument('--resize', type=int, default=256, help='resize size for image when you trained (default: 256)')
    parser.add_argument('--test_augmentation', type=str, default='BasicAugmentation', help='test augmentation (default: BasicAugmentation)')
    parser.add_argument('--model', type=str, default='EfficientNet_b0', help='model type (default: EfficientNet_b0)')
    parser.add_argument('--model_path', type=str, default='best.pth', help='model path (default: best.pth)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    inference(data_dir, model_dir, output_dir, args)
