import argparse
import glob
import json
import os
import random
import re
import itertools
from scipy.special import softmax
from importlib import import_module
from pathlib import Path

import warnings
warnings.filterwarnings(action='ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import torch
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.data import Subset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable

from adamp import AdamP

from dataset import MaskBaseDataset, MaskDataset, DatasetFromSubset, convert_gender_age
from loss import create_criterion, AverageMeter

MEAN = (0.56019358, 0.52410121, 0.501457)
STD = (0.23318603, 0.24300033, 0.24567522)

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): patience epoch이 지나도록 val loss가 개선되지 않을 경우 학습 중단.
     
            verbose (bool): True일 경우 각 validation loss의 개선 사항 메세지 출력.

            delta (float): 개선되었다고 간주하는 monitered quantity의 최소 변화
         
            path (str): checkpoint저장 경로
                          
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)

        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def grid_image(np_images, gts, preds, n=9, shuffle=False):
    batch_size = np_images.shape[0]
    assert n <= batch_size
    choices = random.choices(range(batch_size), k=9) if shuffle else list(range(n))
    figure = plt.figure(figsize=(12, 18))  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
    plt.subplots_adjust(top=0.9)               # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
    n_grid = np.ceil(n ** 0.5)
    tasks = ["mask", "gender", "age"]
    for idx, choice in enumerate(choices):
        gt = gts[choice].item()
        pred = preds[choice].item()
        image = np_images[choice]
        # title = f"gt: {gt}, pred: {pred}"
        gt_decoded_labels = MaskDataset.decode_multi_class(gt)
        pred_decoded_labels = MaskDataset.decode_multi_class(pred)
        title = "\n".join([
            f"{task} - gt: {gt_label}, pred: {pred_label}"
            for gt_label, pred_label, task
            in zip(gt_decoded_labels, pred_decoded_labels, tasks)
        ])

        plt.subplot(n_grid, n_grid, idx + 1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)

    return figure

def get_all_datas(model, device, dataloader):
    """ dataloader의 모든 data들을 가져온다. """
    model.eval()
    
    all_images = torch.tensor([]).to(device)
    all_labels = torch.tensor([]).to(device)
    all_preds = torch.tensor([]).to(device)

    with torch.no_grad():
        for idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            if args.dataset == 'MaskDataset' or args.dataset == 'MaskOldDataset':
                labels = labels.argmax(dim = -1)

            preds = model(images)
            all_images = torch.cat((all_images, images))
            all_labels = torch.cat((all_labels, labels))
            all_preds = torch.cat((all_preds, preds))

    return all_images, all_labels, all_preds

def log_confusion_matrix(labels, preds, num_classes):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 9))
    fig.suptitle("Confusion Matrix", fontsize=16)
    cmap = plt.cm.Oranges

    cm = confusion_matrix(labels, preds)
    
    axes[0].imshow(cm, interpolation="nearest", cmap=cmap)

    axes[0].set_xticks(range(num_classes))
    axes[0].set_yticks(range(num_classes))
    axes[0].set_ylabel("True label")
    axes[0].set_xlabel("Predicted label")

    thresh = cm.max() / 2.0

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        axes[0].text(
            j, i, cm[i, j], horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    np.fill_diagonal(cm, 0)
    axes[1].imshow(cm, interpolation="nearest", cmap=cmap)

    axes[1].set_xticks(range(num_classes))
#     axes[1].set_xticklabels(classes)
    axes[1].set_yticks([])
    axes[1].set_xlabel("Predicted label")

    thresh = cm.max() / 2.0

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        axes[1].text(
            j, i, cm[i, j], horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )
    plt.show()
    return fig

def _log_plots_image(ax, image, pred, pred_label, true_label, num_classes):
    ax.grid(False)
    color = "blue" if pred_label == true_label else "red"
    
    classes = np.array([str(i) for i in range(num_classes)])
    
    ax.imshow(np.moveaxis(image, 0, -1))

    ax.set_xlabel(
        "pred: {} {:2.0f}% | (true: {})".format(
            classes[pred_label], 100 * pred[pred_label], classes[true_label]
        ),
        color=color,
    )

def _log_plots_distribution(ax, pred, pred_label, true_label, num_classes):
    ax.grid(False)
    ax.set_ylim([0, 1])

    thisplot = ax.bar(range(num_classes), pred, color="#777777")

    thisplot[pred_label].set_color("red")
    thisplot[true_label].set_color("blue")


def plots_result(images, labels, preds, num_classes, title="plots_result"):
    preds = softmax(preds, axis=1)

    num_rows = num_cols = int(len(images) ** 0.5)
    num_images = num_rows * num_cols

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols * 2, figsize=(36, 18))
    fig.suptitle(title, fontsize=54)
    plt.setp(axes, xticks=[], yticks=[])

    for idx in range(num_images):
        image, pred, label = images[idx], preds[idx], int(labels[idx])

        num_row, num_col = idx // num_rows, idx % num_cols
        pred_label = np.argmax(pred)

        _log_plots_image(
            axes[num_row][num_col * 2], image, pred, pred_label, label, num_classes
        )

        _log_plots_distribution(
            axes[num_row][num_col * 2 + 1], pred, pred_label, label, num_classes
        )

    return fig

def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train(data_dir, model_dir, args):
    seed_everything(args.seed)

    save_dir = increment_path(os.path.join(model_dir, args.name))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    info = pd.read_csv('/opt/ml/input/data/train/train.csv')

    info['gender_age'] = info.apply(lambda x: convert_gender_age(x.gender, x.age), axis = 1)
    n_fold = int(1 / args.val_ratio)

    skf = StratifiedKFold(n_splits = n_fold, shuffle=True)
    info.loc[:, 'fold'] = 0
    for fold_num, (train_index, val_index) in enumerate(skf.split(X = info.index, y = info.gender_age.values)):
        info.loc[info.iloc[val_index].index, 'fold'] = fold_num

    fold_idx = 0
    train = info[info.fold != fold_idx].reset_index(drop=True)
    val = info[info.fold == fold_idx].reset_index(drop=True)

    # -- dataset
    dataset_module = getattr(import_module("dataset"), args.dataset)  # default: MaskDataset

    # -- augmentation
    train_transform_module = getattr(import_module("dataset"), args.train_augmentation)  # default: BaseAugmentation
    val_transform_module = getattr(import_module("dataset"), args.val_augmentation)  # default: BaseAugmentation

    train_transform = train_transform_module(
        resize=args.resize,
        mean=MEAN,
        std=STD,
    )
    val_transform = val_transform_module(
        resize=args.resize,
        mean=MEAN,
        std=STD,
    )
    
    print(train_transform.transform, val_transform.transform)

    if args.dataset == 'MaskDataset' or args.dataset == 'MaskOldDataset':
        if args.dataset == 'MaskOldDataset':
            old_transform_module = getattr(import_module('dataset'), args.old_augmentation)

            old_transform = old_transform_module(
                resize=args.resize,
                mean=MEAN,
                std=STD,
            )
            train_dataset = dataset_module(data_dir, train, train_transform, old_transform)
            if args.val_old:
                val_dataset = dataset_module(data_dir, val, val_transform, old_transform)
            else:
                val_dataset = dataset_module(data_dir, val, val_transform)
        else:
            train_dataset = dataset_module(data_dir, train, train_transform)
            val_dataset = dataset_module(data_dir, val, val_transform)
    else:
        dataset = dataset_module(
            data_dir=data_dir,
        )

        # dataset.set_transform(transform)
        # -- data_loader
        train_set, val_set = dataset.split_dataset()
        if args.val_old:
            old_transform_module = getattr(import_module('dataset'), args.old_augmentation)

            old_transform = old_transform_module(
                resize=args.resize,
                mean=MEAN,
                std=STD,
            )

            train_dataset = DatasetFromSubset(
                train_set, transform = train_transform, old_transform = old_transform
            )
        else:
            train_dataset = DatasetFromSubset(
                train_set, transform = train_transform
            )
        val_dataset = DatasetFromSubset(
            val_set, transform = val_transform
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=True,
        pin_memory=use_cuda,
        #drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.valid_batch_size,
        num_workers=1,
        shuffle=False,
        pin_memory=use_cuda,
        #drop_last=True,
    )

    # -- model
    model_module = getattr(import_module("model"), args.model)  # default: BaseModel
    model = model_module(
        num_classes=args.num_classes
    ).to(device)
    model = torch.nn.DataParallel(model)

    # -- loss & metric
    if args.criterion == 'f1' or args.criterion == 'label_smoothing' or args.criterion == 'f1cross':
        criterion = create_criterion(args.criterion, classes = args.num_classes)
    else:
        criterion = create_criterion(args.criterion)
    
    if args.optimizer == 'AdamP':
        optimizer = AdamP(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    else:
        opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: SGD
        optimizer = opt_module(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=2, eta_min=1e-6)
    elif args.scheduler == 'reduce':
        scheduler = ReduceLROnPlateau(optimizer, factor = 0.5, patience = 5)
    elif args.scheduler == 'step':
        scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)
    else:
        scheduler = None

    # -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    best_val_acc = 0
    best_val_loss = np.inf
    print("This notebook use [%s]."%(device))

    early_stopping = EarlyStopping(patience = args.patience, verbose = True)

    for epoch in range(args.epochs):
        # train loop
        model.train()
        loss_value = 0
        matches = 0

        train_loss, train_acc = AverageMeter(), AverageMeter()

        for idx, train_batch in enumerate(train_loader):
            inputs, labels = train_batch
            if args.dataset == 'MaskDataset' or args.dataset == 'MaskOldDataset':
                labels = labels.argmax(dim = -1)
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            if args.mixup and (idx + epoch) % 2:
                inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, alpha = 1.0)
                inputs, labels_a, labels_b = map(Variable, (inputs, labels_a, labels_b))

                outputs = model(inputs)
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)

                _, predicted = torch.max(outputs.data, 1)

                correct = (lam * predicted.eq(labels_a.data).cpu().sum().float() + (1 - lam) * predicted.eq(labels_b.data).cpu().sum().float())
                acc = correct / len(labels)
            
            else:
                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)
                loss = criterion(outs, labels)
                acc = (preds == labels).sum().item() / len(labels)

            loss.backward()
            optimizer.step()

            #loss_value += loss.item()
            #matches += (preds == labels).sum().item()

            train_loss.update(loss.item(), len(labels))
            train_acc.update(acc, len(labels))

            if (idx + 1) % args.log_interval == 0:
                #train_loss = loss_value / args.log_interval
                #train_acc = matches / args.batch_size / args.log_interval
                train_f1_acc = f1_score(preds.cpu().detach().type(torch.int), labels.cpu().detach().type(torch.int), average='macro')
                current_lr = get_lr(optimizer)
                print(
                    f"Epoch[{epoch + 1}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss.avg:.4f} || training accuracy {train_acc.avg:4.2%} || train_f1_acc {train_f1_acc:.4} || lr {current_lr}"
                )
                logger.add_scalar("Train/loss", train_loss.avg, epoch * len(train_loader) + idx)
                logger.add_scalar("Train/accuracy", train_acc.avg, epoch * len(train_loader) + idx)

                loss_value = 0
                matches = 0

        scheduler.step()
        
        val_loss, val_acc = AverageMeter(), AverageMeter()
        # val loop
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            val_labels_items = np.array([])
            val_preds_items = np.array([])
            figure = None
            for val_batch in val_loader:
                inputs, labels = val_batch
                if args.dataset == 'MaskDataset' or args.dataset == 'MaskOldDataset':
                    labels = labels.argmax(dim = -1)

                inputs = inputs.to(device)
                labels = labels.to(device)

                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)

                #loss_item = criterion(outs, labels).item()
                #acc_item = (labels == preds).sum().item()
                #val_loss_items.append(loss_item)
                #val_acc_items.append(acc_item)
                
                loss = criterion(outs, labels)
                acc = (preds == labels).sum().item() / len(labels)

                val_loss.update(loss.item(), len(labels))
                val_acc.update(acc, len(labels))

                val_labels_items = np.concatenate([val_labels_items, labels.cpu().numpy()])
                val_preds_items = np.concatenate([val_preds_items, preds.cpu().numpy()])

                if figure is None:
                    if epoch % 2:
                        images, labels, preds = get_all_datas(model, device, val_loader)
                        figure = log_confusion_matrix(labels.cpu().numpy(), np.argmax(preds.cpu().numpy(), axis=1), args.num_classes)
                        # figure2 = plots_result(images.cpu().numpy()[:36], labels.cpu().numpy()[:36], preds.cpu().numpy()[:36], args.num_classes, title="plots_result")
                    else:
                        inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                        inputs_np = val_dataset.denormalize_image(inputs_np, MEAN, STD)
                        figure = grid_image(inputs_np, labels, preds, 9, False)

            # val_loss = np.sum(val_loss_items) / len(val_loader)
            # val_acc = np.sum(val_acc_items) / len(val_set)
            val_f1_acc = f1_score(val_labels_items.astype(np.int), val_preds_items.astype(np.int), average='macro')
            
            best_val_acc = max(best_val_acc, val_acc.avg)
            # best_val_loss = min(best_val_loss, val_loss)
            if val_loss.avg < best_val_loss:
                print(f"New best model for val loss : {val_loss.avg:4.2%}! saving the best model..")
                torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
                best_val_loss = val_loss.avg
            torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
            print(
                f"[Val] acc : {val_acc.avg:4.2%}, loss : {val_loss.avg:.4f} || val_f1_acc : {val_f1_acc:.4} || "
                f"best acc : {best_val_acc:4.2%}, best loss : {best_val_loss:.4f}"
            )
            logger.add_scalar("Val/loss", val_loss.avg, epoch)
            logger.add_scalar("Val/accuracy", val_acc.avg, epoch)
            logger.add_figure("results", figure, epoch)
            # logger.add_figure("results1", figure2, epoch)
            
            early_stopping(val_loss.avg, model)

            if early_stopping.early_stop:
                print('Early stopping...')
                break

            print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=2021, help='random seed (default: 2021)')
    parser.add_argument('--old', type=int, default=60, help='old (default: 60)')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train (default: 1)')
    parser.add_argument('--dataset', type=str, default='MaskSplitByProfileDataset', help='dataset augmentation type (default: MaskSplitByProfileDataset)')
    parser.add_argument('--train_augmentation', type=str, default='BasicAugmentation', help='train data augmentation type (default: BasicAugmentation)')
    parser.add_argument('--val_augmentation', type=str, default='BasicAugmentation', help='valid data augmentation type (default: BasicAugmentation)')
    parser.add_argument('--old_augmentation', type=str, default='Augmentation4', help='old data augmentation type (default: Augmentation4)')
    parser.add_argument('--val_old', type=bool, default=False, help='val_old (default: False)')
    parser.add_argument('--mixup', type=bool, default=False, help='mixup (default: False)')
    parser.add_argument("--num_classes", type=int, default=18, help='num_classes')
    parser.add_argument("--resize", nargs="+", type=int, default=256, help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=12, help='input batch size for validing (default: 12)')
    parser.add_argument('--model', type=str, default='EfficientNet_b0', help='model type (default: EfficientNet_b0)')
    parser.add_argument('--patience', type=int, default=5, help='patience (default: 5)')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type (default: Adam)')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate (default: 3e-4)')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay (default: 5e-4)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='f1', help='criterion type (default: f1)')
    parser.add_argument('--lr_decay_step', type=int, default=10, help='learning rate scheduler deacy step (default: 10)')
    parser.add_argument('--scheduler', type=str, default='cosine', help='learning rate scheduler (default: CosineAnnealingLR)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)