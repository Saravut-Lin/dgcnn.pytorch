import torch
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import numpy as np
import sklearn.metrics as metrics

import sys
import os
# Ensure we can import custom dataset module from the data/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'data'))
from market77_h5 import Market77SegDataset
from model import DGCNN_semseg_s3dis
from util import cal_loss, IOStream

import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import gc

def _init_(args):
    os.makedirs(os.path.join('outputs', args.exp_name, 'models'), exist_ok=True)
    io = IOStream(os.path.join('outputs', args.exp_name, 'run.log'))
    io.cprint('Experiment: %s' % args.exp_name)
    io.cprint(str(args))
    return io

def calculate_iou(pred_np, seg_np, num_classes):
    I = np.zeros(num_classes)
    U = np.zeros(num_classes)
    for cls in range(num_classes):
        I[cls] = np.sum((pred_np == cls) & (seg_np == cls))
        U[cls] = np.sum((pred_np == cls) | (seg_np == cls))
    return I / (U + 1e-10)

def train(args, io):
    train_dataset = Market77SegDataset(h5_path=args.data_path, split='train', use_color=True)
    test_dataset  = Market77SegDataset(h5_path=args.data_path, split='test', use_color=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers, drop_last=True)
    test_loader  = DataLoader(test_dataset,  batch_size=args.test_batch_size,
                              shuffle=False, num_workers=args.workers)

    total_train = len(train_loader.dataset)

    device = torch.device('cuda' if args.cuda else 'cpu')

    # Model and override final layer for num_classes
    model = DGCNN_semseg_s3dis(args).to(device)
    model.conv9 = nn.Conv1d(256, args.classes, kernel_size=1, bias=False).to(device)
    model = nn.DataParallel(model)

    # Optimizer: Adam with initial LR = args.lr
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Scheduler: cosine annealing LR
    scheduler = CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # Use fixed class weights to address class imbalance
    weights = torch.tensor([0.5555556, 5.0], dtype=torch.float).to(device)
    io.cprint(f"Using fixed class weights: {weights.cpu().numpy()}")
    criterion = nn.CrossEntropyLoss(weight=weights)

    best_iou = 0.0
    # Early-stopping setup
    patience = 20
    no_improve_count = 0

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(args.epochs):
        epoch_start = time.time()
        io.cprint(f"--- Starting Epoch {epoch+1}/{args.epochs} ---")

        model.train()
        train_loss = 0.0
        all_pred = []
        all_true = []

        for batch_idx, (data, seg) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} train", ncols=80), start=1):
            #data = data.permute(0, 2, 1).to(device)         # (B,3,N)
            data = data.to(device)                          # (B, C, N)
            seg  = seg.to(device)                            # (B,N)
            optimizer.zero_grad()
            seg_pred = model(data)                           # (B,C,N)
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()# (B,N,C)

            loss = criterion(seg_pred.view(-1, args.classes), seg.view(-1))
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * data.size(0)
            pred = seg_pred.max(dim=2)[1]                    # (B,N)

            all_pred.append(pred.cpu().numpy().reshape(-1))
            all_true.append(seg.cpu().numpy().reshape(-1))

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']
        io.cprint(f"Epoch {epoch+1} training time: {epoch_time:.2f}s, LR: {current_lr:.6f}")

        train_pred = np.concatenate(all_pred)
        train_true = np.concatenate(all_true)
        train_acc = metrics.accuracy_score(train_true, train_pred)
        ious = calculate_iou(train_pred, train_true, args.classes)
        mean_iou = np.nanmean(ious)

        avg_train_loss = train_loss / len(train_dataset)
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_acc)

        # Log per-class IoU
        io.cprint(
            f"Epoch {epoch+1}: Train Loss {avg_train_loss:.4f},"
            f" Acc {train_acc:.4f},"
            f" IoU_Class0 {ious[0]:.4f},"
            f" IoU_Class1 {ious[1]:.4f},"
            f" mIoU {mean_iou:.4f}"
        )

        # Validation
        val_start = time.time()
        model.eval()
        val_pred = []
        val_true = []
        val_loss = 0.0
        with torch.no_grad():
            for data, seg in test_loader:
                #data = data.permute(0, 2, 1).to(device)
                data = data.to(device)                          # (B, C, N)
                seg  = seg.to(device)
                seg_pred = model(data).permute(0, 2, 1).contiguous()
                loss = criterion(seg_pred.view(-1, args.classes), seg.view(-1))
                val_loss += loss.item() * data.size(0)
                pred = seg_pred.max(dim=2)[1]
                val_pred.append(pred.cpu().numpy().reshape(-1))
                val_true.append(seg.cpu().numpy().reshape(-1))

        val_pred = np.concatenate(val_pred)
        val_true = np.concatenate(val_true)
        val_acc = metrics.accuracy_score(val_true, val_pred)
        val_ious = calculate_iou(val_pred, val_true, args.classes)
        val_miou = np.nanmean(val_ious)

        avg_val_loss = val_loss / len(test_dataset)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)

        val_time = time.time() - val_start
        # Log per-class IoU for validation
        io.cprint(
            f"Epoch {epoch+1}: Val Acc {val_acc:.4f},"
            f" IoU_Class0 {val_ious[0]:.4f},"
            f" IoU_Class1 {val_ious[1]:.4f},"
            f" Val mIoU {val_miou:.4f}"
        )
        '''
        io.cprint(f"Validation time: {val_time:.2f}s")
        if args.cuda:
            max_mem = torch.cuda.max_memory_allocated() / (1024**2)
            io.cprint(f"Max GPU memory allocated: {max_mem:.1f} MB")
        '''

        # Step LR scheduler each epoch
        scheduler.step()

        # Early-stopping and checkpoint
        if val_miou > best_iou:
            best_iou = val_miou
            no_improve_count = 0
            # Save best
            torch.save(model.state_dict(),
                       os.path.join('outputs', args.exp_name, 'models', 'best_model.pth'))
            # Save checkpoint every epoch
            torch.save({
                'epoch': epoch + 1,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'best_iou': best_iou,
            }, os.path.join('outputs', args.exp_name, 'models', f'ckpt_epoch_{epoch+1}.pth'))
        else:
            no_improve_count += 1

        # Stop if no improvement for `patience` epochs
        if no_improve_count >= patience:
            io.cprint(f"No improvement in {patience} epochs. Stopping early.")
            break

    # Plot Loss
    epochs = range(1, args.epochs + 1)
    plt.figure()
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"outputs/{args.exp_name}/loss_curve.png")
    plt.close()

    # Plot Accuracy
    plt.figure()
    plt.plot(epochs, train_accuracies, label='Train Acc')
    plt.plot(epochs, val_accuracies, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f"outputs/{args.exp_name}/accuracy_curve.png")
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Market77 Semantic Segmentation Training')
    parser.add_argument('--exp_name',      type=str,   default='market77_exp')
    parser.add_argument('--model',         type=str,   default='dgcnn', choices=['dgcnn'])
    parser.add_argument('--data_path',     type=str,   required=True,
                        help='Path to Market77 HDF5 file')
    parser.add_argument('--batch_size',    type=int,   default=16)
    parser.add_argument('--test_batch_size', type=int, default=16)
    parser.add_argument('--epochs',        type=int,   default=100)
    parser.add_argument('--use_sgd',       type=bool,  default=True)
    parser.add_argument('--lr',            type=float, default=0.001)
    parser.add_argument('--momentum',      type=float, default=0.9)
    parser.add_argument('--scheduler',     type=str,   default='cos', choices=['cos','step'])
    parser.add_argument('--no_cuda',       type=bool,  default=False)
    parser.add_argument('--seed',          type=int,   default=1)
    parser.add_argument('--num_points',    type=int,   default=20480)
    parser.add_argument('--dropout',       type=float, default=0.5)
    parser.add_argument('--emb_dims',      type=int,   default=1024)
    parser.add_argument('--k',             type=int,   default=20)
    parser.add_argument('--classes',       type=int,   default=2,
                        help='Number of segmentation classes')
    parser.add_argument('--workers',       type=int,   default=0)
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    io = _init_(args)
    train(args, io)