#!/usr/bin/env python3
import re, argparse, os
import matplotlib.pyplot as plt

def parse_log(log_path):
    epochs = []
    train_losses, train_accs, val_accs = [], [], []

    # matches: "Epoch 1: Train Loss 1.0823, Acc 0.6576"
    train_re = re.compile(r"Epoch\s+(\d+):\s*Train Loss\s*([\d\.]+),\s*Acc\s*([\d\.]+)")
    # matches: "Epoch 1: Val Acc 0.6345"
    val_re   = re.compile(r"Epoch\s+(\d+):\s*Val Acc\s*([\d\.]+)")

    with open(log_path, 'r') as f:
        for line in f:
            m = train_re.search(line)
            if m:
                e, loss, acc = m.groups()
                e, loss, acc = int(e), float(loss), float(acc)
                epochs.append(e)
                train_losses.append(loss)
                train_accs.append(acc)
            m = val_re.search(line)
            if m:
                e, acc = m.groups()
                e, acc = int(e), float(acc)
                val_accs.append(acc)

    return epochs, train_losses, train_accs, val_accs

def plot_two(xs, ys1, ys2, labels, ylabel, outpath):
    plt.figure(figsize=(6,4))
    plt.plot(xs, ys1, label=labels[0])
    plt.plot(xs, ys2, label=labels[1])
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def plot_one(xs, ys, label, ylabel, outpath):
    plt.figure(figsize=(6,4))
    plt.plot(xs, ys, label=label)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--log',    required=True, help="path to run.log")
    p.add_argument('--outdir', required=True, help="where to save plots")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    epochs, t_loss, t_acc, v_acc = parse_log(args.log)

    if t_acc and v_acc:
        plot_two(
            epochs, t_acc, v_acc,
            ['Train Acc','Val Acc'], 'Accuracy',
            os.path.join(args.outdir,'accuracy.png')
        )
        print("→ Saved accuracy.png")
    else:
        print("⚠️  Could not find both train and val accuracy in log.")

    if t_loss:
        # val_loss unavailable, so only train
        plot_one(
            epochs, t_loss,
            'Train Loss', 'Loss',
            os.path.join(args.outdir,'loss.png')
        )
        print("→ Saved loss.png (train only)")
        print("⚠️  No validation loss logged; if you need Val Loss, patch your training script to log it and re-run.")
    else:
        print("⚠️  No train-loss lines found in log.")