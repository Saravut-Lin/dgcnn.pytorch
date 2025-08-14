DGCNN (PyTorch) — Semantic Segmentation for MiniMarket Scenes

Dynamic Graph CNN (DGCNN) implemented in PyTorch for 3D point‑cloud semantic segmentation (object vs. background) on a MiniMarket‑style dataset.
This repo includes training on HDF5 datasets and inference on raw .pcd scenes.

⸻

Quick Start

1) Environment

Create and activate the conda environment from the provided YAML:

# clone the repo
git clone https://github.com/Saravut-Lin/dgcnn.pytorch.git
cd dgcnn.pytorch

# create environment
conda env create -f dgcnn_env.yml
conda activate dgcnn  # (or the name defined inside the YAML)

The YAML pins Python / PyTorch and the core dependencies used for training and Open3D‑based visualization.

⸻

2) Dataset
    1.    Generate the dataset using the MiniMarket processing repo:

git clone https://github.com/msorour/MiniMarket_dataset_processing.git
cd MiniMarket_dataset_processing
# Follow that repo's instructions to produce the HDF5 file


    2.    Copy the generated .h5/.hdf5 file into this repo’s dataset/ folder:

dgcnn.pytorch/
├─ dataset/
│  └─ market_semseg.h5        # ← your generated file (name is up to you)



By default, the training script expects the dataset under ./dataset. If you use a different path or filename, adjust the corresponding argument or constant in the training script.

⸻

3) Train

Run the market semantic‑segmentation training script:

python main_semseg_market.py

    •    Check the script header/args for optional flags (e.g., epochs, batch size, k‑NN, logging directory).
    •    The script prints and saves checkpoints; note the best checkpoint path (e.g., at the “best epoch”).

⸻

4) Inference on Real‑World Scenes (.pcd)

After training, run inference on a raw .pcd file using your saved checkpoint:

python dgcnn_inference.py --ckpt /path/to/best_checkpoint.pth --pcd /path/to/scene.pcd

    •    The inference utility reads a PCD, removes invalid/NaN points, optionally chunks the cloud with voting, and writes/visualizes the predicted segmentation (object vs. background).
    •    If your script uses different flag names, pass the checkpoint and PCD paths according to the script help (-h).

⸻

Results (Summary)

Training performance
    •    Loss: 1.08 → ~0.024 over 100 epochs
    •    Accuracy: 65.8% → 99.8%
    •    mIoU: 0.42 → 0.987
    •    Class IoU: background 0.997, object 0.974
    •    Best checkpoint: epoch 85 (consistent plateau after LR restarts)

Validation performance
    •    Accuracy: 63% → 99.7%
    •    Best mIoU: 0.985 at epoch 85
    •    Class IoU (final): background 0.997, object 0.970
    •    Training/validation curves nearly overlap after epoch 30 (gap ≤ 0.003), indicating minimal overfitting despite a 1:9 foreground/background imbalance.

Qualitative test example (best epoch 85)
Object is orange; background is blue.

Real‑world PCD inference (10 scenes)
    •    Valid points per scene after NaN removal: ~222k–371k
    •    Fraction predicted as target: 36.8–56.6% (mean ~49%)
    •    Typical behavior: correctly localizes the object region but boundaries can be coarse, with some leakage onto neighboring cylindrical items when objects are tightly packed.

If you want the README to render these figures on GitHub, place the images at:
    •    figures/dgcnn_IoU.png
    •    figures/inference_dgcnn.png

⸻

Repo Layout (minimal)

dgcnn.pytorch/
├─ main_semseg_market.py      # training entry point (MiniMarket semseg)
├─ dgcnn_inference.py         # inference on real-world PCDs
├─ dgcnn_env.yml              # conda environment
├─ dataset/                   # put your generated HDF5 here
├─ checkpoints/               # (optional) where you save models
└─ figures/                   # images for README (optional)


⸻

Tips & Troubleshooting
    •    CUDA / memory: If you hit OOM, reduce batch size or neighborhood size k.
    •    PCD ingest: Ensure your .pcd files are valid and not all NaNs; the inference script removes invalid points before prediction.
    •    Paths: Most issues come from wrong dataset/checkpoint paths—double‑check CLI args and any constants at the top of the scripts.

⸻

Acknowledgments
    •    Model: Dynamic Graph CNN (DGCNN) for point‑cloud learning (Wang et al.).
    •    Dataset preparation: MiniMarket_dataset_processing (see link above).

⸻

License

See the repository’s LICENSE file for details.

⸻

Citation

If you use this code or results, please cite the DGCNN paper and the MiniMarket processing toolkit as appropriate.

@inproceedings{wang2019dynamic,
  title={Dynamic Graph CNN for Learning on Point Clouds},
  author={Wang, Yue and Sun, Yongbin and Liu, Ziwei and Sarma, Sanjay E. and Bronstein, Michael M. and Solomon, Justin M.},
  booktitle={ACM Transactions on Graphics (TOG)},
  year={2019}
}


⸻

Reproduction checklist
    •    Create conda env from dgcnn_env.yml
    •    Generate HDF5 with MiniMarket_dataset_processing → place in ./dataset
    •    python main_semseg_market.py → note best checkpoint (e.g., epoch 85)
    •    python dgcnn_inference.py --ckpt <best.pth> --pcd <scene.pcd> → visualize predictions
