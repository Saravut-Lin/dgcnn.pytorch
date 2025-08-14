#!/usr/bin/env python3
"""
dgcnn_inference.py: Inference script for DGCNN semantic segmentation
Handles real-world PCD files with proper preprocessing and outputs PLY segmentation masks
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import open3d as o3d
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

# Add model directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import DGCNN_semseg_s3dis

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DGCNNInference:
    def __init__(self, checkpoint_path, num_classes=2, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        self.num_points = 20480  # Model expects this many points
        self.k = 20  # k-nearest neighbors for graph construction
        
        # Load model
        self.model = self._load_model(checkpoint_path)
        logging.info(f"Model loaded from {checkpoint_path}")
        
    def _load_model(self, checkpoint_path):
        """Load DGCNN model from checkpoint"""
        # Create args namespace for model initialization
        class Args:
            def __init__(self, num_classes):
                self.emb_dims = 1024
                self.k = 20
                self.dropout = 0.5
                self.classes = num_classes
        
        args = Args(self.num_classes)
        model = DGCNN_semseg_s3dis(args).to(self.device)
        
        # Override final conv layer for binary segmentation
        model.conv9 = torch.nn.Conv1d(256, self.num_classes, kernel_size=1, bias=False).to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get('model_state', checkpoint.get('state_dict', checkpoint))
        else:
            state_dict = checkpoint
            
        # Remove 'module.' prefix if present (from DataParallel)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict)
        model.eval()
        
        return model
    
    def preprocess_point_cloud(self, points, colors=None):
        """Preprocess point cloud to match training data format"""
        # Center the point cloud
        centroid = np.mean(points, axis=0)
        points_centered = points - centroid
        
        # Normalize to unit sphere
        max_dist = np.max(np.linalg.norm(points_centered, axis=1))
        if max_dist > 0:
            points_normalized = points_centered / max_dist
        else:
            points_normalized = points_centered
            
        # Combine with colors if available
        if colors is not None:
            # Normalize colors to [0, 1]
            colors_normalized = colors / 255.0 if colors.max() > 1 else colors
            features = np.concatenate([points_normalized, colors_normalized], axis=1)
        else:
            # If no colors, use zero padding
            features = np.concatenate([points_normalized, np.zeros((len(points), 3))], axis=1)
            
        return features, centroid, max_dist
    
    def sample_points(self, features, target_num_points):
        """Sample or pad points to reach target number"""
        num_points = features.shape[0]
        
        if num_points >= target_num_points:
            # Random sampling
            indices = np.random.choice(num_points, target_num_points, replace=False)
            return features[indices], indices
        else:
            # Pad by repeating points
            pad_num = target_num_points - num_points
            pad_indices = np.random.choice(num_points, pad_num, replace=True)
            padded_features = np.vstack([features, features[pad_indices]])
            # Return None for indices since we're padding
            return padded_features, None
    
    def inference_patch(self, patch_features):
        """Run inference on a single patch"""
        # Convert to tensor and reshape for model input
        # Model expects (batch_size, channels, num_points)
        patch_tensor = torch.from_numpy(patch_features).float()
        patch_tensor = patch_tensor.transpose(0, 1).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(patch_tensor)  # (1, num_classes, num_points)
            predictions = logits.argmax(dim=1).squeeze().cpu().numpy()
            
        return predictions
    
    def segment_iterative_nn(self, points, colors=None):
        """
        Iteratively pull the closest self.num_points via k-NN,
        run inference on each cluster, and accumulate predictions
        so that every point gets processed exactly once.
        """
        N = len(points)
        processed = np.zeros(N, dtype=bool)
        all_preds = np.zeros(N, dtype=np.int32)
        vote_counts = np.zeros(N, dtype=np.int32)

        # Start with all points unprocessed
        unprocessed_idx = np.arange(N)
        while unprocessed_idx.size > 0:
            # Determine cluster size (<= num_points)
            k = min(self.num_points, unprocessed_idx.size)
            nbrs = NearestNeighbors(n_neighbors=k).fit(points[unprocessed_idx])
            seed = points[unprocessed_idx[0]].reshape(1, -1)
            _, nbr_inds = nbrs.kneighbors(seed)
            local_inds = nbr_inds[0]
            cluster_idx = unprocessed_idx[local_inds]

            # Extract coordinates and optional colors
            cluster_pts = points[cluster_idx]
            cluster_cols = colors[cluster_idx] if colors is not None else None

            # Preprocess + sample/pad to fixed size
            feats, _, _ = self.preprocess_point_cloud(cluster_pts, cluster_cols)
            if cluster_idx.size < self.num_points:
                feats, sample_ids = self.sample_points(feats, self.num_points)
            else:
                sample_ids = np.arange(self.num_points)

            # Inference
            preds = self.inference_patch(feats)

            # Map votes back
            for i in range(cluster_idx.size):
                src = sample_ids[i] if sample_ids is not None else i
                orig = cluster_idx[src]
                all_preds[orig] += preds[i]
                vote_counts[orig] += 1

            # Mark as processed
            processed[cluster_idx] = True
            unprocessed_idx = np.where(~processed)[0]

        # Finalize predictions
        mask = vote_counts > 0
        all_preds[mask] = (all_preds[mask] / vote_counts[mask]).round().astype(np.int32)

        return points, (colors if colors is not None else np.zeros((N, 3))), all_preds
    
    def process_pcd_file(self, pcd_path, output_path, grid_size=0.15, overlap=0.3):
        """Process a PCD file and save segmentation result as PLY"""
        logging.info(f"Loading PCD file: {pcd_path}")
        
        # Load PCD file
        pcd = o3d.io.read_point_cloud(pcd_path)
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) * 255 if pcd.has_colors() else None

        # Remove NaN points from loaded data
        valid_mask = ~np.isnan(points).any(axis=1)
        num_total = len(points)
        num_removed = np.sum(~valid_mask)
        if num_removed > 0:
            logging.info(f"Removing {num_removed} NaN points (from {num_total}); proceeding with {num_total - num_removed} points")
            points = points[valid_mask]
            if colors is not None:
                colors = colors[valid_mask]
        else:
            logging.info(f"No NaN points found; proceeding with {num_total} points")
        
        # Segment the point cloud
        segmented_points, segmented_colors, predictions = self.segment_iterative_nn(points, colors)
        
        # Create segmentation visualization
        # Class 0: Background (blue), Class 1: Target object (red)
        segmentation_colors = np.zeros((len(predictions), 3))
        # Class 0 as target (red), Class 1 as alien/other object (blue)
        segmentation_colors[predictions == 0] = [255, 0, 0]   # Red for target object
        segmentation_colors[predictions == 1] = [0, 0, 255]   # Blue for alien/other object
        
        # Create output point cloud
        output_pcd = o3d.geometry.PointCloud()
        output_pcd.points = o3d.utility.Vector3dVector(segmented_points)
        output_pcd.colors = o3d.utility.Vector3dVector(segmentation_colors / 255.0)
        
        # Save as PLY
        o3d.io.write_point_cloud(output_path, output_pcd)
        logging.info(f"Saved segmentation result to: {output_path}")
        
        # Print statistics
        num_target = np.sum(predictions == 0)
        num_background = np.sum(predictions == 1)
        logging.info(f"Segmentation statistics:")
        logging.info(f"  Target object points: {num_target} ({100*num_target/len(predictions):.1f}%)")
        logging.info(f"  Background points: {num_background} ({100*num_background/len(predictions):.1f}%)")
        
        return output_pcd


def main():
    parser = argparse.ArgumentParser(description='DGCNN Inference for Point Cloud Segmentation')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth file)')
    parser.add_argument('--input_pcd', type=str, required=True,
                        help='Path to input PCD file')
    parser.add_argument('--output_ply', type=str, required=True,
                        help='Path to output PLY file')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Number of segmentation classes')
    parser.add_argument('--grid_size', type=float, default=0.15,
                        help='Size of processing grid in meters')
    parser.add_argument('--overlap', type=float, default=0.3,
                        help='Overlap ratio between patches (0-1)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize the result after saving')
    
    args = parser.parse_args()
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output_ply)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize inference engine
    inference = DGCNNInference(
        checkpoint_path=args.checkpoint,
        num_classes=args.num_classes,
        device=args.device
    )
    
    # Iteratively process realworld_scene_1.pcd through realworld_scene_10.pcd
    durations = []
    input_dir = Path(args.input_pcd).parent
    output_dir = Path(args.output_ply).parent
    for i in range(1, 11):
        pcd_file = input_dir / f"realworld_scene_{i}.pcd"
        output_file = output_dir / f"scene_segmented_{i}.ply"
        logging.info(f"Processing {pcd_file}")
        start_time = time.time()
        output_pcd = inference.process_pcd_file(
            pcd_path=str(pcd_file),
            output_path=str(output_file),
            grid_size=args.grid_size,
            overlap=args.overlap
        )
        elapsed = time.time() - start_time
        logging.info(f"Inference on {pcd_file} took {elapsed:.2f} seconds")
        durations.append(elapsed)
    # Print mean inference time
    if durations:
        mean_time = sum(durations) / len(durations)
        print(f"Mean inference time for {len(durations)} PCD files: {mean_time:.2f} seconds")
    # Visualize last result if requested
    if args.visualize:
        logging.info("Visualizing last result...")
        o3d.visualization.draw_geometries([output_pcd], window_name="Segmentation Result")


if __name__ == "__main__":
    main()


"""
python dgcnn_inference.py \
    --checkpoint /home/s2671222/dgcnn.pytorch/outputs/market77_optim/models/best_model.pth \
    --input_pcd /home/s2671222/dgcnn.pytorch/segmentation/realworld_scene/realworld_scene_1.pcd \
    --output_ply /home/s2671222/dgcnn.pytorch/segmentation/results/dgcnn_segment/scene_segmented1_dgcn_best_1.ply \
    --grid_size 0.389 \
    --overlap 0.3
"""