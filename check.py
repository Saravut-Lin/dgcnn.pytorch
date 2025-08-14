import h5py, numpy as np
with h5py.File("/home/s2671222/dgcnn.pytorch/dataset/market77/jam_hartleys_strawberry_300gm_1200_2048_segmentation_20480_12000", 'r') as f:
    xyz = f['seg_points'][0]
    rgb = f['seg_colors'][0]
print("TRAIN xyz min:", xyz.min(0), "max:", xyz.max(0))
print("TRAIN rgb range:", rgb.min(), rgb.max())


import open3d as o3d, numpy as np
pcd   = o3d.io.read_point_cloud("/home/s2671222/dgcnn.pytorch/segmentation/realworld_scene/realworld_scene_1.pcd")
pts   = np.asarray(pcd.points)
valid = ~np.isnan(pts).any(1)
pts   = pts[valid]               # drop NaNs just like infer_pcd()
print("Kept", pts.shape[0], "points out of", len(valid))
print("SCENE xyz min:", pts.min(0), "max:", pts.max(0))
if pcd.has_colors():
    print("SCENE rgb range:", np.asarray(pcd.colors)[valid].min(),
                                  np.asarray(pcd.colors)[valid].max())