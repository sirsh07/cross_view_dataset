import os
import numpy as np
# from colmap_loader import read_model  # Replace with your actual COLMAP model reader
from hloc.utils.read_write_model import Camera, Image, Point3D, read_model, write_model, rotmat2qvec, qvec2rotmat
from utils.colmap_utils import get_camera_matrix
import torch
from utils.metric import camera_to_rel_deg

def load_colmap_data(colmap_data_folder):
    # print('Loading COLMAP data...')
    input_format = '.bin'
    cameras, images, _ = read_model(colmap_data_folder, ext=input_format)
    # print(f'num_cameras: {len(cameras)}')
    # print(f'num_images: {len(images)}')
    # print(f'num_points3D: {len(points3D)}')
    

    colmap_pose_dict = {}

    # Loop through COLMAP images
    for img_id, img_info in images.items():
        img_name = img_info.name

        C_R_G, C_t_G = qvec2rotmat(img_info.qvec), img_info.tvec

        # invert
        G_t_C = -C_R_G.T @ C_t_G
        G_R_C = C_R_G.T

        cam_info = cameras[img_info.camera_id]
        cam_params = cam_info.params
        K, _ = get_camera_matrix(camera_params=cam_params, camera_model=cam_info.model)

        colmap_pose_dict[img_name] = (K, G_R_C, G_t_C)

    return colmap_pose_dict, _



def compute_pose_errors(gt_poses_dict, pred_poses_dict):
    """Computes translation and rotation errors between two sets of poses."""
    
    test_imgs = list(pred_poses_dict.keys())
    
    gt_poses = []
    pred_poses = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for img in test_imgs:
        
        gt_pose = gt_poses_dict[img]
        pred_pose = pred_poses_dict[img]
        
        gt_R, gt_T = gt_pose[1], gt_pose[2]
        pred_R, pred_T = pred_pose[1], pred_pose[2]
        
        gt_m = np.eye(4)
        gt_m[:3, :4] = np.hstack((gt_R, gt_T.reshape(3, 1)))
        
        pred_m = np.eye(4)
        pred_m[:3, :4] = np.hstack((pred_R, pred_T.reshape(3, 1)))
        
        gt_poses.append(gt_m)
        pred_poses.append(pred_m)
        
    gt_poses = np.array(gt_poses, dtype=np.float32)
    pred_poses = np.array(pred_poses, dtype=np.float32)
    
    gt_poses = torch.from_numpy(gt_poses).to(device)
    pred_poses = torch.from_numpy(pred_poses).to(device)
    
    rel_rangle_deg, rel_tangle_deg = camera_to_rel_deg(pred_poses, gt_poses, device, batch_size=1)
    
    print(f"    --  Mean Rot   Error (Deg) for this scene: {rel_rangle_deg.mean():10.2f}")
    print(f"    --  Mean Trans Error (Deg) for this scene: {rel_tangle_deg.mean():10.2f}")
    
    import pdb; pdb.set_trace()

    rel_rangle_deg = rel_rangle_deg.cpu().numpy()
    # assert rel_rangle_deg.shape[0] == 1
    rel_tangle_deg = rel_tangle_deg.cpu().numpy()[-1]

    rError = float(rel_rangle_deg)
    tError = float(rel_tangle_deg)
    
    Racc_5 = np.mean(rError < 5) * 100
    Racc_10 = np.mean(rError < 10) * 100
    Racc_15 = np.mean(rError < 15) * 100
    Racc_30 = np.mean(rError < 30) * 100

    Tacc_5 = np.mean(tError < 5) * 100
    Tacc_10 = np.mean(tError < 10) * 100
    Tacc_15 = np.mean(tError < 15) * 100
    Tacc_30 = np.mean(tError < 30) * 100

    print(f"RRA @ 5 deg: {Racc_5:10.2f}")
    print(f"RRA @ 10 deg: {Racc_10:10.2f}")
    print(f"RRA @ 15 deg: {Racc_15:10.2f}")
    print(f"RRA @ 30 deg: {Racc_30:10.2f}")
    print('---------------------------------')
    print(f"RTA @ 5 deg: {Tacc_5:10.2f}")
    print(f"RTA @ 10 deg: {Tacc_10:10.2f}")
    print(f"RTA @ 15 deg: {Tacc_15:10.2f}")
    print(f"RTA @ 30 deg: {Tacc_30:10.2f}")
    
    import pdb; pdb.set_trace()

def summarize_errors(errors):
    """Prints mean and std of translation and rotation errors."""
    trans_errors = [e['translation_error'] for e in errors.values()]
    rot_errors = [e['rotation_error_deg'] for e in errors.values()]
    print(f"Translation Error: mean = {np.mean(trans_errors):.4f}, std = {np.std(trans_errors):.4f}")
    print(f"Rotation Error (deg): mean = {np.mean(rot_errors):.4f}, std = {np.std(rot_errors):.4f}")

def main():
    
    poses1 = load_colmap_data("/home/sirsh/cv_dataset/dataset_50sites/data/aerial/train/ID0001/ge_metadata/ID0001/")
    poses2 = load_colmap_data("/home/sirsh/cv_dataset/dataset_50sites/data/aerial/train/ID0001/ge_metadata/ID0001_right/")
    poses3 = load_colmap_data("/home/sirsh/cv_dataset/dataset_50sites/data/aerial/train/ID0001/ge_metadata/ID0001_left/")
    
    poses = {
        **poses1[0],
        **poses2[0],
        **poses3[0]
    }
    
    pred_poses = load_colmap_data("/home/sirsh/cv_dataset/dataset_50sites/colmap/results/aerial/train/ID0001/p100/output/sparse")
    
    compute_pose_errors(poses, pred_poses[0])
    
    
    # cams1, imgs1 = load_colmap_model(model_path_1, ext)
    # cams2, imgs2 = load_colmap_model(model_path_2, ext)

    # poses1 = get_camera_poses(imgs1)
    # poses2 = get_camera_poses(imgs2)

    # errors = compute_pose_errors(poses1, poses2)
    # summarize_errors(errors)

if __name__ == "__main__":
    main()
