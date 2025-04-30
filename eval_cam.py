import os
import numpy as np
# from colmap_loader import read_model  # Replace with your actual COLMAP model reader
from hloc.utils.read_write_model import Camera, Image, Point3D, read_model, write_model, rotmat2qvec, qvec2rotmat
from utils.colmap_utils import get_camera_matrix
import torch
from utils.metric import camera_to_rel_deg
import glob
import pandas as pd
import tqdm


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

def compute_pose_errors(gt_poses_dict, pred_poses_dict, log_name):
    """Computes translation and rotation errors between two sets of poses."""
    
    test_imgs = list(pred_poses_dict.keys())
    
    gt_poses = []
    pred_poses = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for img in test_imgs:
        
        try:
            if log_name.split("_",1)[0] == "street":
                img_name = img.split("_")
                img2 = "_".join([img_name[0], "street", img_name[1]])
                gt_pose = gt_poses_dict[img2]
            else:
                gt_pose = gt_poses_dict[img]
        except KeyError:
            print(f"Image {img} not found in ground truth poses for {log_name}.")
            import pdb; pdb.set_trace()
        
        try:
            pred_pose = pred_poses_dict[img]
        except KeyError:
            print(f"Image {img} not found in predicted poses for {log_name}.")
            import pdb; pdb.set_trace()
        
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
    
    rError = rel_rangle_deg.cpu().numpy()
    # assert rel_rangle_deg.shape[0] == 1
    # rel_tangle_deg = rel_tangle_deg.cpu().numpy()[0]
    tError = rel_tangle_deg.cpu().numpy().squeeze(-1)

    # rError = float(rel_rangle_deg)
    # tError = float(rel_tangle_deg)
    
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
    
    return {
        'rotation_error_deg': rel_rangle_deg.mean().cpu().numpy(),
        'translation_error': rel_tangle_deg.mean().cpu().numpy(),
        'Racc_5': Racc_5,
        'Racc_10': Racc_10,
        'Racc_15': Racc_15,
        'Racc_30': Racc_30,
        'Tacc_5': Tacc_5,
        'Tacc_10': Tacc_10,
        'Tacc_15': Tacc_15,
        'Tacc_30': Tacc_30
    }
    

def get_all_colmap_folders(base_path):
    """
    Recursively finds all 'sparse' directories under the given base COLMAP directory.

    Args:
        base_path (str): Path like /home/.../colmap

    Returns:
        List[str]: Sorted list of full paths to 'sparse' folders
    """
    pattern = os.path.join(base_path, "results/*/train/ID*/p*/output/sparse")
    sparse_dirs = glob.glob(pattern)
    return sorted(sparse_dirs)


def get_all_mast3r_folders(base_path):
    """
    Recursively finds all 'sparse' directories under the given base COLMAP directory.

    Args:
        base_path (str): Path like /home/.../colmap

    Returns:
        List[str]: Sorted list of full paths to 'sparse' folders
    """
    pattern = os.path.join(base_path, "results/*/train/ID*/p*/output/reconstruction/0")
    sparse_dirs = glob.glob(pattern)
    return sorted(sparse_dirs)

def summarize_errors(errors):
    """Prints mean and std of translation and rotation errors."""
    trans_errors = [e['translation_error'] for e in errors.values()]
    rot_errors = [e['rotation_error_deg'] for e in errors.values()]
    print(f"Translation Error: mean = {np.mean(trans_errors):.4f}, std = {np.std(trans_errors):.4f}")
    print(f"Rotation Error (deg): mean = {np.mean(rot_errors):.4f}, std = {np.std(rot_errors):.4f}")

def main():
    
    # poses1 = load_colmap_data("/home/sirsh/cv_dataset/dataset_50sites/data/aerial/train/ID0001/ge_metadata/ID0001/")
    # poses2 = load_colmap_data("/home/sirsh/cv_dataset/dataset_50sites/data/aerial/train/ID0001/ge_metadata/ID0001_right/")
    # poses3 = load_colmap_data("/home/sirsh/cv_dataset/dataset_50sites/data/aerial/train/ID0001/ge_metadata/ID0001_left/")
    
    # poses = {
    #     **poses1[0],
    #     **poses2[0],
    #     **poses3[0]
    # }
    
    # pred_poses = load_colmap_data("/home/sirsh/cv_dataset/dataset_50sites/colmap/results/aerial/train/ID0001/p100/output/sparse")
    
    
    # compute_pose_errors(poses, pred_poses[0])
    
    # if os.path.exists("./cache_files/colmap_folders.txt"):
    #     with open("./cache_files/colmap_folders.txt", "r") as f:
    #         colmap_folders = f.read().splitlines()
    # else:
    #     colmap_folders = get_all_colmap_folders("/home/sirsh/cv_dataset/dataset_50sites/colmap")
    #     with open("./cache_files/colmap_folders.txt", "w") as f:
    #         f.write("\n".join(colmap_folders))
        
    if os.path.exists("./cache_files/master_folders.txt"):
        with open("./cache_files/master_folders.txt", "r") as f:
            master_folders = f.read().splitlines()
    else:
        master_folders = get_all_mast3r_folders("/home/sirsh/cv_dataset/dataset_50sites/master")
        with open("./cache_files/master_folders.txt", "w") as f:
            f.write("\n".join(master_folders))
        
    
    model_names = []
    model_paths = []
    setup_names = []
    site_ids = []
    annots = []
    registration_stats = []
    rotation_error_deg = []
    translation_error = []
    Racc_5 = []
    Racc_10 = []
    Racc_15 = []
    Racc_30 = []
    Tacc_5 = []
    Tacc_10 = []
    Tacc_15 = []
    Tacc_30 = []
    
    # for colmap_folder in colmap_folders:
    # for colmap_folder in tqdm.tqdm(colmap_folders, desc="Processing COLMAP folders"):
    #     # print(f"Processing COLMAP folder: {colmap_folder}")
    #     try:
    #         pred_poses = load_colmap_data(colmap_folder)
    #     except:
    #         print(f"Error loading COLMAP data from {colmap_folder}. Skipping...")
    #         continue
        
    #     _, setup, _, site_id, annot, _, _ = colmap_folder.rsplit("/",6)
        
    #     metadata_folder = os.path.join("/home/sirsh/cv_dataset/dataset_50sites/data", setup, "train", site_id, "ge_metadata")
    #     meta_folders = os.listdir(metadata_folder)
        
    #     data_folder = os.path.join("/home/sirsh/cv_dataset/dataset_50sites/data", setup, "train", site_id, annot, "images")
    #     num_images = len(os.listdir(data_folder))
    #     num_registered_images = len(list(pred_poses[0].keys()))
        
    #     gt_poses = {}
        
    #     for meta_folder in meta_folders:
    #         pose_folder = os.path.join(metadata_folder, meta_folder)
    #         pose_data = load_colmap_data(pose_folder)
    #         gt_poses.update(pose_data[0])  
            
    #     errors = compute_pose_errors(gt_poses, pred_poses[0],f"{setup}_{site_id}_{annot}")
        
    #     model_names.append("colmap")
    #     model_paths.append(colmap_folder)
    #     setup_names.append(setup)
    #     site_ids.append(site_id)
    #     annots.append(annot)
    #     registration_stats.append(f"{str(num_registered_images).zfill(3)}/{str(num_images).zfill(3)}")
    #     rotation_error_deg.append(errors['rotation_error_deg'])
    #     translation_error.append(errors['translation_error'])
    #     Racc_5.append(errors['Racc_5'])
    #     Racc_10.append(errors['Racc_10'])
    #     Racc_15.append(errors['Racc_15'])
    #     Racc_30.append(errors['Racc_30'])
    #     Tacc_5.append(errors['Tacc_5'])
    #     Tacc_10.append(errors['Tacc_10'])
    #     Tacc_15.append(errors['Tacc_15'])
    #     Tacc_30.append(errors['Tacc_30'])
        
    
    for master_folder in tqdm.tqdm(master_folders, desc="Processing Master folders"):
        # print(f"Processing Master folder: {master_folder}")
        try:
            mast3r_poses = load_colmap_data(master_folder)
        except:
            print(f"Error loading MAST3R data from {master_folder}. Skipping...")
            continue
        
        _, setup, _, site_id, annot, _, _ = master_folder.rsplit("/",6)
        
        import pdb; pdb.set_trace()
        
        metadata_folder = os.path.join("/home/sirsh/cv_dataset/dataset_50sites/data", setup, "train", site_id, "ge_metadata")
        meta_folders = os.listdir(metadata_folder)
        
        data_folder = os.path.join("/home/sirsh/cv_dataset/dataset_50sites/data", setup, "train", site_id, annot, "images")
        num_images = len(os.listdir(data_folder))
        num_registered_images = len(list(mast3r_poses[0].keys()))
        
        gt_poses = {}
        
        for meta_folder in meta_folders:
            pose_folder = os.path.join(metadata_folder, meta_folder)
            pose_data = load_colmap_data(pose_folder)
            gt_poses.update(pose_data[0])  
            
        errors = compute_pose_errors(gt_poses, mast3r_poses[0],f"{setup}_{site_id}_{annot}")
        
        model_names.append("mast3r")
        model_paths.append(master_folder)
        setup_names.append(setup)
        site_ids.append(site_id)
        annots.append(annot)
        registration_stats.append(f"{str(num_registered_images).zfill(3)}/{str(num_images).zfill(3)}")
        rotation_error_deg.append(errors['rotation_error_deg'])
        translation_error.append(errors['translation_error'])
        Racc_5.append(errors['Racc_5'])
        Racc_10.append(errors['Racc_10'])
        Racc_15.append(errors['Racc_15'])
        Racc_30.append(errors['Racc_30'])
        Tacc_5.append(errors['Tacc_5'])
        Tacc_10.append(errors['Tacc_10'])
        Tacc_15.append(errors['Tacc_15'])
        Tacc_30.append(errors['Tacc_30'])
    
        
    results_dict = {
        'model_name': model_names,
        'model_path': model_paths,
        'setup_names': setup_names,
        'site_ids': site_ids,
        'annots': annots,
        'registration_stats': registration_stats,
        'rotation_error_deg': rotation_error_deg,
        'translation_error': translation_error,
        'Racc_5': Racc_5,
        'Racc_10': Racc_10,
        'Racc_15': Racc_15,
        'Racc_30': Racc_30,
        'Tacc_5': Tacc_5,
        'Tacc_10': Tacc_10,
        'Tacc_15': Tacc_15,
        'Tacc_30': Tacc_30
    }
    
    # Save the results to a CSV file
    # pd.DataFrame(results_dict).to_csv("./cache_files/colmap_results.csv", index=False)
    pd.DataFrame(results_dict).to_csv("./cache_files/master_results.csv", index=False)
    
        
    # for master_folder in master_folders:
    #     print(f"Processing Master folder: {master_folder}")
    #     pred_poses = load_colmap_data(master_folder)
        
        # compute_pose_errors(colmap_poses[0], mast3r_poses[0])
    
    # cams1, imgs1 = load_colmap_model(model_path_1, ext)
    # cams2, imgs2 = load_colmap_model(model_path_2, ext)

    # poses1 = get_camera_poses(imgs1)
    # poses2 = get_camera_poses(imgs2)

    # errors = compute_pose_errors(poses1, poses2)
    # summarize_errors(errors)

if __name__ == "__main__":
    main()
