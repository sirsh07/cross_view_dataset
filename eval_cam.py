import os
import numpy as np
# from colmap_loader import read_model  # Replace with your actual COLMAP model reader
from hloc.utils.read_write_model import Camera, Image, Point3D, read_model, write_model, rotmat2qvec, qvec2rotmat

def load_colmap_model(model_path, ext=".bin"):
    """Loads COLMAP cameras and images from the given path."""
    cameras, images, _ = read_model(model_path, ext=ext)
    return cameras, images

def get_camera_poses(images):
    """Returns a dictionary of image name to 4x4 camera-to-world pose matrices."""
    poses = {}
    for image in images.values():
        R = image.qvec2rotmat()
        t = image.tvec.reshape(3, 1)
        cam_to_world = np.eye(4)
        cam_to_world[:3, :3] = R.T
        cam_to_world[:3, 3] = -R.T @ t
        poses[image.name] = cam_to_world
    return poses

def compute_pose_errors(poses1, poses2):
    """Computes translation and rotation errors between two sets of poses."""
    errors = {}
    for name in poses1:
        if name not in poses2:
            continue
        T1 = poses1[name]
        T2 = poses2[name]
        translation_error = np.linalg.norm(T1[:3, 3] - T2[:3, 3])
        rotation_diff = T1[:3, :3].T @ T2[:3, :3]
        cos_theta = np.clip((np.trace(rotation_diff) - 1) / 2, -1.0, 1.0)
        rotation_error = np.degrees(np.arccos(cos_theta))
        errors[name] = {
            'translation_error': translation_error,
            'rotation_error_deg': rotation_error
        }
    return errors

def summarize_errors(errors):
    """Prints mean and std of translation and rotation errors."""
    trans_errors = [e['translation_error'] for e in errors.values()]
    rot_errors = [e['rotation_error_deg'] for e in errors.values()]
    print(f"Translation Error: mean = {np.mean(trans_errors):.4f}, std = {np.std(trans_errors):.4f}")
    print(f"Rotation Error (deg): mean = {np.mean(rot_errors):.4f}, std = {np.std(rot_errors):.4f}")

def main():
    
    cams1, imgs1 = load_colmap_model("/home/sirsh/cv_dataset/dataset_50sites/data/aerial/train/ID0001/ge_metadata/ID0001/")
    
    import pdb; pdb.set_trace()
    
    # cams1, imgs1 = load_colmap_model(model_path_1, ext)
    # cams2, imgs2 = load_colmap_model(model_path_2, ext)

    # poses1 = get_camera_poses(imgs1)
    # poses2 = get_camera_poses(imgs2)

    # errors = compute_pose_errors(poses1, poses2)
    # summarize_errors(errors)

if __name__ == "__main__":
    main()
