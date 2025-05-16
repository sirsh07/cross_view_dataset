# cross_view_dataset
This repository accompanies our paper:

> **Varying Altitude Dataset**  
> (*Under Review at NeurIPS 2025 data track)
## 📦 Dataset Access

You can access the dataset from the following sources:

- 🔗 [Download from Hugging Face](https://huggingface.co/datasets/letsGoBlind/Varying_Altitude_Dataset)
- 🔗 [Download from Kaggle](https://www.kaggle.com/datasets/zhyw86/varying-altitude-dataset/)

Both versions contain the same content: satellite, aerial, and ground-level imagery packaged in ZIP batches.

## 📁 Repository Structure

```bash
.
├── metadata.jsonld        # Croissant-compliant dataset metadata
├── Batch1.zip             # 10 sites of full multi-scale data
├── Batch2.zip             # 10 sites
├── Batch3.zip             # 10 sites
├── ID0102.zip             # A single-site example (small and easy to examine)
└── README.md              # This file


##Dataset Structure

# sample .txt split file 
/home/zhyw86/WorkSpace/google-earth/sampling/street/random/ID0001_street/ID0001_street_train.txt

# sample data folder
/home/zhyw86/WorkSpace/google-earth/data/street/ID0001_street/footage/

# sample image path
/home/sirsh/cv_dataset/dataset_50sites/data/aerial/train/ID0001/p100/images


```

### follow this steps
```
[X] Datasets 

python dataset_stats.py --dir /home/zhyw86/WorkSpace/google-earth --output_csv csv_stats/zen_gearth_d_4_22.csv --output_plot mat_plots/zen_gearth_d_4_22.png



1. python dataset_gen.py
2. python gen_colmap_metadata.py --base_input_dir /home/sirsh/cv_dataset/dataset_30sites/data/ --base_output_dir /home/sirsh/cv_dataset/dataset_30sites/colmap/metadata/
3. run_colmap.slurm (make necessary changes to the file)
4. run_mast3r.slurm (make necessary changes to the file)
5. 




python gen_colmap_metadata.py --base_input_dir /home/sirsh/cv_dataset/dataset_50sites/data/ --base_output_dir /home/sirsh/cv_dataset/dataset_50sites/colmap/metadata/

[X] Create symlink of data


[X] different data and severity

[ ] run colmap

run_colmap.slurm

[ ] run dust3r

CUDA_VISIBLE_DEVICES=0 python -W ignore ./init_geo.py -s ${SOURCE_PATH} -m ${MODEL_PATH} --n_views ${N_VIEW} --focal_avg --co_vis_dsp --conf_aware_ranking

CUDA_VISIBLE_DEVICES=0 python init_geo.py -s /home/sirsh/cv_dataset/dataset_50sites/data/aerial/train/ID0001/p100 -m /home/sirsh/cv_dataset/duster_temp/instant_splat/ID0001 --n_views 4 --focal_avg --co_vis_dsp --conf_aware_ranking --ckpt_path /home/sirsh/aerial_gen/aerial_scene_gen/InstantSplat/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth

CUDA_VISIBLE_DEVICES=0 python init_geo.py -s /home/sirsh/cv_dataset/dataset_50sites/data/aerial_street/train/ID0001/p100 -m /home/sirsh/cv_dataset/duster_temp/instant_splat/ID0001 --n_views 0 --focal_avg --co_vis_dsp --conf_aware_ranking --ckpt_path /home/sirsh/aerial_gen/aerial_scene_gen/InstantSplat/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth --infer_video




CUDA_VISIBLE_DEVICES=0 python init_geo.py -s /home/sirsh/cv_dataset/dataset_50sites/data/aerial_street/train/ID0001/p100 -m /home/sirsh/cv_dataset/duster_temp/instant_splat_2/ID0001 --n_views 0 --focal_avg --co_vis_dsp --conf_aware_ranking --ckpt_path /home/sirsh/aerial_gen/aerial_scene_gen/InstantSplat/mast3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth --infer_video

/home/sirsh/cv_dataset/dataset_30sites/data/aerial_street/train/ID0001/p100

\
    > ${MODEL_PATH}/01_init_geo.log 2>&1
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Co-visible Global Geometry Initialization completed. Log saved in ${MODEL_PATH}/01_init_geo.log"


[ ] run mast3r


python make_pairs.py --dir /home/sirsh/cv_dataset/dataset_50sites/data/aerial/train/ID0001/p100/images --output /home/sirsh/cv_dataset/mastrf_temp/instant_splat/ID0001/pairs.txt --weights ./checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth --scene_graph complete
--retrieval_model ./checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth --scene_graph complete


python make_pairs.py --dir /home/sirsh/cv_dataset/dataset_50sites/data/aerial/train/ID0001/p100/images --output /home/sirsh/cv_dataset/mastrf_temp/instant_splat/ID0001/pairs.txt --scene_graph complete --weights /home/sirsh/aerial_gen/aerial_scene_gen/master_sfm/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth

python kapture_mast3r_mapping.py --weights /home/sirsh/aerial_gen/aerial_scene_gen/master_sfm/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth --dir_same_camera /home/sirsh/cv_dataset/dataset_50sites/data/aerial/train/ID0001/p100/images --output /home/sirsh/cv_dataset/mastrf_temp/instant_splat/ID0001/output/ --pairsfile_path /home/sirsh/cv_dataset/mastrf_temp/instant_splat/ID0001/pairs.txt


python kapture_mast3r_mapping.py --weights ./checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth --dir_same_camera /home/sirsh/cv_gen_eccv25/ucfwr_data/colmap_data_v2025v2/t4v7s1r2/av/output/undistorted_images_folder/ --output ./results/recon/t4v7s1r2/output/ --pairsfile_path ./wriva_pairs_t4v7s1r2.txt




python make_pairs.py --dir /home/sirsh/cv_dataset/dataset_50sites/data/aerial/train/ID0003/p12/images --output /home/sirsh/cv_dataset/mastrf_temp/instant_splat/ID0003/pairs.txt --scene_graph complete --weights /home/sirsh/aerial_gen/aerial_scene_gen/master_sfm/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth

python kapture_mast3r_mapping.py --weights /home/sirsh/aerial_gen/aerial_scene_gen/master_sfm/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth --dir_same_camera /home/sirsh/cv_dataset/dataset_50sites/data/aerial/train/ID0003/p12/images --output /home/sirsh/cv_dataset/mastrf_temp/instant_splat/ID0003/output/ --pairsfile_path /home/sirsh/cv_dataset/mastrf_temp/instant_splat/ID0003/pairs.txt


from hloc.utils.read_write_model import Camera, Image, Point3D, read_model, write_model, rotmat2qvec, qvec2rotmat
cameras, images, _ = read_model("./ID0003_right", ext=".bin")




[ ] run the query pose ransac --- only incase of wriva

[ ] run gsplat


CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default --data_dir /home/smitra/cv_dataset/dataset_50sites/colmap/results/aerial/train/ID0001/p100/output --result_dir /home/smitra/cv_dataset/dataset_50sites/gsplat/aerial/train/ID0001/p100/output --no_normalize_world_space 


CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default --data_dir /home/smitra/cv_dataset/dataset_50sites/colmap/results/aerial/train/ID0001/p100/output --result_dir /home/smitra/cv_dataset/gsplat_temp/aerial/train/ID0001/p100/output --no_normalize_world_space


CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default --data_dir /home/sirsh/cv_dataset/dataset_50sites/colmap/results/aerial/train/ID0001/p100/output --result_dir /home/sirsh/cv_dataset/gsplat_temp/aerial/train/ID0001/p100/output --no_normalize_world_space

[ ] save stats

```

error with mast3r

```
/home/sirsh/cv_dataset/dataset_50sites/mast3r/results/aerial_street/train/ID0007/p12/output/reconstruction
/home/sirsh/cv_dataset/dataset_50sites/mast3r/results/street/train/ID0004/p12/output/reconstruction
/home/sirsh/cv_dataset/dataset_50sites/mast3r/results/street/train/ID0003/p25/output/reconstruction
/home/sirsh/cv_dataset/dataset_50sites/mast3r/results/street/train/ID0003/p12/output/reconstruction
/home/sirsh/cv_dataset/dataset_50sites/mast3r/results/street/train/ID0003/p50/output/reconstruction
/home/sirsh/cv_dataset/dataset_50sites/mast3r/results/street/train/ID0007/p25/output/reconstruction
/home/sirsh/cv_dataset/dataset_50sites/mast3r/results/street/train/ID0007/p12/output/reconstruction
/home/sirsh/cv_dataset/dataset_50sites/mast3r/results/street/train/ID0013/p12/output/reconstruction
/home/sirsh/cv_dataset/dataset_50sites/mast3r/results/street/train/ID0013/p25/output/reconstruction
/home/sirsh/cv_dataset/dataset_50sites/mast3r/results/street/train/ID0008/p12/output/reconstruction
/home/sirsh/cv_dataset/dataset_50sites/mast3r/results/street/train/ID0016/p12/output/reconstruction
/home/sirsh/cv_dataset/dataset_50sites/mast3r/results/street/train/ID0005/p25/output/reconstruction
/home/sirsh/cv_dataset/dataset_50sites/mast3r/results/street/train/ID0005/p12/output/reconstruction
/home/sirsh/cv_dataset/dataset_50sites/mast3r/results/street/train/ID0005/p50/output/reconstruction
/home/sirsh/cv_dataset/dataset_50sites/mast3r/results/street/train/ID0001/p12/output/reconstruction
/home/sirsh/cv_dataset/dataset_50sites/mast3r/results/street/train/ID0012/p12/output/reconstruction
/home/sirsh/cv_dataset/dataset_50sites/mast3r/results/street/train/ID0015/p12/output/reconstruction
/home/sirsh/cv_dataset/dataset_50sites/mast3r/results/aerial/train/ID0007/p25/output/reconstruction
```




python3 run_wriva_reconstruction.py --input_images_folder "/home/sirsh/cv_dataset/dataset_30sites/data/street/train/ID0067/p25/images" --workspace_folder "/home/sirsh/cv_dataset/dataset_30sites/colmap/results/street/train/ID0067/p25/workspace/" --output_folder "/home/sirsh/cv_dataset/dataset_30sites/colmap/results//street/train/ID0067/p25/output" --enable_doppelganger_threshold 200 --max_images 500 --num_threads 6 --input_metadata_folder "/home/sirsh/cv_dataset/dataset_30sites/colmap/metadata//street/train/ID0067/p25/images" --ground_truth_mode


python3 run_wriva_reconstruction.py --input_images_folder "/home/sirsh/cv_dataset/dataset_30sites/data/street/train/ID0003/p12/images" --workspace_folder "/home/sirsh/cv_dataset/dataset_30sites/colmap/results//street/train/ID0003/p12/workspace/" --output_folder "/home/sirsh/cv_dataset/dataset_30sites/colmap/results//street/train/ID0003/p12/output" --enable_doppelganger_threshold 200 --max_images 500 --num_threads 6 --input_metadata_folder "/home/sirsh/cv_dataset/dataset_30sites/colmap/metadata//street/train/ID0003/p12/images" --ground_truth_mode


rsync -avzh --copy-links sirsh@crcv2.eecs.ucf.edu:/home/sirsh/cv_dataset/dataset_30sites/data .



<!-- ssh sirsh@crcv2.eecs.ucf.edu:/home/sirsh/cv_dataset/dataset_30sites/data 'cd /home/sirsh/cv_dataset/dataset_50sites/mast3r && find results -type d -path "*/output/reconstruction/0"' | rsync -avR --files-from=- sirsh@remote:/home/sirsh/cv_dataset/dataset_50sites/mast3r/ /your/local/target/ -->

<!-- rsync -avzh --include="**/reconstruction/**" --exclude="**/images/**"  sirsh@crcv2.eecs.ucf.edu:/home/sirsh/cv_dataset/dataset_50sites/colmap . -->
rsync -avzh --include="**/output/**" --exclude="**/workspace/**"  sirsh@crcv2.eecs.ucf.edu:/home/sirsh/cv_dataset/dataset_50sites/colmap .

rsync -avzh --include="*/" --include="*/reconstruction/**" --exclude="*/reconstruction/images/**" --exclude="*"  sirsh@crcv2.eecs.ucf.edu:/home/sirsh/cv_dataset/dataset_50sites/mast3r .


rsync -avzh --include="*/" --include="*/reconstruction/**" --exclude="*"  sirsh@crcv2.eecs.ucf.edu:/home/sirsh/cv_dataset/dataset_50sites/mast3r .



python make_pairs.py --dir "/home/sirsh/cv_dataset/satellite/aerial_street_ID0001_p50" --output "/home/sirsh/cv_dataset/satellite/master_sfm2/pairs.txt" --weights /home/sirsh/aerial_gen/aerial_scene_gen/master_sfm/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth --scene_graph complete

python kapture_mast3r_mapping.py --weights /home/sirsh/aerial_gen/aerial_scene_gen/master_sfm/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth --dir_same_camera "/home/sirsh/cv_dataset/satellite/aerial_street_ID0001_p50/" --output "/home/sirsh/cv_dataset/satellite/master_sfm2/" --pairsfile_path "/home/sirsh/cv_dataset/satellite/master_sfm2/pairs.txt"



python3 run_wriva_reconstruction.py --input_images_folder "$input_folder" --workspace_folder "$workspace_folder" --output_folder "$output_folder" --enable_doppelganger_threshold 200 --max_images 500 --num_threads 6 --input_metadata_folder "$metadata_folder" --ground_truth_mode



python3 run_wriva_reconstruction.py --input_images_folder "/home/sirsh/cv_dataset/satellite/aerial_street_ID0001_p50" --workspace_folder "/home/sirsh/cv_dataset/satellite/colmap/workspace" --output_folder "/home/sirsh/cv_dataset/satellite/colmap/output" --enable_doppelganger_threshold 200 --max_images 500 --num_threads 6 --input_metadata_folder "/home/sirsh/cv_dataset/satellite/aerial_street_ID0001_p50_metadata" --ground_truth_mode





/home/sirsh/cv_dataset/satellite_v2

python make_pairs.py --dir "/home/sirsh/cv_dataset/satellite_v2/street_ID0001_p50" --output "/home/sirsh/cv_dataset/satellite_v2/master_sfm2/pairs.txt" --weights /home/sirsh/aerial_gen/aerial_scene_gen/master_sfm/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth --scene_graph complete

python kapture_mast3r_mapping.py --weights /home/sirsh/aerial_gen/aerial_scene_gen/master_sfm/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth --dir_same_camera "/home/sirsh/cv_dataset/satellite_v2/street_ID0001_p50/" --output "/home/sirsh/cv_dataset/satellite_v2/master_sfm2/" --pairsfile_path "/home/sirsh/cv_dataset/satellite_v2/master_sfm2/pairs.txt"




python3 run_wriva_reconstruction.py --input_images_folder "/home/sirsh/cv_dataset/satellite_v2/street_ID0001_p50" --workspace_folder "/home/sirsh/cv_dataset/satellite_v2/colmap/workspace" --output_folder "/home/sirsh/cv_dataset/satellite_v2/colmap/output" --enable_doppelganger_threshold 200 --max_images 500 --num_threads 6 --input_metadata_folder "/home/sirsh/cv_dataset/satellite_v2/street_ID0001_p50_metadata" --ground_truth_mode

