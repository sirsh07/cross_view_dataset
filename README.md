# cross_view_dataset


#### dataset directory
```
/home/zhyw86/WorkSpace/google-earth

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


python gen_colmap_metadata.py --base_input_dir /home/sirsh/cv_dataset/dataset_50sites/data/ --base_output_dir /home/sirsh/cv_dataset/dataset_50sites/colmap/metadata/

[X] Create symlink of data


[X] different data and severity

[ ] run colmap

run_colmap.slurm

[ ] run dust3r

CUDA_VISIBLE_DEVICES=0 python -W ignore ./init_geo.py -s ${SOURCE_PATH} -m ${MODEL_PATH} --n_views ${N_VIEW} --focal_avg --co_vis_dsp --conf_aware_ranking

CUDA_VISIBLE_DEVICES=0 python init_geo.py -s /home/sirsh/cv_dataset/dataset_50sites/data/aerial/train/ID0001/p100 -m /home/sirsh/cv_dataset/duster_temp/instant_splat/ID0001 --n_views 4 --focal_avg --co_vis_dsp --conf_aware_ranking --ckpt_path /home/sirsh/aerial_gen/aerial_scene_gen/InstantSplat/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth

CUDA_VISIBLE_DEVICES=0 python init_geo.py -s /home/sirsh/cv_dataset/dataset_50sites/data/aerial/train/ID0001/p100 -m /home/sirsh/cv_dataset/duster_temp/instant_splat/ID0001 --n_views 0 --focal_avg --co_vis_dsp --conf_aware_ranking --ckpt_path /home/sirsh/aerial_gen/aerial_scene_gen/InstantSplat/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth --infer_video


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



