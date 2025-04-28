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

CUDA_VISIBLE_DEVICES=0 python init_geo.py -s /home/sirsh/cv_dataset/dataset_50sites/data/aerial/train/ID0001/p100 -m /home/sirsh/cv_dataset/duster_temp/instant_splat/ID0001 --n_views 0 --focal_avg --co_vis_dsp --conf_aware_ranking --ckpt_path /home/sirsh/aerial_gen/aerial_scene_gen/InstantSplat/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth
\
    > ${MODEL_PATH}/01_init_geo.log 2>&1
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Co-visible Global Geometry Initialization completed. Log saved in ${MODEL_PATH}/01_init_geo.log"


[ ] run mast3r

[ ] run the query pose ransac --- only incase of wriva

[ ] run gsplat

[ ] save stats

```
