# cross_view_dataset


#### dataset directory
```
/home/zhyw86/WorkSpace/google-earth

# sample .txt split file 
/home/zhyw86/WorkSpace/google-earth/sampling/street/random/ID0001_street/ID0001_street_train.txt

# sample data folder
/home/zhyw86/WorkSpace/google-earth/data/street/ID0001_street/footage/


```

### follow this steps
```
[ ] Datasets 

python dataset_stats.py --dir /home/zhyw86/WorkSpace/google-earth --output_csv csv_stats/zen_gearth_d_4_22.csv --output_plot mat_plots/zen_gearth_d_4_22.png


python gen_colmap_metadata.py --base_input_dir /home/sirsh/cv_dataset/dataset_50sites/data/ --base_output_dir /home/sirsh/cv_dataset/dataset_50sites/colmap/metadata/

[ ] Create symlink of data




[ ] different data and severity

[ ] run duster /  mast3r / colmap

[ ] run the query pose ransac --- only incase of wriva

[ ] run gsplat

[ ] save stats

```
