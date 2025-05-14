import os
import pandas as pd
import numpy as np
import json


def get_scores(dir, output_file="gsplat_scores.csv"):
    
    models = []
    conds = []
    idxs = []
    splits = []
    scores = []
    
    psnrs = []
    ssims = []
    lpipss = []
    dreamsims = []
    num_GSs = []
    
    
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith("val_step29999.json"):
                gsplat_file = os.path.join(root, file)
                
                import pdb; pdb.set_trace()
                
                _, model, cond, _, idx, split, _, _, _ = gsplat_file.rsplit("/",8)
                
                # _, model, cond, _, idx, split, _, _, _, _ = gsplat_file.rsplit("/",9)
                
                # Extract the score from the filename
                with open(gsplat_file, "rb") as f:
                    scores = json.load(f)
                    
                models.append(model)
                conds.append(cond)
                idxs.append(idx)
                splits.append(split)
                psnrs.append(scores["psnr"])
                ssims.append(scores["ssim"])
                lpipss.append(scores["lpips"])
                dreamsims.append(scores["dreamsim"])
                num_GSs.append(scores["num_GS"])
     
    df = pd.DataFrame({
        "model": models,
        "cond": conds,
        "idx": idxs,
        "split": splits,
        "psnr": psnrs,
        "ssim": ssims,
        "lpips": lpipss,
        "dreamsim": dreamsims,
        "num_GS": num_GSs
    })
    
    df.to_csv(output_file, index=False)
    
     
    
if __name__ == "__main__":
    # Example usagse
    # dir = "/home/sirsh/cv_dataset/dataset_50sites/gsplat/"
    # scores = get_scores(dir,"./recon_scores/colmap_gsplat_scores.csv")
    
    # dir = "/home/sirsh/cv_dataset/dataset_50sites/gsplat_mast3r"
    # dir = "/home/smitra/cv_dataset/dataset_50sites/gsplat"
    # scores = get_scores(dir,"./recon_scores/mast3r_gsplatmcmc_scores.csv")
    
    dir = "/home/smitra/cv_dataset/dataset_50sites/gsplat_mcmc_mast3r_sfm"
    scores = get_scores(dir,"./recon_scores/mast3r_sfm_gsplatmcmc_scores.csv")
    
    print(scores) 
    