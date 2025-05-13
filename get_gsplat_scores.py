import os
import pandas as pd
import numpy as np
import json


def get_scores(dir):
    
    models = []
    conds = []
    idxs = []
    splits = []
    scores = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith("val_step29999.json"):
                gsplat_file = os.path.join(root, file)
                
                _, model, cond, _, idx, split, _, _, _ = gsplat_file.rsplit("/",8)
                
                # Extract the score from the filename
                with open(gsplat_file, "rb") as f:
                    scores = json.load(f)
     
                
                import pdb; pdb.set_trace()
    
    
     
    
if __name__ == "__main__":
    # Example usage
    dir = "/home/sirsh/cv_dataset/dataset_50sites/gsplat/"
    scores = get_scores(dir)
    print(scores) 
    