import os
import pandas as pd
import numpy as np


def get_scores(dir):
    
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith("val_step29999.json"):
                gsplat_file = os.path.join(root, file)
                import pdb; pdb.set_trace()
    
    
     
    
if __name__ == "__main__":
    # Example usage
    dir = "/home/sirsh/cv_dataset/dataset_50sites/gsplat/"
    scores = get_scores(dir)
    print(scores) 
    