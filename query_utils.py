import os





def get_test_splits(root_dir: str = "/home/zhyw86/WorkSpace/google-earth/sampling"):
    
    street_split = "/home/zhyw86/WorkSpace/google-earth/sampling/street/random/ID0001_street/ID0001_street_test.txt"
    right_split = "/home/zhyw86/WorkSpace/google-earth/sampling/aerial/right/ID0001/ID0001_right_test.txt"
    middle_split = "/home/zhyw86/WorkSpace/google-earth/sampling/aerial/middle/random/ID0001/ID0001_test.txt"
    left_split = "/home/zhyw86/WorkSpace/google-earth/sampling/aerial/left/ID0001/ID0001_left_test.txt"

    
    for ids in os.listdir("/home/sirsh/cv_dataset/dataset_50sites/colmap/metadata/aerial_street/train"):
        
        if not os.path.exists(street_split.replace("ID0001", ids)):
            print(f"street_split: {street_split.replace('ID0001', ids)} not exists")
        if not os.path.exists(right_split.replace("ID0001", ids)):
            print(f"right_split: {right_split.replace('ID0001', ids)} not exists")
        if not os.path.exists(middle_split.replace("ID0001", ids)):
            print(f"middle_split: {middle_split.replace('ID0001', ids)} not exists")
        if not os.path.exists(left_split.replace("ID0001", ids)):
            print(f"left_split: {left_split.replace('ID0001', ids)} not exists")
        

if __name__ == "__main__":
    get_test_splits()

