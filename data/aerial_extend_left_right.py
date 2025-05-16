from pathlib import Path

# Base path
base_path = Path("./google_earth/ID0058/")
id_number = base_path.name

# Input and output file paths
train_txt = base_path / f"{id_number}_train.txt"
test_txt = base_path / f"{id_number}_test.txt"
left_train_txt = base_path / f"{id_number}_left_train.txt"
left_test_txt = base_path / f"{id_number}_left_test.txt"
right_train_txt = base_path / f"{id_number}_right_train.txt"
right_test_txt = base_path / f"{id_number}_right_test.txt"

# Function to inject "left" or "right" into filenames
def modify_filenames(input_path, view_label):
    with open(input_path, "r") as f:
        filenames = [line.strip() for line in f if line.strip()]

    modified = []
    for name in filenames:
        parts = name.split("_")
        if len(parts) == 2 and parts[1].endswith(".jpeg"):
            prefix, suffix = parts
            suffix_number = suffix.replace(".jpeg", "")
            new_name = f"{prefix}_{view_label}_{suffix_number}.jpeg"
            modified.append(new_name)
        else:
            print(f"Skipping unexpected format: {name}")
    return modified

# Modify and save for left and right views
left_train = modify_filenames(train_txt, "left")
left_test = modify_filenames(test_txt, "left")
right_train = modify_filenames(train_txt, "right")
right_test = modify_filenames(test_txt, "right")

# Save to respective files
left_train_txt.write_text("\n".join(left_train))
left_test_txt.write_text("\n".join(left_test))
right_train_txt.write_text("\n".join(right_train))
right_test_txt.write_text("\n".join(right_test))

left_train_txt, left_test_txt, right_train_txt, right_test_txt
