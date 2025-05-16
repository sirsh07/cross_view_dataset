from pathlib import Path
import json
import random
import matplotlib.pyplot as plt

def process_camera_data_with_views(base_path_str):
    base_path = Path(base_path_str)
    id_number = base_path.name

    json_path = base_path / f"{id_number}.json"
    footage_path = base_path / "footage"

    # Output paths
    train_json_path = base_path / f"{id_number}_aerial_train.json"
    test_json_path = base_path / f"{id_number}_aerial_test.json"
    train_txt_path = base_path / f"{id_number}_train.txt"
    test_txt_path = base_path / f"{id_number}_test.txt"
    top_view_path = base_path / f"{id_number}_top_view.png"

    left_train_txt = base_path / f"{id_number}_left_train.txt"
    left_test_txt = base_path / f"{id_number}_left_test.txt"
    right_train_txt = base_path / f"{id_number}_right_train.txt"
    right_test_txt = base_path / f"{id_number}_right_test.txt"

    # Load JSON
    with open(json_path, "r") as f:
        aerial_data = json.load(f)

    all_frames = aerial_data["cameraFrames"]

    indices = list(range(len(all_frames)))
    random.shuffle(indices)
    test_indices = sorted(indices[:240])
    train_indices = sorted(indices[240:360])

    header = {k: v for k, v in aerial_data.items() if k != "cameraFrames"}
    train_json = {**header, "cameraFrames": [all_frames[i] for i in train_indices]}
    test_json = {**header, "cameraFrames": [all_frames[i] for i in test_indices]}

    def format_filename(idx): return f"{id_number}_{idx:03d}.jpeg"
    train_filenames = [format_filename(i) for i in train_indices]
    test_filenames = [format_filename(i) for i in test_indices]

    # Save plain filenames
    train_txt_path.write_text("\n".join(train_filenames))
    test_txt_path.write_text("\n".join(test_filenames))

    # Save JSONs
    with open(train_json_path, "w") as f:
        json.dump(train_json, f, indent=2)
    with open(test_json_path, "w") as f:
        json.dump(test_json, f, indent=2)

    # Generate left/right variants
    def inject_view(filenames, view):
        modified = []
        for name in filenames:
            parts = name.split("_")
            if len(parts) == 2 and parts[1].endswith(".jpeg"):
                prefix, suffix = parts
                suffix_number = suffix.replace(".jpeg", "")
                new_name = f"{prefix}_{view}_{suffix_number}.jpeg"
                modified.append(new_name)
            else:
                modified.append(name)  # fallback
        return modified

    left_train = inject_view(train_filenames, "left")
    right_train = inject_view(train_filenames, "right")
    left_test = inject_view(test_filenames, "left")
    right_test = inject_view(test_filenames, "right")

    left_train_txt.write_text("\n".join(left_train))
    right_train_txt.write_text("\n".join(right_train))
    left_test_txt.write_text("\n".join(left_test))
    right_test_txt.write_text("\n".join(right_test))

    # Plotting
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    def plot_frames_2d(frames, marker, label):
        xs = [f["position"]["x"] for f in frames]
        ys = [f["position"]["y"] for f in frames]
        ax.scatter(xs, ys, label=label, marker=marker, alpha=0.7)
    plot_frames_2d(train_json["cameraFrames"], "o", "Train")
    plot_frames_2d(test_json["cameraFrames"], "x", "Test")
    ax.set_title(f"{id_number} Camera Poses (Top View)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    plt.tight_layout()
    plt.savefig(top_view_path)
    plt.close()

    return {
        "train_json": str(train_json_path),
        "test_json": str(test_json_path),
        "train_txt": str(train_txt_path),
        "test_txt": str(test_txt_path),
        "top_view_png": str(top_view_path),
        "left_train_txt": str(left_train_txt),
        "right_train_txt": str(right_train_txt),
        "left_test_txt": str(left_test_txt),
        "right_test_txt": str(right_test_txt),
    }

# Example usage:
results = process_camera_data_with_views("./google_earth/ID0102/")
