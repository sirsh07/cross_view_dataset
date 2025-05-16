from pathlib import Path
import json
import random
import matplotlib.pyplot as plt

def process_street_camera_data(base_path_str, train_size=30, test_size=70):
    base_path = Path(base_path_str)
    id_number = base_path.name.replace("_street", "")  # Extract ID like ID0016 from ID0016_street

    json_path = base_path / f"{id_number}_street.json"
    footage_path = base_path / "footage"

    # Output paths
    train_json_path = base_path / f"{id_number}_street_train.json"
    test_json_path = base_path / f"{id_number}_street_test.json"
    train_txt_path = base_path / f"{id_number}_street_train.txt"
    test_txt_path = base_path / f"{id_number}_street_test.txt"
    top_view_path = base_path / f"{id_number}_street_top_view.png"

    # Load JSON file
    with open(json_path, "r") as f:
        street_data = json.load(f)

    all_frames = street_data["cameraFrames"]
    total_samples = len(all_frames)
    print(f"Available frames: {total_samples}")
    train_size = int(0.3 * total_samples)
    test_size = total_samples - train_size

    indices = list(range(total_samples))
    random.shuffle(indices)
    test_indices = sorted(indices[:test_size])
    train_indices = sorted(indices[test_size:test_size + train_size])

    # Generate new JSON structures with header
    header = {k: v for k, v in street_data.items() if k != "cameraFrames"}
    train_json = {**header, "cameraFrames": [all_frames[i] for i in train_indices]}
    test_json = {**header, "cameraFrames": [all_frames[i] for i in test_indices]}

    # Generate filenames
    def format_filename(idx): return f"{id_number}_{idx:03d}.jpeg"

    train_filenames = [format_filename(i) for i in train_indices]
    test_filenames = [format_filename(i) for i in test_indices]

    # Write to .txt
    with open(train_txt_path, "w") as f:
        for name in train_filenames:
            f.write(name + "\n")
    with open(test_txt_path, "w") as f:
        for name in test_filenames:
            f.write(name + "\n")

    # Save updated JSON
    with open(train_json_path, "w") as f:
        json.dump(train_json, f, indent=2)
    with open(test_json_path, "w") as f:
        json.dump(test_json, f, indent=2)

    # Plotting top view
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    def plot_frames_2d(frames, marker, label):
        xs = [f["position"]["x"] for f in frames]
        ys = [f["position"]["y"] for f in frames]
        ax.scatter(xs, ys, label=label, marker=marker, alpha=0.7)

    plot_frames_2d(train_json["cameraFrames"], "o", "Train")
    plot_frames_2d(test_json["cameraFrames"], "x", "Test")

    ax.set_title(f"{id_number} Street Camera Poses (Top View)")
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
        "top_view_png": str(top_view_path)
    }

# Run the function on provided street path
results = process_street_camera_data(
    "./google_earth/ID0102_street/"
)
results
