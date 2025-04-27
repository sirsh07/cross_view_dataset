import os
import pandas as pd
import argparse
from anytree import Node, RenderTree

# def build_directory_tree(directory_path):
#     root_node = Node(directory_path)
#     nodes = {directory_path: root_node}

#     for root, dirs, files in os.walk(directory_path):
#         parent_node = nodes[root]
#         for d in dirs:
#             dir_path = os.path.join(root, d)
#             nodes[dir_path] = Node(d, parent=parent_node)
#         for f in files:
#             Node(f, parent=parent_node)

#     return root_node

import matplotlib.pyplot as plt

def build_directory_tree_with_file_counts(directory_path):
    stats = []
    root_node = Node(f"{directory_path} (Files: 0)")
    nodes = {directory_path: root_node}

    for root, dirs, files in os.walk(directory_path):
        file_count = len(files)
        parent_node = nodes[root]
        stats.append({
            "directory": root,
            "files_count": file_count
        })
        for d in dirs:
            dir_path = os.path.join(root, d)
            nodes[dir_path] = Node(f"{d} (Files: 0)", parent=parent_node)
        # Update the current directory node with the file count
        parent_node.name = f"{os.path.basename(root) or root} (Files: {file_count})"

    return root_node, stats

def visualize_tree(root_node):
    for pre, _, node in RenderTree(root_node):
        print(f"{pre}{node.name}")

def save_stats_to_csv(stats, output_csv):
    df = pd.DataFrame(stats)
    df.to_csv(output_csv, index=False)
    print(f"Directory statistics saved to {output_csv}")

def plot_stats(stats, output_plot):
    df = pd.DataFrame(stats)
    df = df.sort_values(by="files_count", ascending=False)  # Sort by file count for better visualization

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.barh(df["directory"], df["files_count"], color="skyblue")
    plt.xlabel("Number of Files")
    plt.ylabel("Directories")
    plt.title("File Counts in Directories")
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_plot)
    print(f"Plot saved to {output_plot}")
    # plt.show()

def main():
    parser = argparse.ArgumentParser(description="Generate directory tree, save stats to CSV, and plot file counts.")
    parser.add_argument("--dir", type=str, required=True, help="Path to the directory")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to save the CSV file")
    parser.add_argument("--output_plot", type=str, required=True, help="Path to save the plot image")
    args = parser.parse_args()

    # Build directory tree and collect stats
    print(f"Building directory tree for: {args.dir}")
    root_node, stats = build_directory_tree_with_file_counts(args.dir)

    # Visualize the tree
    visualize_tree(root_node)

    # Save stats to CSV
    save_stats_to_csv(stats, args.output_csv)

    # Plot the stats
    plot_stats(stats, args.output_plot)

if __name__ == "__main__":
    main()