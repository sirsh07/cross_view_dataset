import os
import pandas as pd
import argparse
from anytree import Node, RenderTree

def build_directory_tree(directory_path):
    root_node = Node(directory_path)
    nodes = {directory_path: root_node}

    for root, dirs, files in os.walk(directory_path):
        parent_node = nodes[root]
        for d in dirs:
            dir_path = os.path.join(root, d)
            nodes[dir_path] = Node(d, parent=parent_node)
        for f in files:
            Node(f, parent=parent_node)

    return root_node

def visualize_tree(root_node):
    for pre, _, node in RenderTree(root_node):
        print(f"{pre}{node.name}")

def main():
    parser = argparse.ArgumentParser(description="Generate and visualize directory tree.")
    parser.add_argument("--dir", type=str, help="Path to the directory")
    args = parser.parse_args()

    # Build and visualize the directory tree
    print(f"Building directory tree for: {args.dir}")
    root_node = build_directory_tree(args.dir)
    visualize_tree(root_node)

if __name__ == "__main__":
    main()