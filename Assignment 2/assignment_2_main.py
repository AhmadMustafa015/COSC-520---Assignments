# ------------------------------------------------------
# Tree Data Structures Benchmark and Analysis
# ------------------------------------------------------
#
# INSTRUCTIONS FOR RUNNING THE CODE:
#
# 1. Environment Setup:
#    - Ensure Python 3.6+ is installed
#    - Install required packages: pip install matplotlib numpy bintrees
#
# 2. Running the Benchmark:
#    - Execute: python assignment_2_main.py
#    - This will run all benchmarks and generate comparison plots
#    - Results are saved as PNG files in the current directory
#
# 3. Running the Unit Tests:
#    - Execute: python assignment_2_main.py -t
#    - This runs tests for all tree implementations
#
# 4. Output Files:
#    - insert_benchmark.png: Performance comparison for insertion operations
#    - search_benchmark.png: Performance comparison for search operations
#    - delete_benchmark.png: Performance comparison for deletion operations
#    - benchmark_datasets.csv: The datasets used for benchmarking
#
# 5. Adjusting Parameters:
#    - Dataset sizes can be modified in run_experiment() function
#    - B-Tree order can be changed in the data_structures dictionary
#
# Note: The full benchmark may take several minutes to complete with
# larger dataset sizes.
#
# ------------------------------------------------------

import time
import random
import matplotlib.pyplot as plt
import unittest
import numpy as np
import os
import sys
from bintrees import AVLTree, RBTree  # Use bintrees library for AVLTree

# --- Binary Search Tree Implementation ---


class BSTNode:
    """
    Binary Search Tree Node class.

    Attributes:
        key: The value stored in the node
        left: Reference to left child node
        right: Reference to right child node
    """

    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None


class BST:
    """
    Binary Search Tree implementation.
    Provides methods for insert, search and delete operations.
    """

    def __init__(self):
        """Initialize an empty Binary Search Tree."""
        self.root = None

    def insert(self, key):
        """
        Insert a key into the BST.

        Args:
            key: The value to be inserted
        """
        self.root = self._insert_recursive(self.root, key)

    def _insert_recursive(self, node, key):
        """
        Helper method to recursively insert a key.

        Args:
            node: Current node being examined
            key: Value to be inserted

        Returns:
            Updated node after insertion
        """
        if node is None:
            return BSTNode(key)
        if key < node.key:
            node.left = self._insert_recursive(node.left, key)
        elif key > node.key:
            node.right = self._insert_recursive(node.right, key)
        return node

    def search(self, key):
        """
        Search for a key in the BST.

        Args:
            key: Value to search for

        Returns:
            Node containing the key if found, None otherwise
        """
        return self._search_recursive(self.root, key)

    def _search_recursive(self, node, key):
        """
        Helper method to recursively search for a key.

        Args:
            node: Current node being examined
            key: Value to search for

        Returns:
            Node containing the key if found, None otherwise
        """
        if node is None or node.key == key:
            return node
        if key < node.key:
            return self._search_recursive(node.left, key)
        return self._search_recursive(node.right, key)

    def delete(self, key):
        """
        Delete a key from the BST.

        Args:
            key: Value to be deleted
        """
        self.root = self._delete_recursive(self.root, key)

    def _delete_recursive(self, node, key):
        """
        Helper method to recursively delete a key.

        Args:
            node: Current node being examined
            key: Value to be deleted

        Returns:
            Updated node after deletion
        """
        if node is None:
            return node
        if key < node.key:
            node.left = self._delete_recursive(node.left, key)
        elif key > node.key:
            node.right = self._delete_recursive(node.right, key)
        else:
            if node.left is None:
                return node.right
            elif node.right is None:
                return node.left

            temp = self._min_value_node(node.right)
            node.key = temp.key
            node.right = self._delete_recursive(node.right, temp.key)
        return node

    def _min_value_node(self, node):
        """
        Find the node with minimum value in a subtree.

        Args:
            node: Root of the subtree

        Returns:
            Node with minimum value
        """
        current = node
        while current.left is not None:
            current = current.left
        return current


# --- B-Tree Implementation ---


class BTreeNode:
    """
    B-Tree Node class.

    Attributes:
        leaf: Boolean indicating if this is a leaf node
        keys: List of keys stored in the node
        children: List of child nodes
    """

    def __init__(self, leaf=True):
        self.leaf = leaf
        self.keys = []
        self.children = []


class BTree:
    """
    B-Tree implementation with a specific order.
    Provides methods for insert, search and delete operations.
    """

    def __init__(self, order):
        """
        Initialize an empty B-Tree with specified order.

        Args:
            order: The order of the B-Tree, determining the maximum number of children
        """
        self.root = BTreeNode(True)
        self.order = order  # Maximum number of children

    def insert(self, key):
        """
        Insert a key into the B-Tree.

        Args:
            key: Value to be inserted
        """
        root = self.root
        # Check if root is full
        if len(root.keys) == (2 * self.order) - 1:
            # Create new root
            temp = BTreeNode(False)  # Not a leaf
            self.root = temp
            # Add old root as child of new root
            temp.children.append(root)
            # Split old root
            self._split_child(temp, 0)
            # Insert key into new root
            self._insert_non_full(temp, key)
        else:
            # Root is not full, insert directly
            self._insert_non_full(root, key)

    def _insert_non_full(self, x, key):
        """
        Helper method to insert a key into a non-full node.

        Args:
            x: Node to insert into
            key: Value to be inserted
        """
        i = len(x.keys) - 1
        if x.leaf:
            # Insert key into leaf node
            # First add space for the new key
            x.keys.append(None)
            # Shift all greater keys to the right
            while i >= 0 and key < x.keys[i]:
                x.keys[i + 1] = x.keys[i]
                i -= 1
            # Insert the key
            x.keys[i + 1] = key
        else:
            # Find child where key should be inserted
            while i >= 0 and key < x.keys[i]:
                i -= 1
            i += 1
            # Check if child is full
            if len(x.children[i].keys) == (2 * self.order) - 1:
                # Split child if it's full
                self._split_child(x, i)
                # Determine which child to go to after split
                if key > x.keys[i]:
                    i += 1
            # Recursively insert into child
            self._insert_non_full(x.children[i], key)

    def _split_child(self, x, i):
        """
        Split a child node when it becomes too full.

        Args:
            x: Parent node
            i: Index of the child to split
        """
        t = self.order
        y = x.children[i]  # Child to split
        z = BTreeNode(y.leaf)  # New child node

        # Move keys from y to z
        mid_key = y.keys[t - 1]  # Middle key moves up to parent
        z.keys = y.keys[t:]  # Copy keys t to 2t-1 to z
        y.keys = y.keys[: t - 1]  # Keep keys 0 to t-2 in y

        # If not leaf, move children too
        if not y.leaf:
            z.children = y.children[t:]  # Move children t to 2t to z
            y.children = y.children[:t]  # Keep children 0 to t-1 in y

        # Insert z as child of x
        x.children.insert(i + 1, z)
        # Insert middle key into x
        x.keys.insert(i, mid_key)

    def search(self, key, x=None):
        """
        Search for a key in the B-Tree.

        Args:
            key: Value to search for
            x: Node to begin search from (default is root)

        Returns:
            True if key is found, False otherwise
        """
        if x is None:
            x = self.root

        # Find index where key should be
        i = 0
        while i < len(x.keys) and key > x.keys[i]:
            i += 1

        # Check if we found the key
        if i < len(x.keys) and key == x.keys[i]:
            return True

        # If this is a leaf and we didn't find the key, it's not in the tree
        if x.leaf:
            return False

        # Recursively search the appropriate child
        return self.search(key, x.children[i])

    def delete(self, key):
        """
        Delete a key from the B-Tree.

        Args:
            key: Value to be deleted
        """
        self._delete(self.root, key)

        # If root is empty and has children, make first child the new root
        if len(self.root.keys) == 0 and not self.root.leaf:
            self.root = self.root.children[0]

    def _delete(self, x, key):
        """
        Helper method to recursively delete a key.

        Args:
            x: Current node being examined
            key: Value to be deleted
        """
        t = self.order
        i = 0

        # Find position of key in node x
        while i < len(x.keys) and key > x.keys[i]:
            i += 1

        # If key is in this node
        if i < len(x.keys) and key == x.keys[i]:
            if x.leaf:
                # Case 1: Key is in leaf node - simply remove it
                x.keys.pop(i)
            else:
                # Case 2: Key is in internal node
                self._delete_internal_node(x, i)
        else:
            # Key is not in this node
            if x.leaf:
                # Key not in tree
                return

            # Check if child needs filling
            if len(x.children[i].keys) < t:
                self._fill_child(x, i)

            # If last child was merged, go to the previous child
            if i >= len(x.children):
                i = len(x.children) - 1

            # Recursively delete from child
            self._delete(x.children[i], key)

    def _delete_internal_node(self, x, i):
        """
        Delete a key from an internal node.

        Args:
            x: Current node
            i: Index of key to delete
        """
        t = self.order
        key = x.keys[i]

        # Case 2a: Predecessor can be used
        if len(x.children[i].keys) >= t:
            # Find predecessor
            pred = self._get_predecessor(x.children[i])
            x.keys[i] = pred
            # Recursively delete predecessor from child
            self._delete(x.children[i], pred)

        # Case 2b: Successor can be used
        elif len(x.children[i + 1].keys) >= t:
            # Find successor
            succ = self._get_successor(x.children[i + 1])
            x.keys[i] = succ
            # Recursively delete successor from child
            self._delete(x.children[i + 1], succ)

        # Case 2c: Both children have t-1 keys, merge them
        else:
            # Merge children i and i+1 with key i between them
            self._merge_children(x, i)
            # Delete key from the merged child
            self._delete(x.children[i], key)

    def _get_predecessor(self, x):
        """
        Get the predecessor key (rightmost key in left subtree).

        Args:
            x: Current node

        Returns:
            Predecessor key
        """
        # Go to rightmost leaf node
        while not x.leaf:
            x = x.children[-1]
        # Return rightmost key
        return x.keys[-1]

    def _get_successor(self, x):
        """
        Get the successor key (leftmost key in right subtree).

        Args:
            x: Current node

        Returns:
            Successor key
        """
        # Go to leftmost leaf node
        while not x.leaf:
            x = x.children[0]
        # Return leftmost key
        return x.keys[0]

    def _fill_child(self, x, i):
        """
        Fill a child node that has fewer than t-1 keys.

        Args:
            x: Parent node
            i: Index of the child to fill
        """
        t = self.order

        # Try to borrow from left sibling
        if i > 0 and len(x.children[i - 1].keys) >= t:
            self._borrow_from_prev(x, i)

        # Try to borrow from right sibling
        elif i < len(x.children) - 1 and len(x.children[i + 1].keys) >= t:
            self._borrow_from_next(x, i)

        # Merge with sibling
        else:
            if i < len(x.children) - 1:
                # Merge with right sibling
                self._merge_children(x, i)
            else:
                # Merge with left sibling
                self._merge_children(x, i - 1)

    def _borrow_from_prev(self, x, i):
        """
        Borrow a key from the previous child.

        Args:
            x: Parent node
            i: Index of the current child
        """
        child = x.children[i]
        sibling = x.children[i - 1]

        # Shift all keys in child to make room
        child.keys.insert(0, x.keys[i - 1])

        # If not leaf, move child pointer too
        if not child.leaf:
            child.children.insert(0, sibling.children.pop())

        # Move key from sibling to parent
        x.keys[i - 1] = sibling.keys.pop()

    def _borrow_from_next(self, x, i):
        """
        Borrow a key from the next child.

        Args:
            x: Parent node
            i: Index of the current child
        """
        child = x.children[i]
        sibling = x.children[i + 1]

        # Move key from parent to child
        child.keys.append(x.keys[i])

        # Move key from sibling to parent
        x.keys[i] = sibling.keys.pop(0)

        # If not leaf, move child pointer too
        if not sibling.leaf:
            child.children.append(sibling.children.pop(0))

    def _merge_children(self, x, i):
        """
        Merge two child nodes.

        Args:
            x: Parent node
            i: Index of the first child to merge
        """
        child = x.children[i]
        sibling = x.children[i + 1]

        # Add key from parent to child
        child.keys.append(x.keys.pop(i))

        # Move all keys from sibling to child
        child.keys.extend(sibling.keys)

        # If not leaf, move all child pointers too
        if not child.leaf:
            child.children.extend(sibling.children)

        # Remove sibling from parent's children
        x.children.pop(i + 1)


# --- Wrapper class for AVLTree from bintrees ---


class AVLTreeWrapper:
    """
    Wrapper for AVLTree from bintrees library to provide a consistent API.

    This wrapper adapts the bintrees.AVLTree class to match our BST interface.
    """

    def __init__(self):
        """Initialize an empty AVL Tree."""
        self.tree = AVLTree()

    def insert(self, key):
        """
        Insert a key into the AVL tree.

        Args:
            key: Value to be inserted
        """
        # For bintrees.AVLTree, we need both key and value
        # We'll use the key as the value as well
        self.tree.insert(key, key)

    def search(self, key):
        """
        Search for a key in the AVL tree.

        Args:
            key: Value to search for

        Returns:
            True if key is found, None otherwise
        """
        try:
            self.tree.get(key)
            return True
        except KeyError:
            return None

    def delete(self, key):
        """
        Delete a key from the AVL tree.

        Args:
            key: Value to be deleted
        """
        try:
            self.tree.remove(key)
        except KeyError:
            pass  # Key not found, do nothing


# --- Data Generation and Benchmarking ---


def generate_datasets(sizes, seed=42):
    """
    Generates multiple datasets of different sizes for benchmarking.

    Args:
        sizes: List of dataset sizes to generate
        seed: Random seed for reproducibility

    Returns:
        Dictionary mapping sizes to datasets
    """
    datasets = {}
    random.seed(seed)

    print("Generating datasets...")
    for size in sizes:
        print(f"  Generating dataset with {size} elements...")
        datasets[size] = random.sample(range(size * 3), size)

    return datasets


def save_datasets_csv(datasets, filename="benchmark_datasets.csv"):
    """
    Save all datasets to a CSV file with each column representing a dataset of a specific size.

    Args:
        datasets: Dictionary mapping sizes to datasets
        filename: Name of the CSV file to save to
    """
    import csv

    # Find the max length across all datasets
    max_length = max(len(dataset) for dataset in datasets.values())

    # Prepare headers
    headers = [f"size_{size}" for size in sorted(datasets.keys())]

    print(f"Saving datasets to {filename}...")
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)

        # Write data rows, padding shorter datasets with empty cells
        for i in range(max_length):
            row = []
            for size in sorted(datasets.keys()):
                dataset = datasets[size]
                row.append(dataset[i] if i < len(dataset) else "")
            writer.writerow(row)

    print(f"Datasets saved to {filename}")


def benchmark(data_structures, datasets, operations):
    """
    Benchmarks the given data structures with specified operations and datasets.

    Args:
        data_structures: Dictionary of data structures to benchmark
        datasets: Dictionary mapping sizes to datasets
        operations: Dictionary of operations to benchmark

    Returns:
        Dictionary with benchmarking results for each data structure and operation
    """
    results = {}
    dataset_sizes = sorted(datasets.keys())

    # Initialize results structure
    for ds_name in data_structures:
        results[ds_name] = {op_name: [] for op_name in operations}

    # First loop over dataset sizes for fair comparison
    for size in dataset_sizes:
        dataset = datasets[size]
        print(f"\nBenchmarking with dataset size {size}...")

        # For each dataset size, benchmark all data structures
        for ds_name, ds_class in data_structures.items():
            print(f"  Testing {ds_name}...")

            # For each operation type
            for op_name, op_func in operations.items():
                print(f"    Operation: {op_name}")

                # Create a new instance for this test
                if ds_name == "BTree (Order=5)":
                    ds = BTree(5)
                else:
                    ds = ds_class()

                # For search and delete, we need to insert the elements first
                if op_name in ["search", "delete"]:
                    for key in dataset:
                        ds.insert(key)

                # Benchmark the operation
                start_time = time.time()

                if op_name == "insert":
                    for key in dataset:
                        op_func(ds, key)
                elif op_name == "search":
                    for key in dataset:
                        op_func(ds, key)
                elif op_name == "delete":
                    for key in dataset:
                        op_func(ds, key)

                end_time = time.time()
                elapsed = end_time - start_time
                results[ds_name][op_name].append(elapsed)
                print(f"      Completed in {elapsed:.2f} seconds")

    return results, dataset_sizes



def plot_results(results, dataset_sizes, operation):
    """
    Plots the benchmarking results with improved visualization.

    Args:
        results: Dictionary with benchmarking results
        dataset_sizes: List of dataset sizes tested
        operation: Operation to plot (insert, search, or delete)
    """
    plt.figure(figsize=(12, 8))

    markers = ["o", "s", "^", "D", "x"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for i, (ds_name, op_results) in enumerate(results.items()):
        plt.plot(
            dataset_sizes,
            op_results[operation],
            label=ds_name,
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
            linewidth=2,
            markersize=8,
        )

    plt.xlabel("Dataset Size (n)", fontsize=14)
    plt.ylabel("Time (seconds)", fontsize=14)
    plt.title(f"Runtime Complexity: {operation.capitalize()} Operation", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Use log scale for x-axis to better visualize growth patterns
    plt.xscale("log")

    plt.savefig(f"{operation}_benchmark.png", dpi=300)
    print(f"Plot saved as {operation}_benchmark.png")
    plt.close()


# --- Unit Tests ---


class TestDataStructures(unittest.TestCase):
    """Unit tests for tree data structures."""

    def test_bst_insert_search(self):
        """Test BST insertion and search functionality."""
        bst = BST()
        test_values = [50, 30, 70, 20, 40, 60, 80]

        # Insert values
        for val in test_values:
            bst.insert(val)

        # Test search for existing values
        for val in test_values:
            self.assertIsNotNone(bst.search(val), f"BST search failed for {val}")

        # Test search for non-existing values
        for val in [10, 35, 55, 90]:
            self.assertIsNone(bst.search(val), f"BST incorrectly found {val}")

    def test_bst_delete(self):
        """Test BST deletion functionality."""
        bst = BST()
        test_values = [50, 30, 70, 20, 40, 60, 80]

        # Insert values
        for val in test_values:
            bst.insert(val)

        # Delete leaf node
        bst.delete(20)
        self.assertIsNone(bst.search(20))

        # Delete node with one child
        bst.delete(30)
        self.assertIsNone(bst.search(30))
        self.assertIsNotNone(bst.search(40))

        # Delete node with two children
        bst.delete(70)
        self.assertIsNone(bst.search(70))
        self.assertIsNotNone(bst.search(60))
        self.assertIsNotNone(bst.search(80))

    def test_btree(self):
        """Test B-Tree operations."""
        btree = BTree(3)  # Order 3
        test_values = [10, 20, 5, 15, 30, 25, 35]

        # Insert values
        for val in test_values:
            btree.insert(val)

        # Test search
        for val in test_values:
            self.assertTrue(btree.search(val), f"B-Tree search failed for {val}")

        # Test search for non-existing values
        for val in [1, 12, 22, 40]:
            self.assertFalse(btree.search(val), f"B-Tree incorrectly found {val}")

        # Test delete
        btree.delete(20)
        self.assertFalse(btree.search(20))
        self.assertTrue(btree.search(10))
        self.assertTrue(btree.search(30))


# --- Run Experiment Function ---


def run_experiment():
    """
    Run the tree data structures benchmarking experiment.

    Generates random datasets, benchmarks different operations across tree data structures,
    and produces visualization plots of the results.
    """
    # Define dataset sizes for benchmarking
    dataset_sizes = [100000, 1000000, 5000000, 10000000]  # Adjusted for reasonable run time

    # Generate all datasets upfront
    datasets = generate_datasets(dataset_sizes)

    # Save datasets to CSV for reference and reproducibility
    save_datasets_csv(datasets)

    # Define data structures to benchmark
    data_structures = {
        "BST": BST,
        "AVLTree": AVLTreeWrapper,
        "BTree (Order=5)": BTree,
    }

    # Define operations to benchmark
    operations = {
        "insert": lambda ds, key: ds.insert(key),
        "search": lambda ds, key: ds.search(key),
        "delete": lambda ds, key: ds.delete(key),
    }

    print("\nStarting benchmarking...")
    results, used_sizes = benchmark(data_structures, datasets, operations)

    print("\nGenerating plots...")
    for operation in ["insert", "search", "delete"]:
        plot_results(results, used_sizes, operation)

    print("\nBenchmark complete. Plots saved as .png files.")
    print("\nNOTE: All datasets have been saved in 'benchmark_datasets.csv'.")

# --- Main Execution ---

if __name__ == "__main__":
    # Check if -t flag is provided to run unit tests
    if len(sys.argv) > 1 and sys.argv[1] == "-t":
        # Remove the "-t" argument so that unittest does not get confused
        sys.argv = [sys.argv[0]]
        unittest.main()
    else:
        # Run the experiment
        run_experiment()
