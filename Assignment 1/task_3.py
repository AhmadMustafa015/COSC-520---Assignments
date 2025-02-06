#!/usr/bin/env python3
"""
GitHub Repository: https://github.com/AhmadMustafa015/COSC-520---Assignments.git

This script compares three membership testing methods for a login checker:
    1. Hashing (using Python's set)
    2. Bloom Filter
    3. Cuckoo Filter

It generates a synthetic dataset of login strings, inserts them into each data structure,
and measures the average lookup time for randomly chosen queries.

Setup and Running Instructions:
    - Ensure Python 3.x is installed.
    - Install required packages:
          pip install matplotlib
          pip install pybloom-live
          pip install cuckoofilter
    - To run the experiment:
          python task_3.py
    - To run unit tests:
          python task_3.py -t
       (Alternatively: python -m unittest task_3.py)

Author: Ahmad Abdel-Qader
Date: 2025-02-02
"""

import random
import string
import time
import math
import csv
import matplotlib.pyplot as plt
import unittest
import sys
from pybloom_live import BloomFilter
from cuckoofilter import CuckooFilter

# ---------------------------
# Utility Functions
# ---------------------------
def generate_dataset(n, string_length=8):
    """
    Generate a list of n random login strings.

    Input:
        n (int): Number of login strings to generate.
        string_length (int): Length of each login string.

    Output:
        list: A list containing n randomly generated login strings.
    """
    dataset = []
    for _ in range(n):
        # Create a random string from lowercase letters and digits.
        s = ''.join(random.choices(string.ascii_lowercase + string.digits, k=string_length))
        dataset.append(s)
    return dataset

def save_dataset_csv(dataset, filename="dataset.csv"):
    """
    Save the dataset to a CSV file with a header 'login'.

    Input:
        dataset (list): List of login strings.
        filename (str): Name of the CSV file to be saved.

    Output:
        None (writes the CSV file to disk)
    """
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["login"])
        for login in dataset:
            writer.writerow([login])
    print(f"Dataset saved to {filename}")

def measure_lookup_time(func, queries, repeat=1):
    """
    Measure the average lookup time per query for a membership test function.

    Input:
        func (callable): A function accepting a single query and returning a boolean.
        queries (list): A list of query strings.
        repeat (int): Number of times to repeat the full set for averaging.

    Output:
        float: Average lookup time per query in seconds.
    """
    total_time = 0.0
    for _ in range(repeat):
        start = time.perf_counter()
        for q in queries:
            func(q)
        end = time.perf_counter()
        total_time += (end - start)
    return total_time / (len(queries) * repeat)

# ---------------------------
# Main Experiment Function
# ---------------------------
def run_experiment():
    """
    Run the experiment comparing hashing, Bloom filter, and Cuckoo filter.

    Input:
        None

    Output:
        None (prints timing results and saves a lookup comparison plot)
    """
    # Experiment parameters
    dataset_size = 10000000      # 10 million login strings
    query_count = 1000          # Number of random lookup queries
    false_positive_prob = 0.01  # Desired false positive rate for Bloom filter

    # Generate synthetic dataset and save to CSV.
    dataset = generate_dataset(dataset_size)
    save_dataset_csv(dataset, "dataset.csv")

    # Build the three data structures.
    # 1. Hashing: Use Python's set.
    hash_set = set(dataset)

    # 2. Bloom Filter.
    bloom = BloomFilter(dataset_size, false_positive_prob)
    for item in dataset:
        bloom.add(item)

    # 3. Cuckoo Filter.
    cuckoo = CuckooFilter(dataset_size, fingerprint_size=8)
    for item in dataset:
        cuckoo.insert(item)

    # Select a set of random queries from the dataset and generate new queries.
    queries = random.sample(dataset, query_count//2)
    queries.extend(generate_dataset(query_count//2))

    # Measure average lookup time (in seconds) for each structure.
    hash_time   = measure_lookup_time(lambda q: (q in hash_set), queries, repeat=3)
    bloom_time  = measure_lookup_time(lambda q: bloom.__contains__(q), queries, repeat=3)
    cuckoo_time = measure_lookup_time(lambda q: cuckoo.contains(q), queries, repeat=3)

    # Print timing results.
    print(f"Average lookup time for Hashing (set): {hash_time:.6e} seconds")
    print(f"Average lookup time for Bloom Filter: {bloom_time:.6e} seconds")
    print(f"Average lookup time for Cuckoo Filter: {cuckoo_time:.6e} seconds")

    # Plot the results.
    labels = ['Hashing', 'Bloom Filter', 'Cuckoo Filter']
    times = [hash_time, bloom_time, cuckoo_time]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, times, color=['blue', 'green', 'red'])
    plt.ylabel("Average Lookup Time (seconds)")
    plt.title("Lookup Time Comparison")
    plt.savefig("lookup_comparison.png")
    plt.show()
    print("Lookup time comparison plot saved as lookup_comparison.png")

# ---------------------------
# Unit Tests
# ---------------------------
class TestFilters(unittest.TestCase):
    """
    Unit tests for BloomFilter and CuckooFilter implementations.
    """
    def setUp(self):
        """
        Setup a small dataset and corresponding filters for testing.
        """
        self.dataset = ["user1", "user2", "user3", "user4", "user5"]
        self.bloom = BloomFilter(capacity=10, error_rate=0.01)
        for item in self.dataset:
            self.bloom.add(item)
        self.cuckoo = CuckooFilter(capacity=10, fingerprint_size=8)
        for item in self.dataset:
            self.cuckoo.insert(item)

    def test_bloom_filter_positive(self):
        """
        Test that all inserted items are reported as present by BloomFilter.
        """
        for item in self.dataset:
            self.assertTrue(self.bloom.__contains__(item), f"Bloom filter failed to find {item}")

    def test_bloom_filter_negative(self):
        """
        Test that a non-inserted item is reported as not present by BloomFilter.
        Note: Bloom filters may have false positives, but the chance is low given p=0.01.
        """
        self.assertFalse(self.bloom.__contains__("nonexistent_user"),
                         "Bloom filter falsely reported a nonexistent item")

    def test_cuckoo_filter_positive(self):
        """
        Test that all inserted items are reported as present by CuckooFilter.
        """
        for item in self.dataset:
            self.assertTrue(self.cuckoo.contains(item), f"Cuckoo filter failed to find {item}")

    def test_cuckoo_filter_negative(self):
        """
        Test that a non-inserted item is reported as not present by CuckooFilter.
        """
        self.assertFalse(self.cuckoo.contains("nonexistent_user"),
                         "Cuckoo filter falsely reported a nonexistent item")

# ---------------------------
# Main Execution
# ---------------------------
if __name__ == "__main__":
    # If the script is executed with "-t" as a command-line argument, run unit tests.
    if len(sys.argv) > 1 and sys.argv[1] == '-t':
        # Remove the "-t" argument so that unittest does not get confused.
        sys.argv = [sys.argv[0]]
        unittest.main()
    else:
        run_experiment()