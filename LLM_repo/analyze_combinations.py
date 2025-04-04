import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import numpy as np
import os

def ensure_plot_dir():
    """Ensure the plot directory exists"""
    if not os.path.exists('plot'):
        os.makedirs('plot')

def load_combinations(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def get_successful_models_count(combinations):
    return sum(1 for model_data in combinations.values() if model_data)

def get_package_frequency(combinations):
    package_freq = defaultdict(int)
    for model_data in combinations.values():
        if model_data:
            for combo in model_data:
                for package in combo.keys():
                    package_freq[package] += 1
    return package_freq

def get_version_distribution(combinations):
    version_dist = defaultdict(lambda: defaultdict(int))
    for model_data in combinations.values():
        if model_data:
            for combo in model_data:
                for package, version in combo.items():
                    version_dist[package][version] += 1
    return version_dist

def create_iteration_progress_plot(iteration_files):
    """Plot the number of successful models across iterations"""
    counts = []
    for i, file in enumerate(iteration_files, 1):
        combinations = load_combinations(file)
        counts.append(get_successful_models_count(combinations))
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(counts) + 1), counts, marker='o')
    plt.title('Number of Successful Models Across Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Number of Successful Models')
    plt.grid(True)
    plt.savefig('plot/iteration_progress.png')
    plt.close()

def create_package_heatmap(iteration_files):
    """Create a heatmap of package co-occurrence"""
    # Create a set of all packages
    all_packages = set()
    for file in iteration_files:
        combinations = load_combinations(file)
        for model_data in combinations.values():
            if model_data:
                for combo in model_data:
                    all_packages.update(combo.keys())
    
    # Create co-occurrence matrix
    n_packages = len(all_packages)
    cooccurrence = np.zeros((n_packages, n_packages))
    package_list = list(all_packages)
    
    for file in iteration_files:
        combinations = load_combinations(file)
        for model_data in combinations.values():
            if model_data:
                for combo in model_data:
                    for i, pkg1 in enumerate(package_list):
                        for j, pkg2 in enumerate(package_list):
                            if pkg1 in combo and pkg2 in combo:
                                cooccurrence[i, j] += 1
    
    # Normalize the co-occurrence matrix by dividing each row by its diagonal value
    normalized_cooccurrence = np.zeros_like(cooccurrence)
    for i in range(n_packages):
        if cooccurrence[i, i] > 0:  # Avoid division by zero
            normalized_cooccurrence[i, :] = cooccurrence[i, :] / cooccurrence[i, i]
    
    # Create heatmap
    plt.figure(figsize=(15, 12))
    sns.heatmap(normalized_cooccurrence, 
                xticklabels=package_list,
                yticklabels=package_list,
                cmap='YlOrRd',
                annot=True,
                fmt='.2f',  # Show 2 decimal places
                vmin=0,
                vmax=1)  # Set the range from 0 to 1
    plt.title('Normalized Package Co-occurrence Heatmap')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('plot/package_heatmap.png')
    plt.close()

def create_version_distribution_plot(iteration_files):
    """Create a plot showing version distribution for major packages"""
    # Combine all iterations
    all_combinations = {}
    for file in iteration_files:
        combinations = load_combinations(file)
        for model_id, model_data in combinations.items():
            if model_data:
                if model_id not in all_combinations:
                    all_combinations[model_id] = []
                all_combinations[model_id].extend(model_data)
    
    version_dist = get_version_distribution(all_combinations)
    
    # Focus on major packages
    major_packages = ['transformers', 'torch', 'numpy', 'datasets', 'tokenizers']
    
    plt.figure(figsize=(15, 8))
    for i, package in enumerate(major_packages, 1):
        if package in version_dist:
            versions = list(version_dist[package].keys())
            counts = list(version_dist[package].values())
            plt.subplot(2, 3, i)
            plt.bar(versions, counts)
            plt.title(f'{package} Version Distribution')
            plt.xticks(rotation=45, ha='right')
            plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('plot/version_distribution.png')
    plt.close()

def analyze_version_conflicts(iteration_files):
    """Analyze and report version conflicts"""
    all_combinations = {}
    for file in iteration_files:
        combinations = load_combinations(file)
        for model_id, model_data in combinations.items():
            if model_data:
                if model_id not in all_combinations:
                    all_combinations[model_id] = []
                all_combinations[model_id].extend(model_data)
    
    version_dist = get_version_distribution(all_combinations)
    
    # Find packages with multiple versions
    conflicts = {}
    for package, versions in version_dist.items():
        if len(versions) > 1:
            conflicts[package] = versions
    
    # Write conflicts to a file
    with open('plot/version_conflicts.txt', 'w') as f:
        f.write("Version Conflicts Analysis\n")
        f.write("=======================\n\n")
        for package, versions in conflicts.items():
            f.write(f"\n{package}:\n")
            for version, count in versions.items():
                f.write(f"  {version}: {count} occurrences\n")

def main():
    # Ensure plot directory exists
    ensure_plot_dir()
    
    # List of iteration files
    iteration_files = [
        'successful_combinations 1.json',
        'successful_combinations 2.json',
        'successful_combinations 3.json',
        'successful_combinations 4.json',
        'successful_combinations 5.json',
        'successful_combinations 6.json',
        'successful_combinations 7.json'
    ]
    
    # Create all visualizations
    create_iteration_progress_plot(iteration_files)
    create_package_heatmap(iteration_files)
    create_version_distribution_plot(iteration_files)
    analyze_version_conflicts(iteration_files)
    
    print("Analysis complete. Generated files in plot/ directory:")
    print("1. iteration_progress.png - Shows progress across iterations")
    print("2. package_heatmap.png - Shows package co-occurrence")
    print("3. version_distribution.png - Shows version distribution for major packages")
    print("4. version_conflicts.txt - Detailed analysis of version conflicts")

if __name__ == "__main__":
    main() 