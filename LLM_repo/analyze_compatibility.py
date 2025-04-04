import json
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import seaborn as sns

def load_all_combinations(iteration_files):
    """Load and combine all iteration files"""
    all_combinations = {}
    for file in iteration_files:
        with open(file, 'r') as f:
            data = json.load(f)
            for model_id, model_data in data.items():
                if model_data:
                    if model_id not in all_combinations:
                        all_combinations[model_id] = []
                    all_combinations[model_id].extend(model_data)
    return all_combinations

def create_compatibility_graph(combinations):
    """Create a graph showing package version compatibility"""
    G = nx.Graph()
    
    # Add nodes and edges based on successful combinations
    for model_data in combinations.values():
        for combo in model_data:
            # Add all package versions as nodes
            for pkg1, ver1 in combo.items():
                node1 = f"{pkg1}=={ver1}"
                G.add_node(node1, package=pkg1, version=ver1)
                
                # Add edges between all pairs of packages in this combination
                for pkg2, ver2 in combo.items():
                    if pkg1 < pkg2:  # Avoid duplicate edges
                        node2 = f"{pkg2}=={ver2}"
                        G.add_edge(node1, node2, weight=1)
    
    return G

def plot_compatibility_network(G, output_file):
    """Plot the compatibility network"""
    plt.figure(figsize=(20, 20))
    
    # Use spring layout for better visualization
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, 
                          node_color='lightblue',
                          node_size=2000,
                          alpha=0.7)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos,
                          edge_color='gray',
                          width=1,
                          alpha=0.5)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos,
                           font_size=8,
                           font_weight='bold')
    
    plt.title("Package Version Compatibility Network", pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def create_compatibility_matrix(G, output_file):
    """Create a heatmap of package version compatibility"""
    # Get all nodes
    nodes = list(G.nodes())
    
    # Create compatibility matrix
    n = len(nodes)
    matrix = np.zeros((n, n))
    
    # Fill matrix with edge weights
    for i, node1 in enumerate(nodes):
        for j, node2 in enumerate(nodes):
            if G.has_edge(node1, node2):
                matrix[i, j] = G[node1][node2]['weight']
    
    # Create heatmap
    plt.figure(figsize=(15, 15))
    sns.heatmap(matrix,
                xticklabels=nodes,
                yticklabels=nodes,
                cmap='YlOrRd',
                annot=True,
                fmt='g')
    plt.title("Package Version Compatibility Matrix")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def analyze_compatibility_stats(G):
    """Analyze and print compatibility statistics"""
    # Group nodes by package
    package_groups = defaultdict(list)
    for node in G.nodes():
        package = G.nodes[node]['package']
        package_groups[package].append(node)
    
    # Print statistics
    print("\nCompatibility Analysis:")
    print("=====================")
    for package, versions in package_groups.items():
        print(f"\n{package}:")
        print(f"  Number of versions: {len(versions)}")
        print("  Versions:")
        for version in sorted(versions):
            print(f"    - {version}")
            # Print compatible packages
            compatible = [n for n in G.neighbors(version)]
            print(f"      Compatible with: {', '.join(compatible)}")

def main():
    # List of iteration files
    iteration_files = [
        f'successful_combinations {i}.json'
        for i in range(1, 8)
    ]
    
    # Load and combine all combinations
    combinations = load_all_combinations(iteration_files)
    
    # Create compatibility graph
    G = create_compatibility_graph(combinations)
    
    # Create visualizations
    plot_compatibility_network(G, 'plot/compatibility_network.png')
    create_compatibility_matrix(G, 'plot/compatibility_matrix.png')
    
    # Print statistics
    analyze_compatibility_stats(G)
    
    print("\nVisualization complete. Generated files:")
    print("1. plot/compatibility_network.png - Network visualization of package compatibility")
    print("2. plot/compatibility_matrix.png - Heatmap of package compatibility")

if __name__ == "__main__":
    main() 