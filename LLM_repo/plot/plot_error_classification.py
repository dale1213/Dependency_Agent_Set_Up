import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

def load_error_classifications(file_path):
    """Load error classification data from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def aggregate_error_statistics(data):
    """Aggregate error statistics across all models."""
    total_errors = 0
    category_counts = defaultdict(int)
    dependency_counts = defaultdict(int)
    
    for model_data in data.values():
        stats = model_data['error_statistics']
        total_errors += stats['total_errors']
        
        # Aggregate category counts
        for category, count in stats['categories'].items():
            category_counts[category] += count
            
        # Aggregate dependency counts
        for dep, count in stats['missing_dependencies'].items():
            dependency_counts[dep] += count
    
    return total_errors, category_counts, dependency_counts

def plot_error_categories(category_counts, output_dir):
    """Create a bar plot of error categories."""
    plt.figure(figsize=(12, 6))
    
    # Sort categories by count
    categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
    categories_names = [cat[0] for cat in categories]
    categories_counts = [cat[1] for cat in categories]
    
    # Create bar plot
    bars = plt.bar(categories_names, categories_counts)
    
    # Customize plot
    plt.title('Distribution of Error Categories', pad=20)
    plt.xlabel('Error Category')
    plt.ylabel('Number of Occurrences')
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(output_dir, 'error_categories.png'))
    plt.close()

def plot_top_dependencies(dependency_counts, output_dir, top_n=10):
    """Create a bar plot of top N missing dependencies."""
    plt.figure(figsize=(12, 6))
    
    # Sort dependencies by count and get top N
    dependencies = sorted(dependency_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    dep_names = [dep[0] for dep in dependencies]
    dep_counts = [dep[1] for dep in dependencies]
    
    # Create bar plot
    bars = plt.bar(dep_names, dep_counts)
    
    # Customize plot
    plt.title(f'Top {top_n} Missing Dependencies', pad=20)
    plt.xlabel('Dependency')
    plt.ylabel('Number of Occurrences')
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(output_dir, 'top_dependencies.png'))
    plt.close()

def plot_error_timeline(data, output_dir):
    """Create a timeline plot of errors."""
    plt.figure(figsize=(15, 6))
    
    # Collect all errors with timestamps
    all_errors = []
    for model_id, model_data in data.items():
        for error in model_data['errors']:
            all_errors.append({
                'timestamp': error['timestamp'],
                'category': error['category'],
                'model_id': model_id
            })
    
    # Sort errors by timestamp
    all_errors.sort(key=lambda x: x['timestamp'])
    
    # Create timeline plot
    categories = list(set(error['category'] for error in all_errors))
    timeline_data = defaultdict(list)
    
    for error in all_errors:
        timeline_data[error['category']].append(error['timestamp'])
    
    # Plot each category
    for category in categories:
        if timeline_data[category]:
            plt.scatter(timeline_data[category], [category] * len(timeline_data[category]), 
                       label=category, alpha=0.6)
    
    # Customize plot
    plt.title('Error Timeline by Category', pad=20)
    plt.xlabel('Timestamp')
    plt.ylabel('Error Category')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(output_dir, 'error_timeline.png'))
    plt.close()

def main():
    # Create plot directory if it doesn't exist
    plot_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(plot_dir, exist_ok=True)
    
    # Load error classification data
    input_file = os.path.join(os.path.dirname(plot_dir), 'model_cards_text-classificatio_error_classification.json')
    data = load_error_classifications(input_file)
    
    # Aggregate statistics
    total_errors, category_counts, dependency_counts = aggregate_error_statistics(data)
    
    # Print summary statistics
    print(f"\nTotal number of errors: {total_errors}")
    print("\nError categories distribution:")
    for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{category}: {count}")
    
    print("\nTop missing dependencies:")
    for dep, count in sorted(dependency_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{dep}: {count}")
    
    # Create visualizations
    plot_error_categories(category_counts, plot_dir)
    plot_top_dependencies(dependency_counts, plot_dir)
    plot_error_timeline(data, plot_dir)
    
    print(f"\nPlots have been saved to: {plot_dir}")

if __name__ == "__main__":
    main() 