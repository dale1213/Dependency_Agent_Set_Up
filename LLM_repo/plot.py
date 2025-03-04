import json
import matplotlib.pyplot as plt

with open("LLM_repo/results_larger_set_statistics.json", "r") as f:
    stats = json.load(f)

summary = stats["summary"]
labels = ['Success', 'Failure']
sizes = [summary["success_count"], summary["failure_count"]]
colors = ['green', 'red']

plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.title('Repository Success vs Failure')
plt.axis('equal')
plt.savefig('LLM_repo/plot/success_failure_pie.png')


cluster_dist = stats["cluster_distribution"]
clusters = list(cluster_dist.keys())
counts = list(cluster_dist.values())

plt.figure(figsize=(10, 6))
plt.barh(clusters, counts, color='skyblue')
plt.xlabel('Issue Count')
plt.title('Issue Cluster Distribution')
plt.tight_layout()
plt.savefig('LLM_repo/plot/issue_cluster_distribution.png')

pattern_counts = {}
for pattern, _ in stats["identified_patterns"]:
    pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

patterns = list(pattern_counts.keys())
pattern_vals = list(pattern_counts.values())

plt.figure(figsize=(10, 6))
plt.barh(patterns, pattern_vals, color='lightgreen')
plt.xlabel('Occurrences')
plt.title('Identified Patterns Occurrences')
plt.tight_layout()
plt.savefig('LLM_repo/plot/identified_patterns_occurrences.png')
