import json
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# 设置美观的样式
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("pastel")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# 确保输出目录存在
os.makedirs('LLM_repo/plot', exist_ok=True)

# 从JSON文件加载统计数据
with open("LLM_repo/results_larger_set_statistics.json", "r") as f:
    stats = json.load(f)

def generate_plots():
    """生成所有可视化图表"""
    generate_success_failure_pie()
    generate_cluster_distribution()
    generate_identified_patterns()

def generate_success_failure_pie():
    """生成成功/失败比例饼图"""
    summary = stats["summary"]
    labels = ['成功', '失败']
    sizes = [summary["success_count"], summary["failure_count"]]
    explode = (0.1, 0)  # 突出成功部分
    
    # 自定义颜色
    colors = ['#4CAF50', '#F44336']
    
    fig, ax = plt.figure(figsize=(10, 8)), plt.subplot(111)
    wedges, texts, autotexts = ax.pie(
        sizes, 
        explode=explode, 
        labels=labels, 
        colors=colors, 
        autopct='%1.1f%%', 
        startangle=90,
        shadow=True,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )
    
    # 设置标签和自动文本的样式
    plt.setp(autotexts, size=14, weight='bold')
    plt.setp(texts, size=14, weight='bold')
    
    # 添加标题和注释
    ax.set_title('仓库处理成功率分析', fontweight='bold', pad=20)
    plt.annotate(
        f'总仓库数: {summary["total_repositories"]}', 
        xy=(0, -0.1), 
        xycoords='axes fraction',
        fontsize=12,
        ha='center'
    )
    
    plt.axis('equal')  # 确保饼图是圆的
    plt.tight_layout()
    plt.savefig('LLM_repo/plot/success_failure_pie.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_cluster_distribution():
    """生成问题集群分布图"""
    cluster_dist = stats["cluster_distribution"]
    clusters = list(cluster_dist.keys())
    counts = list(cluster_dist.values())
    
    # 排序以便更好地展示
    sorted_indices = np.argsort(counts)
    clusters = [clusters[i] for i in sorted_indices]
    counts = [counts[i] for i in sorted_indices]
    
    # 创建自定义颜色映射
    colors = sns.color_palette("viridis", len(clusters))
    
    fig, ax = plt.subplots(figsize=(12, 10))
    bars = ax.barh(clusters, counts, color=colors, alpha=0.8, height=0.6)
    
    # 添加数值标签
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.3, 
            bar.get_y() + bar.get_height()/2, 
            f'{int(width)}', 
            ha='left', 
            va='center', 
            fontweight='bold'
        )
    
    # 设置标签和标题
    ax.set_xlabel('问题数量', fontweight='bold')
    ax.set_title('问题类型分布', fontweight='bold', pad=20)
    
    # 设置网格线只在x轴显示
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, linestyle='--', alpha=0.6)
    
    # 添加背景颜色
    ax.set_facecolor('#f8f9fa')
    
    # 移除框架
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    plt.tight_layout()
    plt.savefig('LLM_repo/plot/issue_cluster_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_identified_patterns():
    """生成已识别模式图"""
    pattern_counts = {}
    for pattern, _ in stats["identified_patterns"]:
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
    
    # 筛选出出现频率较高的模式（如果太多）
    if len(pattern_counts) > 12:
        sorted_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)
        patterns = [p[0] for p in sorted_patterns[:12]]
        pattern_vals = [p[1] for p in sorted_patterns[:12]]
    else:
        patterns = list(pattern_counts.keys())
        pattern_vals = list(pattern_counts.values())
    
    # 排序以便更好地展示
    sorted_indices = np.argsort(pattern_vals)
    patterns = [patterns[i] for i in sorted_indices]
    pattern_vals = [pattern_vals[i] for i in sorted_indices]
    
    # 创建自定义的调色板
    colors = sns.color_palette("RdYlGn_r", len(patterns))
    
    fig, ax = plt.subplots(figsize=(14, 8))
    bars = ax.barh(patterns, pattern_vals, color=colors, alpha=0.85, height=0.6)
    
    # 添加数值标签
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.1, 
            bar.get_y() + bar.get_height()/2, 
            f'{int(width)}', 
            ha='left', 
            va='center', 
            fontweight='bold'
        )
    
    # 设置标签和标题
    ax.set_xlabel('出现次数', fontweight='bold')
    ax.set_title('已识别问题模式分布', fontweight='bold', pad=20)
    
    # 设置网格线只在x轴显示
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, linestyle='--', alpha=0.6)
    
    # 自定义背景
    ax.set_facecolor('#f8f9fa')
    
    # 移除框架
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    plt.tight_layout()
    plt.savefig('LLM_repo/plot/identified_patterns_occurrences.png', dpi=300, bbox_inches='tight')
    plt.close()

# 如果直接运行此脚本，则生成所有图表
if __name__ == "__main__":
    generate_plots()
    print("图表生成完成！保存在 LLM_repo/plot/ 目录下。")
