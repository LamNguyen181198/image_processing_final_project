#!/usr/bin/env python3
"""
Visualize training data distribution and feature statistics.

Usage:
    python visualize_training_data.py training_data_features.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path


def visualize_dataset(csv_path):
    """Create visualizations of the training dataset"""
    
    csv_path = Path(csv_path).resolve()
    df = pd.read_csv(csv_path)
    
    print("=" * 80)
    print(f"DATASET: {csv_path.name}")
    print("=" * 80)
    print(f"Total samples: {len(df)}")
    print(f"Total features: {len(df.columns) - 2}")  # Excluding filename and label
    print(f"\nLabel distribution:")
    print(df['label'].value_counts().to_string())
    print()
    
    # Create output directory for plots
    output_dir = csv_path.parent / "visualizations"
    output_dir.mkdir(exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # 1. Label distribution bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    label_counts = df['label'].value_counts()
    colors = sns.color_palette("husl", len(label_counts))
    label_counts.plot(kind='bar', ax=ax, color=colors)
    ax.set_title('Distribution of Noise Types in Dataset', fontsize=16, fontweight='bold')
    ax.set_xlabel('Noise Type', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Add count labels on bars
    for i, v in enumerate(label_counts):
        ax.text(i, v + 0.3, str(v), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plot_path = output_dir / "label_distribution.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Saved: {plot_path}")
    plt.close()
    
    # 2. Feature statistics summary
    feature_cols = [col for col in df.columns if col not in ['filename', 'label']]
    
    print(f"\nFeature Statistics:")
    print("=" * 80)
    stats = df[feature_cols].describe()
    print(stats.to_string())
    
    # 3. Feature correlation heatmap (top 15 features)
    fig, ax = plt.subplots(figsize=(14, 12))
    top_features = feature_cols[:15]  # First 15 features
    corr = df[top_features].corr()
    
    sns.heatmap(corr, annot=False, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title('Feature Correlation Heatmap (Top 15 Features)', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plot_path = output_dir / "feature_correlation.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Saved: {plot_path}")
    plt.close()
    
    # 4. Box plots for key discriminative features by noise type
    key_features = ['kurtosis', 'skewness', 'noise_variance', 'salt_pepper_score', 
                    'impulse_ratio', 'var_mean_ratio']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, feature in enumerate(key_features):
        if feature in df.columns:
            sns.boxplot(data=df, x='label', y=feature, ax=axes[idx], palette='Set2')
            axes[idx].set_title(f'{feature}', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('')
            axes[idx].set_xticklabels(axes[idx].get_xticklabels(), rotation=45, ha='right')
    
    plt.suptitle('Key Features Distribution by Noise Type', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plot_path = output_dir / "feature_boxplots.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Saved: {plot_path}")
    plt.close()
    
    # 5. Scatter plot: variance-mean relationship by noise type
    if 'r2_linear' in df.columns and 'r2_quadratic' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for label in df['label'].unique():
            subset = df[df['label'] == label]
            ax.scatter(subset['r2_linear'], subset['r2_quadratic'], 
                      label=label, alpha=0.7, s=100, edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel('R² Linear Fit (Var vs Mean)', fontsize=12)
        ax.set_ylabel('R² Quadratic Fit (Var vs Mean)', fontsize=12)
        ax.set_title('Variance-Mean Relationship Analysis', fontsize=16, fontweight='bold')
        ax.legend(title='Noise Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = output_dir / "variance_mean_scatter.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Saved: {plot_path}")
        plt.close()
    
    print("\n" + "=" * 80)
    print(f"All visualizations saved to: {output_dir}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Visualize training dataset distribution and features'
    )
    parser.add_argument('csv_file', help='Path to the features CSV file')
    
    args = parser.parse_args()
    
    try:
        visualize_dataset(args.csv_file)
        return 0
    except Exception as e:
        print(f"\n[X] Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
