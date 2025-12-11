#!/usr/bin/env python3
"""
Train Random Forest Classifier for Noise Type Detection

This script:
1. Loads training data from CSV
2. Trains a Random Forest classifier
3. Evaluates performance with cross-validation
4. Tests on hold-out test set
5. Generates performance reports and visualizations
6. Saves the trained model

Usage:
    python train_random_forest.py
    python train_random_forest.py --train-csv ../training_data_features_train.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import joblib
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    f1_score
)

import warnings
warnings.filterwarnings('ignore')


def load_data(train_csv, test_csv):
    """Load training and test datasets"""
    
    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    
    # Separate features and labels
    feature_cols = [col for col in train_df.columns if col not in ['filename', 'label']]
    
    X_train = train_df[feature_cols].values
    y_train = train_df['label'].values
    
    X_test = test_df[feature_cols].values
    y_test = test_df['label'].values
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {len(feature_cols)}")
    print(f"Classes: {len(np.unique(y_train))}")
    
    print(f"\nTraining label distribution:")
    for label, count in zip(*np.unique(y_train, return_counts=True)):
        print(f"  {label:15s}: {count:2d} samples")
    
    print(f"\nTest label distribution:")
    for label, count in zip(*np.unique(y_test, return_counts=True)):
        print(f"  {label:15s}: {count:2d} samples")
    
    return X_train, y_train, X_test, y_test, feature_cols


def train_model(X_train, y_train):
    """Train Random Forest classifier"""
    
    print("\n" + "=" * 80)
    print("TRAINING RANDOM FOREST CLASSIFIER")
    print("=" * 80)
    
    # Initialize Random Forest with balanced class weights
    rf_model = RandomForestClassifier(
        n_estimators=100,           # 100 trees in the forest
        max_depth=None,             # Nodes expanded until pure
        min_samples_split=2,        # Minimum samples to split node
        min_samples_leaf=1,         # Minimum samples in leaf
        class_weight='balanced',    # Handle class imbalance
        random_state=42,            # Reproducibility
        n_jobs=-1                   # Use all CPU cores
    )
    
    print("\nModel Configuration:")
    print(f"  - Trees: {rf_model.n_estimators}")
    print(f"  - Class weights: {rf_model.class_weight}")
    print(f"  - Random state: {rf_model.random_state}")
    
    print("\nTraining model...")
    rf_model.fit(X_train, y_train)
    print("[OK] Model trained successfully")
    
    return rf_model


def evaluate_with_cv(model, X_train, y_train, cv=5):
    """Evaluate model using cross-validation"""
    
    print("\n" + "=" * 80)
    print(f"CROSS-VALIDATION ({cv}-FOLD)")
    print("=" * 80)
    
    # Perform cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    cv_f1_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_weighted')
    
    print(f"\nAccuracy scores per fold: {cv_scores}")
    print(f"Mean accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print(f"\nF1 scores per fold: {cv_f1_scores}")
    print(f"Mean F1 score: {cv_f1_scores.mean():.4f} (+/- {cv_f1_scores.std() * 2:.4f})")
    
    # Get cross-validated predictions for confusion matrix
    y_pred_cv = cross_val_predict(model, X_train, y_train, cv=cv)
    
    return cv_scores, y_pred_cv


def evaluate_on_test(model, X_test, y_test):
    """Evaluate model on test set"""
    
    print("\n" + "=" * 80)
    print("TEST SET EVALUATION")
    print("=" * 80)
    
    # Predict on test set
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Test F1 Score: {f1:.4f}")
    
    print("\n" + "-" * 80)
    print("CLASSIFICATION REPORT")
    print("-" * 80)
    print(classification_report(y_test, y_pred, zero_division=0))
    
    return y_pred, accuracy, f1


def plot_confusion_matrix(y_true, y_pred, classes, title, save_path):
    """Plot confusion matrix"""
    
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Count'})
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Saved: {save_path}")
    plt.close()


def plot_feature_importance(model, feature_names, save_path, top_n=15):
    """Plot feature importance"""
    
    # Get feature importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Select top N features
    top_indices = indices[:top_n]
    top_features = [feature_names[i] for i in top_indices]
    top_importances = importances[top_indices]
    
    # Plot
    plt.figure(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, top_n))
    plt.barh(range(top_n), top_importances[::-1], color=colors[::-1])
    plt.yticks(range(top_n), top_features[::-1])
    plt.xlabel('Feature Importance', fontsize=12)
    plt.title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Saved: {save_path}")
    plt.close()
    
    # Print top features
    print("\n" + "-" * 80)
    print(f"TOP {top_n} MOST IMPORTANT FEATURES")
    print("-" * 80)
    for i, idx in enumerate(top_indices, 1):
        print(f"{i:2d}. {feature_names[idx]:30s} {importances[idx]:.4f}")


def plot_performance_summary(cv_scores, test_accuracy, save_path):
    """Plot performance summary"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Cross-validation scores
    ax1.plot(range(1, len(cv_scores) + 1), cv_scores, 'o-', linewidth=2, markersize=8)
    ax1.axhline(y=cv_scores.mean(), color='r', linestyle='--', 
                label=f'Mean: {cv_scores.mean():.4f}')
    ax1.fill_between(range(1, len(cv_scores) + 1), 
                      cv_scores.mean() - cv_scores.std(), 
                      cv_scores.mean() + cv_scores.std(), 
                      alpha=0.2, color='red')
    ax1.set_xlabel('Fold', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Cross-Validation Accuracy', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])
    
    # Train vs Test comparison
    metrics = ['CV Mean', 'Test']
    scores = [cv_scores.mean(), test_accuracy]
    colors = ['#2ecc71', '#3498db']
    
    bars = ax2.bar(metrics, scores, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training vs Test Performance', fontsize=14, fontweight='bold')
    ax2.set_ylim([0, 1.05])
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.4f}\n({score*100:.2f}%)',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Saved: {save_path}")
    plt.close()


def save_model(model, feature_names, model_path, metadata):
    """Save trained model and metadata"""
    
    print("\n" + "=" * 80)
    print("SAVING MODEL")
    print("=" * 80)
    
    # Save model
    model_data = {
        'model': model,
        'feature_names': feature_names,
        'metadata': metadata
    }
    
    joblib.dump(model_data, model_path)
    print(f"[OK] Model saved to: {model_path}")
    
    # Print model info
    print(f"\nModel Information:")
    print(f"  - Type: Random Forest Classifier")
    print(f"  - Trees: {model.n_estimators}")
    print(f"  - Features: {len(feature_names)}")
    print(f"  - Classes: {len(model.classes_)}")
    print(f"  - Training accuracy: {metadata['cv_accuracy']:.4f}")
    print(f"  - Test accuracy: {metadata['test_accuracy']:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='Train Random Forest classifier for noise detection',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--train-csv', 
                        default='../training_data_features_train.csv',
                        help='Path to training CSV file')
    parser.add_argument('--test-csv', 
                        default='../training_data_features_test.csv',
                        help='Path to test CSV file')
    parser.add_argument('--output-dir', 
                        default='.',
                        help='Directory to save model and plots')
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    script_dir = Path(__file__).parent.resolve()
    train_csv = (script_dir / args.train_csv).resolve()
    test_csv = (script_dir / args.test_csv).resolve()
    output_dir = (script_dir / args.output_dir).resolve()
    output_dir.mkdir(exist_ok=True)
    
    print("\n" + "=" * 80)
    print("RANDOM FOREST NOISE CLASSIFIER TRAINING")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Training data: {train_csv}")
    print(f"Test data: {test_csv}")
    print(f"Output directory: {output_dir}")
    
    # Load data
    X_train, y_train, X_test, y_test, feature_names = load_data(train_csv, test_csv)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Cross-validation
    cv_scores, y_pred_cv = evaluate_with_cv(model, X_train, y_train, cv=5)
    
    # Test evaluation
    y_pred_test, test_accuracy, test_f1 = evaluate_on_test(model, X_test, y_test)
    
    # Get all unique classes
    all_classes = sorted(np.unique(np.concatenate([y_train, y_test])))
    
    # Plot confusion matrix for cross-validation
    plot_confusion_matrix(
        y_train, y_pred_cv, all_classes,
        'Confusion Matrix - Cross-Validation (Training Set)',
        output_dir / 'confusion_matrix_cv.png'
    )
    
    # Plot confusion matrix for test set
    plot_confusion_matrix(
        y_test, y_pred_test, all_classes,
        'Confusion Matrix - Test Set',
        output_dir / 'confusion_matrix_test.png'
    )
    
    # Plot feature importance
    plot_feature_importance(
        model, feature_names,
        output_dir / 'feature_importance.png',
        top_n=15
    )
    
    # Plot performance summary
    plot_performance_summary(
        cv_scores, test_accuracy,
        output_dir / 'performance_summary.png'
    )
    
    # Save model
    metadata = {
        'train_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'num_features': len(feature_names),
        'num_classes': len(all_classes),
        'classes': list(all_classes),
        'cv_accuracy': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'test_accuracy': test_accuracy,
        'test_f1': test_f1
    }
    
    model_path = output_dir / 'random_forest_model.pkl'
    save_model(model, feature_names, model_path, metadata)
    
    # Final summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nFinal Results:")
    print(f"  - Cross-validation accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  - Test accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"  - Test F1 score: {test_f1:.4f}")
    
    print(f"\nGenerated Files:")
    print(f"  - Model: {model_path}")
    print(f"  - CV confusion matrix: {output_dir / 'confusion_matrix_cv.png'}")
    print(f"  - Test confusion matrix: {output_dir / 'confusion_matrix_test.png'}")
    print(f"  - Feature importance: {output_dir / 'feature_importance.png'}")
    print(f"  - Performance summary: {output_dir / 'performance_summary.png'}")
    
    print("\n" + "=" * 80)
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
