#!/usr/bin/env python3
"""
Split training data CSV into train and test sets.

This script:
1. Reads the features CSV file
2. Splits data stratified by noise type (ensures each type in both sets)
3. Saves separate train.csv and test.csv files

Usage:
    python split_train_test.py training_data_features.csv --test-size 0.2
"""

import pandas as pd
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split


def split_dataset(csv_path, test_size=0.2, random_seed=42):
    """Split dataset into train and test sets with stratification"""
    
    csv_path = Path(csv_path).resolve()
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    print("=" * 80)
    print("SPLITTING DATASET INTO TRAIN/TEST")
    print("=" * 80)
    print(f"Input: {csv_path}")
    print(f"Test size: {test_size * 100:.0f}%")
    print(f"Random seed: {random_seed}\n")
    
    # Load dataset
    df = pd.read_csv(csv_path)
    
    print(f"Total samples: {len(df)}")
    print(f"\nLabel distribution:")
    print(df['label'].value_counts().to_string())
    print()
    
    # Check if we have enough samples per class for splitting
    label_counts = df['label'].value_counts()
    min_samples = label_counts.min()
    
    if min_samples < 2:
        print(f"WARNING: Some labels have only {min_samples} sample(s).")
        print("Stratified split may not work well. Consider generating more data.")
        print()
    
    # Split with stratification to maintain label distribution
    try:
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_seed,
            stratify=df['label']
        )
        stratified = True
    except ValueError as e:
        print(f"Could not perform stratified split: {e}")
        print("Falling back to random split without stratification.\n")
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_seed
        )
        stratified = False
    
    # Generate output filenames
    output_dir = csv_path.parent
    base_name = csv_path.stem
    
    train_path = output_dir / f"{base_name}_train.csv"
    test_path = output_dir / f"{base_name}_test.csv"
    
    # Save splits
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print("=" * 80)
    print("SPLIT SUMMARY")
    print("=" * 80)
    print(f"Train set: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Saved to: {train_path}")
    print(f"\nTrain label distribution:")
    print(train_df['label'].value_counts().to_string())
    
    print(f"\n{'-' * 80}")
    print(f"Test set: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
    print(f"  Saved to: {test_path}")
    print(f"\nTest label distribution:")
    print(test_df['label'].value_counts().to_string())
    
    print("=" * 80)
    
    if stratified:
        print("[OK] Stratified split successful - label proportions preserved")
    else:
        print("[!] Random split used - stratification not possible")
    
    print("=" * 80)
    
    return train_df, test_df


def main():
    parser = argparse.ArgumentParser(
        description='Split training dataset into train and test sets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 80/20 train/test split (default)
  python split_train_test.py training_data_features.csv
  
  # 70/30 train/test split
  python split_train_test.py training_data_features.csv --test-size 0.3
  
  # Custom random seed for reproducibility
  python split_train_test.py training_data_features.csv --seed 123
        """
    )
    
    parser.add_argument('csv_file', 
                        help='Path to the CSV file with training data')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Proportion of data for test set (0.0 to 1.0, default: 0.2)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Validate test_size
    if not 0.0 < args.test_size < 1.0:
        parser.error(f"test-size must be between 0.0 and 1.0, got {args.test_size}")
    
    try:
        train_df, test_df = split_dataset(args.csv_file, args.test_size, args.seed)
        return 0
    except Exception as e:
        print(f"\n[X] Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
