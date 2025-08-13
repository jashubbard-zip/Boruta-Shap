#!/usr/bin/env python3
"""
PySpark Example for BorutaShapPySpark

This script demonstrates how to use the PySpark implementation of BorutaShap
for large-scale feature selection.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from BorutaShapPySpark import BorutaShapPySpark
from pyspark.sql import SparkSession
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
import warnings

warnings.filterwarnings('ignore')

def main():
    """Main function demonstrating PySpark BorutaShap usage."""
    
    print("=== BorutaShapPySpark Example ===\n")
    
    # 1. Create sample data
    print("1. Creating sample dataset...")
    X_class, y_class = make_classification(
        n_samples=10000,  # Large dataset
        n_features=50,    # 50 features
        n_informative=10, # Only 10 are actually informative
        n_redundant=20,   # 20 are redundant
        n_repeated=10,    # 10 are repeated
        n_clusters_per_class=2,
        random_state=42
    )
    
    feature_names = [f'feature_{i}' for i in range(X_class.shape[1])]
    df_class = pd.DataFrame(X_class, columns=feature_names)
    df_class['target'] = y_class
    
    print(f"   Dataset shape: {df_class.shape}")
    print(f"   Number of features: {len(feature_names)}")
    print(f"   Target distribution: {df_class['target'].value_counts().to_dict()}\n")
    
    # 2. Initialize PySpark session
    print("2. Initializing PySpark session...")
    spark = SparkSession.builder \
        .appName("BorutaShapPySpark_Example") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .getOrCreate()
    
    print(f"   PySpark version: {spark.version}")
    print(f"   Spark UI: {spark.sparkContext.uiWebUrl}\n")
    
    # 3. Convert to PySpark DataFrame
    print("3. Converting to PySpark DataFrame...")
    spark_df = spark.createDataFrame(df_class)
    print(f"   PySpark DataFrame created with {spark_df.count()} rows")
    print(f"   Number of partitions: {spark_df.rdd.getNumPartitions()}\n")
    
    # 4. Initialize BorutaShapPySpark
    print("4. Initializing BorutaShapPySpark...")
    feature_selector = BorutaShapPySpark(
        spark_session=spark,
        importance_measure='gini',  # Use Gini importance for faster processing
        classification=True,
        percentile=100,
        pvalue=0.05,
        max_iterations=15,  # Limit iterations for demo
        sample_fraction=0.1,  # Sample 10% for calculations
        verbose=True
    )
    print("   BorutaShapPySpark initialized successfully!\n")
    
    # 5. Run feature selection
    print("5. Running feature selection...")
    feature_selector.fit(
        df=spark_df,
        feature_cols=feature_names,
        target_col='target',
        verbose=True
    )
    print("   Feature selection completed!\n")
    
    # 6. Analyze results
    print("6. Analyzing results...")
    feature_ranking = feature_selector.get_feature_ranking()
    
    # Count features by status
    status_counts = {}
    for status in feature_ranking.values():
        status_counts[status] = status_counts.get(status, 0) + 1
    
    print("   Feature Selection Results:")
    print("   " + "=" * 40)
    for status, count in status_counts.items():
        print(f"   {status.capitalize()} features: {count}")
    
    print(f"\n   Accepted features: {feature_selector.accepted_features}")
    print(f"   Rejected features: {feature_selector.rejected_features}")
    print(f"   Tentative features: {feature_selector.tentative_features}\n")
    
    # 7. Create feature subset
    print("7. Creating feature subset...")
    if feature_selector.accepted_features:
        subset_df = feature_selector.subset(spark_df, feature_names)
        print(f"   Original dataset columns: {len(spark_df.columns)}")
        print(f"   Subset dataset columns: {len(subset_df.columns)}")
        print(f"   Subset columns: {subset_df.columns}")
        
        # Show sample of subset data
        print("\n   Sample of subset data:")
        subset_df.show(3)
    else:
        print("   No features were accepted. Consider adjusting parameters.\n")
    
    # 8. Performance analysis
    print("8. Performance analysis...")
    if feature_selector.feature_importance_history:
        print(f"   Number of iterations completed: {len(feature_selector.feature_importance_history)}")
        
        # Show importance evolution for first few accepted features
        if feature_selector.accepted_features:
            print("   Importance evolution for first 3 accepted features:")
            for feature in feature_selector.accepted_features[:3]:
                importances = [history.get(feature, 0) for history in feature_selector.feature_importance_history]
                print(f"   {feature}: {importances}")
    else:
        print("   No importance history available.\n")
    
    # 9. Cleanup
    print("9. Cleaning up...")
    feature_selector.stop_spark()
    print("   Spark session stopped successfully.\n")
    
    print("=== Example completed successfully! ===")
    
    # Summary
    print("\nSummary:")
    print("-" * 30)
    print(f"• Dataset size: {df_class.shape[0]} rows, {df_class.shape[1]-1} features")
    print(f"• Features accepted: {len(feature_selector.accepted_features)}")
    print(f"• Features rejected: {len(feature_selector.rejected_features)}")
    print(f"• Features tentative: {len(feature_selector.tentative_features)}")
    print(f"• Iterations completed: {len(feature_selector.feature_importance_history)}")
    
    if feature_selector.accepted_features:
        print(f"• Reduction in features: {len(feature_names) - len(feature_selector.accepted_features)} features removed")
        print(f"• Feature reduction percentage: {((len(feature_names) - len(feature_selector.accepted_features)) / len(feature_names) * 100):.1f}%")

if __name__ == "__main__":
    main()
