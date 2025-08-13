#!/usr/bin/env python3
"""
Simple test script for BorutaShapPySpark

This script tests the basic functionality of the PySpark implementation
to ensure it works correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from BorutaShapPySpark import BorutaShapPySpark
from pyspark.sql import SparkSession
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
import warnings

warnings.filterwarnings('ignore')

def test_basic_functionality():
    """Test basic functionality of BorutaShapPySpark."""
    print("Testing BorutaShapPySpark basic functionality...")
    
    try:
        # Create small test dataset
        X, y = make_classification(
            n_samples=1000,
            n_features=20,
            n_informative=5,
            n_redundant=5,
            n_repeated=5,
            random_state=42
        )
        
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        
        # Initialize Spark session
        spark = SparkSession.builder \
            .appName("BorutaShapPySpark_Test") \
            .config("spark.sql.adaptive.enabled", "true") \
            .getOrCreate()
        
        # Convert to Spark DataFrame
        spark_df = spark.createDataFrame(df)
        
        # Initialize feature selector
        feature_selector = BorutaShapPySpark(
            spark_session=spark,
            importance_measure='gini',
            classification=True,
            max_iterations=5,  # Small number for testing
            verbose=False
        )
        
        # Run feature selection
        feature_selector.fit(
            df=spark_df,
            feature_cols=feature_names,
            target_col='target',
            verbose=False
        )
        
        # Check results
        assert hasattr(feature_selector, 'accepted_features'), "Missing accepted_features attribute"
        assert hasattr(feature_selector, 'rejected_features'), "Missing rejected_features attribute"
        assert hasattr(feature_selector, 'tentative_features'), "Missing tentative_features attribute"
        
        # Get feature ranking
        ranking = feature_selector.get_feature_ranking()
        assert isinstance(ranking, dict), "Feature ranking should be a dictionary"
        
        # Test subset creation if features were accepted
        if feature_selector.accepted_features:
            subset_df = feature_selector.subset(spark_df, feature_names)
            assert subset_df.count() == spark_df.count(), "Subset should have same number of rows"
        
        # Cleanup
        feature_selector.stop_spark()
        
        print("✅ All basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        return False

def test_parameter_validation():
    """Test parameter validation."""
    print("Testing parameter validation...")
    
    try:
        # Test with invalid importance measure
        try:
            feature_selector = BorutaShapPySpark(importance_measure='invalid')
            print("❌ Should have raised ValueError for invalid importance measure")
            return False
        except ValueError:
            print("✅ Correctly rejected invalid importance measure")
        
        # Test with valid parameters
        feature_selector = BorutaShapPySpark(
            importance_measure='gini',
            classification=True,
            max_iterations=10
        )
        
        print("✅ Parameter validation tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Parameter validation test failed: {str(e)}")
        return False

def main():
    """Run all tests."""
    print("=== BorutaShapPySpark Test Suite ===\n")
    
    tests = [
        test_parameter_validation,
        test_basic_functionality
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"=== Test Results ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! BorutaShapPySpark is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    exit(main())
