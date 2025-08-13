"""
BorutaShapPySpark - A PySpark implementation of the Boruta algorithm with SHAP values
for large-scale feature selection.

This module provides a distributed version of the BorutaShap algorithm that can handle
large datasets by leveraging PySpark's distributed computing capabilities.
"""

import pyspark.sql.functions as F
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, RegressionEvaluator
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
import warnings
import logging

warnings.filterwarnings("ignore")

class BorutaShapPySpark:
    """
    PySpark implementation of BorutaShap for large-scale feature selection.
    
    This class provides a distributed version of the Boruta algorithm that can handle
    datasets that are too large to fit in memory by leveraging PySpark's distributed
    computing capabilities.
    """
    
    def __init__(self, 
                 spark_session: Optional[SparkSession] = None,
                 model=None, 
                 importance_measure='shap',
                 classification=True, 
                 percentile=100, 
                 pvalue=0.05,
                 max_iterations=100,
                 sample_fraction=0.1,
                 num_workers=None):
        """
        Initialize BorutaShapPySpark.
        
        Parameters
        ----------
        spark_session : SparkSession, optional
            PySpark session. If None, will create a new one.
        model : Model Object, optional
            PySpark ML model. If None, will use RandomForest.
        importance_measure : str
            Which importance measure to use: 'shap', 'gini', or 'permutation'
        classification : bool
            True for classification, False for regression
        percentile : int
            Percentile for shadow feature threshold (0-100)
        pvalue : float
            Significance level for feature selection
        max_iterations : int
            Maximum number of iterations for the algorithm
        sample_fraction : float
            Fraction of data to sample for SHAP calculations
        num_workers : int, optional
            Number of workers for distributed processing
        """
        self.spark = spark_session or self._create_spark_session(num_workers)
        self.model = model
        self.importance_measure = importance_measure.lower()
        self.classification = classification
        self.percentile = percentile
        self.pvalue = pvalue
        self.max_iterations = max_iterations
        self.sample_fraction = sample_fraction
        
        # Results storage
        self.feature_importance_history = []
        self.accepted_features = []
        self.rejected_features = []
        self.tentative_features = []
        self.shadow_features = []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def _create_spark_session(self, num_workers: Optional[int] = None) -> SparkSession:
        """Create a PySpark session with appropriate configuration."""
        builder = SparkSession.builder \
            .appName("BorutaShapPySpark") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.sql.adaptive.skewJoin.enabled", "true")
        
        if num_workers:
            builder = builder.config("spark.executor.instances", str(num_workers))
            
        return builder.getOrCreate()
    
    def _create_shadow_features(self, df: DataFrame, feature_cols: List[str]) -> DataFrame:
        """
        Create shadow features by shuffling original features.
        
        Parameters
        ----------
        df : DataFrame
            Input PySpark DataFrame
        feature_cols : List[str]
            List of feature column names
            
        Returns
        -------
        DataFrame
            DataFrame with shadow features added
        """
        df_with_shadows = df
        
        for col in feature_cols:
            shadow_col = f"shadow_{col}"
            # Create shadow feature by shuffling the original column
            df_with_shadows = df_with_shadows.withColumn(
                shadow_col, 
                F.expr(f"shuffle({col})")
            )
            self.shadow_features.append(shadow_col)
            
        return df_with_shadows
    
    def _prepare_features(self, df: DataFrame, feature_cols: List[str]) -> DataFrame:
        """
        Prepare features for ML pipeline by assembling them into a vector.
        
        Parameters
        ----------
        df : DataFrame
            Input PySpark DataFrame
        feature_cols : List[str]
            List of feature column names
            
        Returns
        -------
        DataFrame
            DataFrame with features assembled into a vector column
        """
        assembler = VectorAssembler(
            inputCols=feature_cols,
            outputCol="features",
            handleInvalid="skip"
        )
        
        return assembler.transform(df)
    
    def _get_model(self, feature_cols: List[str]):
        """
        Get or create the ML model for training.
        
        Parameters
        ----------
        feature_cols : List[str]
            List of feature column names
            
        Returns
        -------
        Pipeline
            PySpark ML pipeline with the model
        """
        if self.model is None:
            if self.classification:
                model = RandomForestClassifier(
                    featuresCol="features",
                    labelCol="label",
                    numTrees=100,
                    maxDepth=10
                )
            else:
                model = RandomForestRegressor(
                    featuresCol="features",
                    labelCol="label",
                    numTrees=100,
                    maxDepth=10
                )
        else:
            model = self.model
            
        return Pipeline(stages=[model])
    
    def _calculate_feature_importance(self, 
                                    model, 
                                    df: DataFrame, 
                                    feature_cols: List[str]) -> Dict[str, float]:
        """
        Calculate feature importance using the specified method.
        
        Parameters
        ----------
        model : Pipeline
            Trained PySpark ML pipeline
        df : DataFrame
            Input PySpark DataFrame
        feature_cols : List[str]
            List of feature column names
            
        Returns
        -------
        Dict[str, float]
            Dictionary mapping feature names to importance scores
        """
        if self.importance_measure == 'gini':
            return self._calculate_gini_importance(model, df, feature_cols)
        elif self.importance_measure == 'shap':
            return self._calculate_shap_importance(model, df, feature_cols)
        elif self.importance_measure == 'permutation':
            return self._calculate_permutation_importance(model, df, feature_cols)
        else:
            raise ValueError(f"Unsupported importance measure: {self.importance_measure}")
    
    def _calculate_gini_importance(self, 
                                 model, 
                                 df: DataFrame, 
                                 feature_cols: List[str]) -> Dict[str, float]:
        """Calculate Gini importance from the trained model."""
        # Extract the actual model from the pipeline
        actual_model = model.stages[-1]
        
        # Get feature importances
        importances = actual_model.featureImportances.toArray()
        
        # Create feature importance dictionary
        importance_dict = {}
        for i, col in enumerate(feature_cols):
            importance_dict[col] = float(importances[i])
            
        return importance_dict
    
    def _calculate_shap_importance(self, 
                                 model, 
                                 df: DataFrame, 
                                 feature_cols: List[str]) -> Dict[str, float]:
        """
        Calculate SHAP importance using sampling for large datasets.
        
        Note: This is a simplified implementation. For full SHAP support,
        you may need to use libraries like 'shap' with sampled data.
        """
        # Sample data for SHAP calculation to handle large datasets
        sampled_df = df.sample(fraction=self.sample_fraction, seed=42)
        
        # Convert to pandas for SHAP calculation
        pandas_df = sampled_df.toPandas()
        
        # This is a placeholder - you'll need to implement actual SHAP calculation
        # For now, we'll use a simplified approach
        importance_dict = {}
        for col in feature_cols:
            # Simplified importance calculation
            importance_dict[col] = np.random.random()
            
        return importance_dict
    
    def _calculate_permutation_importance(self, 
                                        model, 
                                        df: DataFrame, 
                                        feature_cols: List[str]) -> Dict[str, float]:
        """Calculate permutation importance."""
        # Sample data for permutation importance calculation
        sampled_df = df.sample(fraction=self.sample_fraction, seed=42)
        
        # This is a placeholder - implement actual permutation importance
        importance_dict = {}
        for col in feature_cols:
            importance_dict[col] = np.random.random()
            
        return importance_dict
    
    def fit(self, 
            df: DataFrame, 
            feature_cols: List[str], 
            target_col: str,
            verbose: bool = True) -> 'BorutaShapPySpark':
        """
        Fit the Boruta algorithm to the data.
        
        Parameters
        ----------
        df : DataFrame
            Input PySpark DataFrame
        feature_cols : List[str]
            List of feature column names
        target_col : str
            Name of the target column
        verbose : bool
            Whether to print progress information
            
        Returns
        -------
        BorutaShapPySpark
            Self for method chaining
        """
        if verbose:
            self.logger.info(f"Starting BorutaShapPySpark with {len(feature_cols)} features")
            self.logger.info(f"Dataset size: {df.count()} rows")
        
        # Rename target column to 'label' for consistency
        df = df.withColumnRenamed(target_col, "label")
        
        # Initialize feature lists
        all_features = feature_cols.copy()
        tentative_features = feature_cols.copy()
        
        iteration = 0
        
        while iteration < self.max_iterations and tentative_features:
            if verbose:
                self.logger.info(f"Iteration {iteration + 1}/{self.max_iterations}")
                self.logger.info(f"Tentative features: {len(tentative_features)}")
            
            # Create shadow features
            df_with_shadows = self._create_shadow_features(df, tentative_features)
            
            # Prepare features for ML
            all_current_features = tentative_features + self.shadow_features
            df_prepared = self._prepare_features(df_with_shadows, all_current_features)
            
            # Train model
            model = self._get_model(all_current_features)
            fitted_model = model.fit(df_prepared)
            
            # Calculate feature importance
            importance_dict = self._calculate_feature_importance(
                fitted_model, df_prepared, all_current_features
            )
            
            # Store importance history
            self.feature_importance_history.append(importance_dict)
            
            # Calculate shadow threshold
            shadow_importances = [importance_dict[col] for col in self.shadow_features]
            shadow_threshold = np.percentile(shadow_importances, self.percentile)
            
            # Evaluate features
            new_accepted = []
            new_rejected = []
            
            for feature in tentative_features:
                importance = importance_dict[feature]
                
                if importance > shadow_threshold:
                    new_accepted.append(feature)
                else:
                    new_rejected.append(feature)
            
            # Update feature lists
            self.accepted_features.extend(new_accepted)
            self.rejected_features.extend(new_rejected)
            
            # Remove processed features from tentative list
            tentative_features = [f for f in tentative_features 
                                if f not in new_accepted and f not in new_rejected]
            
            if verbose:
                self.logger.info(f"Accepted: {len(new_accepted)}, Rejected: {len(new_rejected)}")
            
            iteration += 1
        
        # Handle remaining tentative features
        if tentative_features:
            self.tentative_features = tentative_features
            if verbose:
                self.logger.info(f"Remaining tentative features: {len(tentative_features)}")
        
        if verbose:
            self.logger.info(f"Final results:")
            self.logger.info(f"Accepted features: {len(self.accepted_features)}")
            self.logger.info(f"Rejected features: {len(self.rejected_features)}")
            self.logger.info(f"Tentative features: {len(self.tentative_features)}")
        
        return self
    
    def get_feature_ranking(self) -> Dict[str, str]:
        """
        Get the final feature ranking and status.
        
        Returns
        -------
        Dict[str, str]
            Dictionary mapping feature names to their status
        """
        ranking = {}
        
        for feature in self.accepted_features:
            ranking[feature] = "accepted"
            
        for feature in self.rejected_features:
            ranking[feature] = "rejected"
            
        for feature in self.tentative_features:
            ranking[feature] = "tentative"
            
        return ranking
    
    def subset(self, df: DataFrame, feature_cols: List[str]) -> DataFrame:
        """
        Get a subset of the original data with only the accepted features.
        
        Parameters
        ----------
        df : DataFrame
            Original PySpark DataFrame
        feature_cols : List[str]
            List of all feature column names
            
        Returns
        -------
        DataFrame
            DataFrame with only accepted features
        """
        if not self.accepted_features:
            raise ValueError("No features have been accepted. Run fit() first.")
        
        return df.select(self.accepted_features + [col for col in df.columns 
                                                  if col not in feature_cols])
    
    def stop_spark(self):
        """Stop the Spark session."""
        if self.spark:
            self.spark.stop()
