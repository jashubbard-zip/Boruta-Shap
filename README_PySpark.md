# BorutaShapPySpark - Large-Scale Feature Selection with PySpark

## Overview

BorutaShapPySpark is a distributed implementation of the Boruta algorithm with SHAP values, designed to handle large-scale datasets that don't fit in memory. This implementation leverages PySpark's distributed computing capabilities to provide scalable feature selection for big data applications.

## Key Features

- **Distributed Processing**: Leverages PySpark for handling datasets that exceed memory limits
- **Distributed SHAP**: True distributed SHAP calculation without pulling data into memory
- **Multiple Importance Measures**: Support for SHAP, Gini importance, and permutation importance
- **Memory Efficient**: Processes data in chunks and uses sampling for large datasets
- **Scalable**: Can run on clusters with multiple nodes
- **TreeSHAP Optimization**: Optimized TreeSHAP for tree-based models
- **Flexible**: Compatible with various PySpark ML models
- **Easy Integration**: Simple API similar to the original BorutaShap

## Installation

### Prerequisites

1. **Java 8 or later** (required for PySpark)
2. **Python 3.7+**
3. **PySpark 3.4.0+**

### Install Dependencies

```bash
# Install PySpark and core dependencies
pip install -r requirements-pyspark.txt

# Or install manually
pip install pyspark>=3.4.0 py4j>=0.10.9
pip install scikit-learn pandas numpy scipy shap matplotlib seaborn tqdm statsmodels
```

### Optional Dependencies

```bash
# For better performance
pip install findspark>=1.4.2

# For testing
pip install pytest pytest-cov
```

## Quick Start

### Basic Usage

```python
from BorutaShapPySpark import BorutaShapPySpark
from pyspark.sql import SparkSession
import pandas as pd

# Initialize Spark session
spark = SparkSession.builder \
    .appName("BorutaShapPySpark") \
    .getOrCreate()

# Load your data (example with pandas DataFrame)
df = pd.read_csv('your_large_dataset.csv')
spark_df = spark.createDataFrame(df)

# Initialize feature selector
feature_selector = BorutaShapPySpark(
    spark_session=spark,
    importance_measure='gini',  # or 'shap', 'permutation'
    classification=True,  # False for regression
    max_iterations=100,
    verbose=True
)

# Run feature selection
feature_selector.fit(
    df=spark_df,
    feature_cols=['feature1', 'feature2', 'feature3', ...],
    target_col='target'
)

# Get results
accepted_features = feature_selector.accepted_features
rejected_features = feature_selector.rejected_features
feature_ranking = feature_selector.get_feature_ranking()

# Get subset with only accepted features
subset_df = feature_selector.subset(spark_df, feature_cols)

# Cleanup
feature_selector.stop_spark()
```

### Advanced Configuration

```python
# Custom Spark configuration for large datasets
spark = SparkSession.builder \
    .appName("BorutaShapPySpark") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .config("spark.sql.adaptive.skewJoin.enabled", "true") \
    .config("spark.executor.memory", "8g") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "200") \
    .getOrCreate()

# Advanced feature selector configuration
feature_selector = BorutaShapPySpark(
    spark_session=spark,
    model=your_custom_model,  # Custom PySpark ML model
    importance_measure='shap',
    classification=True,
    percentile=95,  # More lenient threshold
    pvalue=0.01,    # More strict significance level
    max_iterations=200,
    sample_fraction=0.05,  # Sample 5% for SHAP calculations
    num_workers=4   # Specify number of workers
)
```

## API Reference

### BorutaShapPySpark Class

#### Constructor Parameters

- `spark_session` (SparkSession, optional): PySpark session. If None, creates a new one.
- `model` (Model Object, optional): PySpark ML model. If None, uses RandomForest.
- `importance_measure` (str): 'shap', 'gini', or 'permutation'
- `classification` (bool): True for classification, False for regression
- `percentile` (int): Percentile for shadow feature threshold (0-100)
- `pvalue` (float): Significance level for feature selection
- `max_iterations` (int): Maximum number of algorithm iterations
- `sample_fraction` (float): Fraction of data to sample for calculations
- `num_workers` (int, optional): Number of workers for distributed processing

#### Methods

- `fit(df, feature_cols, target_col, verbose=True)`: Run the feature selection algorithm
- `get_feature_ranking()`: Get final feature ranking and status
- `subset(df, feature_cols)`: Get subset with only accepted features
- `stop_spark()`: Stop the Spark session

#### Attributes

- `accepted_features`: List of accepted feature names
- `rejected_features`: List of rejected feature names
- `tentative_features`: List of tentative feature names
- `feature_importance_history`: List of importance dictionaries for each iteration

## Distributed SHAP Implementation

### How It Works

The distributed SHAP implementation uses several key techniques to avoid pulling large datasets into memory:

1. **Background Sampling**: Creates a small background dataset (typically 1% of data) for SHAP baseline
2. **Distributed UDFs**: Uses PySpark UDFs to calculate SHAP values on each partition
3. **TreeSHAP Optimization**: For tree-based models, uses TreeSHAP which is more efficient than KernelSHAP
4. **Batch Processing**: Processes data in batches across cluster nodes
5. **Aggregation**: Aggregates SHAP values across partitions to compute feature importance

### SHAP Calculation Modes

- **TreeSHAP**: Used automatically for RandomForest models (fastest)
- **KernelSHAP**: Used for other model types (more flexible but slower)
- **Fallback**: Falls back to Gini importance if SHAP calculation fails

### Example: Distributed SHAP

```python
# Use distributed SHAP - no data pulled into memory
feature_selector = BorutaShapPySpark(
    spark_session=spark,
    importance_measure='shap',  # Distributed SHAP
    sample_fraction=0.05,       # 5% sample for SHAP calculation
    classification=True
)

# Process large dataset without memory issues
feature_selector.fit(df=large_spark_df, feature_cols=features, target_col='target')
```

## Performance Considerations

### Memory Management

- **Sample Fraction**: Use smaller `sample_fraction` values (0.01-0.1) for very large datasets
- **Background Size**: Automatically optimized but can be tuned for specific use cases
- **Partitioning**: Adjust Spark partitions based on your cluster size
- **Caching**: Cache frequently used DataFrames with `df.cache()`

### Scaling Tips

1. **Cluster Configuration**:
   ```python
   # For large datasets, configure more memory and workers
   spark = SparkSession.builder \
       .config("spark.executor.memory", "16g") \
       .config("spark.driver.memory", "8g") \
       .config("spark.sql.shuffle.partitions", "400") \
       .getOrCreate()
   ```

2. **Importance Measure Selection**:
   - Use `'gini'` for fastest processing
   - Use `'shap'` for most accurate results (but slower)
   - Use `'permutation'` for balanced performance

3. **Iteration Management**:
   - Start with fewer iterations (10-20) for testing
   - Increase for production runs based on convergence

## Examples

### Classification Example

```python
from sklearn.datasets import make_classification
import pandas as pd

# Create sample classification data
X, y = make_classification(n_samples=100000, n_features=100, n_informative=20)
df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(100)])
df['target'] = y

# Convert to Spark DataFrame
spark_df = spark.createDataFrame(df)

# Run feature selection
feature_selector = BorutaShapPySpark(
    spark_session=spark,
    importance_measure='gini',
    classification=True,
    max_iterations=50
)

feature_selector.fit(
    df=spark_df,
    feature_cols=[f'feature_{i}' for i in range(100)],
    target_col='target'
)

print(f"Accepted features: {len(feature_selector.accepted_features)}")
```

### Regression Example

```python
from sklearn.datasets import make_regression

# Create sample regression data
X, y = make_regression(n_samples=50000, n_features=50, n_informative=15)
df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(50)])
df['target'] = y

spark_df = spark.createDataFrame(df)

feature_selector = BorutaShapPySpark(
    spark_session=spark,
    importance_measure='shap',
    classification=False,  # Regression
    sample_fraction=0.1
)

feature_selector.fit(
    df=spark_df,
    feature_cols=[f'feature_{i}' for i in range(50)],
    target_col='target'
)
```

## Troubleshooting

### Common Issues

1. **Memory Errors**:
   - Reduce `sample_fraction`
   - Increase Spark executor memory
   - Use fewer partitions

2. **Slow Performance**:
   - Use `'gini'` importance measure
   - Reduce `max_iterations`
   - Increase cluster resources

3. **No Features Accepted**:
   - Increase `percentile` (make more lenient)
   - Increase `pvalue` (make more lenient)
   - Check data quality and feature-target relationships

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

feature_selector = BorutaShapPySpark(
    spark_session=spark,
    verbose=True
)
```

## Comparison with Original BorutaShap

| Feature | Original BorutaShap | BorutaShapPySpark |
|---------|-------------------|-------------------|
| Dataset Size | Limited by memory | Unlimited (distributed) |
| Processing | Single-threaded | Distributed |
| Memory Usage | High | Low (chunked) |
| Speed | Fast for small data | Fast for large data |
| Setup | Simple | Requires Spark |
| SHAP Support | Full | Limited (sampling) |

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this implementation in your research, please cite:

```bibtex
@software{borutashap_pyspark,
  title={BorutaShapPySpark: Large-Scale Feature Selection with PySpark},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/Boruta-Shap}
}
```
