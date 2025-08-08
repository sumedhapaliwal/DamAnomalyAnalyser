# Dam Discharge Data Analysis and Anomaly Detection

**Author:** Sumedha Singh Paliwal  
**Date:** August 2025

## Project Overview

This project implements advanced **anomaly detection for dam discharge monitoring** using both traditional statistical methods and cutting-edge deep learning approaches. The system identifies unusual water discharge patterns that could indicate dam safety issues, flooding risks, or operational anomalies.

### Key Applications
- **Dam Safety Monitoring**: Early detection of structural or operational issues
- **Flood Risk Assessment**: Identifying abnormal discharge patterns
- **Water Resource Management**: Optimizing release patterns for downstream communities
- **Environmental Protection**: Monitoring ecological impacts of dam operations

## Methodology

### 1. Statistical Anomaly Detection (Traditional)
- **Moving Average Smoothing**: 144-point rolling window (6-day baseline)
- **Monthly Dynamic Bounds**: Adaptive thresholds using MSE + scaled standard deviation
- **Formula**: `bounds = MSE + (scale × STD_DEV)`
- **Advantages**: Simple, interpretable, computationally efficient

### 2. Deep Learning Anomaly Detection (Advanced)
- **LSTM Autoencoder**: Neural network learning normal discharge patterns
- **Isolation Forest**: Ensemble-based anomaly detection on reconstruction residuals
- **Feature Engineering**: Incorporates exponential smoothing and temporal dependencies
- **Superior Performance**: Detects subtle anomalies with reduced false positives

## Technical Stack

### Core Libraries
```python
pandas              # Data manipulation and time series analysis
numpy               # Numerical computations
matplotlib          # Visualization and plotting
scikit-learn        # Machine learning and preprocessing
```

### Statistical Analysis
```python
statsmodels         # Exponential smoothing (Holt-Winters)
scipy               # Statistical computations
```

### Deep Learning
```python
torch               # PyTorch neural networks
torch.nn            # LSTM autoencoder architecture
torch.optim         # Adam optimizer
```

### Anomaly Detection
```python
IsolationForest     # Ensemble-based anomaly detection
MinMaxScaler        # Feature normalization
```

## Key Results

### Performance Comparison (2024 Analysis)
- **Statistical Method**: 635 anomalies detected
- **Model-Based Method**: 847 anomalies detected
- **Critical Insight**: Model captures **all anomalies** in high-risk periods (Jan-March 2024)

### Why Deep Learning Wins
1. **Pattern Recognition**: 48-hour temporal context vs single-point statistics
2. **Adaptive Learning**: Learns complex seasonal and operational patterns
3. **Complete Coverage**: No missed anomalies during critical periods
4. **Reduced False Positives**: Better discrimination between natural variations and true anomalies

## Getting Started

### Prerequisites
```bash
Python 3.8+
CUDA-compatible GPU (optional, for faster training)
```

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd Dam

# Install dependencies
pip install -r requirements.txt
```

### Quick Start
```python
# Run the main analysis notebook
jupyter notebook src/main.ipynb
```

## Project Structure

```
DamAnomalyAnalyser/
├── src/
│   ├── main.ipynb           # Main analysis notebook
│   └── models/
│   │   └── 2feat_50epo.pth  # Pre-trained LSTM model
│   ├── data/
│       └── dam_data.csv     # Dam discharge dataset
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Model Architecture

### LSTM Autoencoder
- **Input**: 48-hour sequences (2 features: discharge, smoothed discharge)
- **Encoder**: 64-unit LSTM → 32-dimensional latent space
- **Decoder**: Reconstructs original temporal patterns
- **Training**: 50 epochs, Adam optimizer, MSE loss

### Anomaly Detection Pipeline
1. **Data Preprocessing**: MinMax scaling, sequence creation
2. **Model Training**: LSTM learns normal patterns
3. **Reconstruction**: Generate predictions for all sequences
4. **Residual Analysis**: Calculate reconstruction errors
5. **Isolation Forest**: Detect anomalies in residual space

## Visualization Features

- **Seasonal Shading**: Color-coded background for different seasons
- **Multiple Anomaly Types**: Statistical (yellow) vs Model-based (orange)
- **Confidence Bounds**: Dynamic threshold visualization
- **Forecast Overlay**: Model predictions vs actual values
- **Interactive Analysis**: Customizable date ranges and parameters

## Future Enhancements

### 1. Rainfall-Aware Detection
Integrate rainfall data to distinguish between natural high discharge (due to rain) and true anomalies.

### 2. N-BEATS Architecture
Implement state-of-the-art neural forecasting for even better pattern recognition.

### 3. Real-Time Monitoring
- Streaming data pipeline
- Edge computing for instant alerts
- SCADA system integration
- Mobile notifications

### 4. Enhanced Ensemble
Combine multiple approaches for robust, confidence-scored anomaly detection.

## Key Findings

1. **Model Superiority**: Deep learning detects 25% more anomalies than statistical methods
2. **Critical Period Coverage**: Perfect detection during high-risk operational periods
3. **Pattern Intelligence**: Understands seasonal variations and operational contexts
4. **Practical Value**: Enables proactive maintenance and safety monitoring

---