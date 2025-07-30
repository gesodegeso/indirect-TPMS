# ===========================================
# docs/installation.md
# ===========================================

# Installation Guide

## System Requirements

- **Python**: 3.8 or higher
- **Operating System**: Windows, macOS, or Linux
- **Memory**: 4GB RAM minimum (8GB recommended for training)
- **Storage**: 1GB free space

## Quick Installation

### Option 1: pip install (Recommended)

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/indirect-TPMS.git
cd indirect-TPMS

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Option 2: Development Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/indirect-TPMS.git
cd indirect-TPMS

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install with development dependencies
pip install -e .[dev]
```

## Dependency Details

### Core Dependencies
- **numpy**: Numerical computing
- **pandas**: Data manipulation
- **scipy**: Scientific computing
- **scikit-learn**: Machine learning algorithms
- **tensorflow**: Deep learning framework
- **tensorflow-probability**: Bayesian neural networks
- **statsmodels**: Time series analysis

### Optional Dependencies
- **matplotlib/seaborn**: Visualization (install with `pip install -e .[viz]`)
- **numba**: JIT compilation for speed improvements

## Verification

Test the installation:

```bash
# Run basic demo
python -m src.main demo

# Run unit tests
pytest tests/

# Check imports
python -c "from src.indirect_tpms_module import TPMSDataProcessor; print('Installation successful!')"
```

## Troubleshooting

### Common Issues

**1. TensorFlow Installation Issues**
```bash
# For CPU-only version
pip install tensorflow-cpu

# For GPU support (requires CUDA)
pip install tensorflow-gpu
```

**2. Missing BLAS Libraries (Linux)**
```bash
sudo apt-get install libblas-dev liblapack-dev
```

**3. Permission Errors (Windows)**
- Run command prompt as administrator
- Or use `--user` flag: `pip install --user -e .`

**4. Virtual Environment Issues**
```bash
# Ensure virtual environment is activated
which python  # Should point to venv/bin/python

# Reinstall if needed
pip uninstall indirect-tpms
pip install -e .
```

### Performance Optimization

**1. Enable GPU Support (Optional)**
```bash
# Install CUDA and cuDNN (refer to TensorFlow docs)
pip install tensorflow-gpu
```

**2. Install Optimized BLAS**
```bash
# Intel MKL (faster on Intel CPUs)
pip install mkl

# OpenBLAS alternative
pip install openblas
```

## Hardware Requirements

### For Development/Testing
- **CPU**: Any modern processor
- **RAM**: 4GB minimum
- **Storage**: 1GB

### For Real Vehicle Integration
- **ECU**: ARM Cortex-A series or equivalent
- **RAM**: 1GB minimum
- **CAN Interface**: For vehicle sensor data
- **Real-time OS**: For deterministic processing

## Next Steps

After installation:
1. Read the [Usage Guide](usage.md)
2. Try the [Basic Example](../examples/basic_usage.py)
3. Explore the [API Reference](api_reference.md)