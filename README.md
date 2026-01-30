# ADNI Variational Quantum Classifier for Alzheimer's Detection

A quantum machine learning approach for detecting Alzheimer's disease using ADNI (Alzheimer's Disease Neuroimaging Initiative) MRI data with Variational Quantum Circuits (VQC).

## Overview

This project implements a Variational Quantum Classifier (VQC) to classify brain MRI scans into three categories:
- **Normal**: Cognitively normal subjects
- **MCI**: Mild Cognitive Impairment
- **AD**: Alzheimer's Disease

The system leverages quantum computing principles through Qiskit to perform classification on classical neuroimaging data.

## Features

- **DICOM Data Processing**: Automated loading and preprocessing of ADNI MRI DICOM files
- **Quantum Circuit Design**: Custom VQC architecture with ZZFeatureMap and RealAmplitudes ansatz
- **Multi-class Classification**: Support for both binary (Normal vs AD) and 3-class (Normal vs MCI vs AD) classification
- **Comprehensive Evaluation**: Detailed metrics, confusion matrices, and visualization
- **Experiment Tracking**: Log and compare multiple training runs

## Project Structure

```
alzheimer-detection-1/
├── ADNI/                          # ADNI MRI data (DICOM files)
│   ├── 002_S_0413/
│   ├── 002_S_0559/
│   └── ...
├── arc table/                     # Clinical metadata
│   ├── Diagnosis/
│   │   └── DXSUM_28Jan2026.csv   # Diagnosis labels
│   └── Tables_28Jan2026/
│       └── All_Subjects_Key_MRI_28Jan2026.csv
├── data_loader.py                 # Data loading and preprocessing
├── vqc_model.py                   # VQC model implementation
├── evaluation.py                  # Metrics and visualization
├── train.py                       # Main training script
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository or navigate to the project directory

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Dependencies
- **Quantum Computing**: qiskit, qiskit-machine-learning
- **Data Processing**: numpy, pandas, pydicom, nibabel, opencv-python
- **Machine Learning**: scikit-learn, torch, torchvision
- **Visualization**: matplotlib, seaborn
- **Utilities**: tqdm, joblib, scipy

## Usage

### Quick Start

Train a 3-class VQC model with default parameters:

```bash
python train.py
```

### Custom Training

Train with specific parameters:

```bash
python train.py \
  --n-features 8 \
  --n-classes 3 \
  --reps 2 \
  --optimizer COBYLA \
  --max-iter 100 \
  --max-subjects 20 \
  --output-dir results
```

### Command Line Arguments

**Data Arguments:**
- `--adni-path`: Path to ADNI data folder (default: `ADNI`)
- `--diagnosis-path`: Path to diagnosis CSV
- `--mri-key-path`: Path to MRI key CSV
- `--scan-type`: MRI scan type (default: `MP-RAGE`)
- `--max-subjects`: Max subjects to load (default: None = all)

**Model Arguments:**
- `--n-features`: Number of features for VQC (default: 8)
- `--n-classes`: Number of classes - 2 or 3 (default: 3)
- `--reps`: Ansatz repetitions (default: 2)
- `--optimizer`: Optimizer - COBYLA, SPSA, or L_BFGS_B (default: COBYLA)
- `--max-iter`: Maximum optimization iterations (default: 100)

**Training Arguments:**
- `--test-size`: Test set proportion (default: 0.2)
- `--random-seed`: Random seed (default: 42)
- `--output-dir`: Results directory (default: `results`)

## Examples

### Binary Classification (Normal vs AD)

```bash
python train.py --n-classes 2 --max-subjects 50
```

### Multi-class with More Iterations

```bash
python train.py --n-classes 3 --max-iter 200 --optimizer SPSA
```

### Limited Dataset Testing

```bash
python train.py --max-subjects 10 --max-iter 50
```

## Model Architecture

### Quantum Circuit Components

1. **Feature Map**: ZZFeatureMap
   - Encodes classical features into quantum states
   - Linear entanglement between qubits
   - Number of qubits: ceil(log2(n_features))

2. **Ansatz**: RealAmplitudes
   - Variational component with trainable parameters
   - Linear entanglement pattern
   - Configurable repetitions (reps)

3. **Measurement**: Sampling-based measurement
   - Uses Qiskit Sampler primitive
   - Produces probability distributions for classification

### Data Processing Pipeline

1. **DICOM Loading**: Load MRI slices from DICOM files
2. **Volume Aggregation**: Max intensity projection to 2D
3. **Feature Extraction**: Statistical features (mean, std, median, percentiles)
4. **Normalization**: StandardScaler normalization
5. **Quantum Encoding**: Feature padding to match qubit requirements

## Output

After training, the following files are generated in the output directory:

- `vqc_circuit_*.png`: Quantum circuit visualization
- `training_history_*.png`: Loss curve during training
- `VQC_test_*_confusion_matrix.png`: Confusion matrix
- `VQC_test_*_classification_report.png`: Per-class metrics heatmap
- `VQC_test_*_metrics_*.json`: Detailed metrics in JSON
- `experiments_*.json`: Experiment log with hyperparameters

## Evaluation Metrics

The system reports:
- **Accuracy**: Overall classification accuracy
- **Precision, Recall, F1-score**: Both macro and weighted averages
- **Per-class Metrics**: Individual performance for each class
- **Confusion Matrix**: Visual representation of predictions

## Data Requirements

### ADNI Dataset Structure
```
ADNI/
└── {SUBJECT_ID}/
    └── MP-RAGE/
        └── {TIMESTAMP}/
            └── {IMAGE_ID}/
                └── *.dcm (DICOM files)
```

### Clinical Data
- `DXSUM_*.csv`: Contains diagnosis labels (DIAGNOSIS column: 1=Normal, 2=MCI, 3=AD)
- `All_Subjects_Key_MRI_*.csv`: Contains MRI metadata and subject information

## Performance Considerations

- **Quantum Simulation**: Training runs on classical simulation of quantum circuits
- **Computation Time**: Depends on:
  - Number of samples
  - Number of qubits (determined by features)
  - Optimizer iterations
  - Ansatz complexity (reps)
  
- **Typical Runtime**: 
  - 10 subjects, 50 iterations: ~5-10 minutes
  - 50 subjects, 100 iterations: ~20-30 minutes

## Limitations

1. **Classical Simulation**: Currently runs on simulated quantum hardware
2. **Feature Dimensionality**: Limited by number of qubits (powers of 2)
3. **Dataset Size**: Small datasets may lead to overfitting
4. **Computational Cost**: Quantum simulation is resource-intensive

## Future Enhancements

- [ ] Support for real quantum hardware (IBM Quantum)
- [ ] Advanced feature extraction (CNN-based features)
- [ ] Hybrid quantum-classical models
- [ ] Cross-validation implementation
- [ ] Hyperparameter optimization
- [ ] 3D volume processing instead of 2D projections
- [ ] Data augmentation techniques
- [ ] Ensemble methods

## Troubleshooting

### Common Issues

**Issue**: "No images loaded"
- Check ADNI data path and folder structure
- Verify DICOM files exist in MP-RAGE folders
- Ensure diagnosis CSV has matching subject IDs

**Issue**: "Feature dimension mismatch"
- Adjust `--n-features` to a value supported by your data
- Features are automatically padded to next power of 2

**Issue**: "Training too slow"
- Reduce `--max-subjects` for faster testing
- Decrease `--max-iter` 
- Use simpler optimizer (COBYLA faster than SPSA)
- Reduce `--reps` for smaller ansatz

## References

- [ADNI Database](http://adni.loni.usc.edu/)
- [Qiskit Machine Learning](https://qiskit.org/ecosystem/machine-learning/)
- [Variational Quantum Classifier](https://qiskit.org/documentation/machine-learning/tutorials/02_neural_network_classifier_and_regressor.html)

## License

This project is for research and educational purposes. Please refer to ADNI data usage agreements for the neuroimaging data.

## Citation

If you use this code, please cite:
```
ADNI Variational Quantum Classifier for Alzheimer's Detection
[Your Name/Institution]
2026
```

## Contact

For questions or issues, please open an issue in the repository.

---

**Note**: This is a research project. The model should not be used for clinical diagnosis without proper validation and regulatory approval.
