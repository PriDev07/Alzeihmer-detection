"""
Main Training Script for ADNI VQC Alzheimer's Detection
End-to-end pipeline: Data loading -> Training -> Evaluation
"""

import numpy as np
import argparse
import os
from datetime import datetime

from data_loader import ADNIDataLoader
from vqc_model import ADNIQuantumClassifier, BinaryVQC, MultiClassVQC
from evaluation import VQCEvaluator, ExperimentTracker


def train_vqc_alzheimer_detection(
    adni_path: str = 'ADNI',
    diagnosis_path: str = 'arc table/Diagnosis/DXSUM_28Jan2026.csv',
    mri_key_path: str = 'arc table/Tables_28Jan2026/All_Subjects_Key_MRI_28Jan2026.csv',
    n_features: int = 8,
    n_classes: int = 3,
    scan_type: str = 'MP-RAGE',
    max_subjects: int = None,
    test_size: float = 0.2,
    reps: int = 2,
    optimizer: str = 'COBYLA',
    max_iter: int = 100,
    output_dir: str = 'results',
    random_seed: int = 42
):
    """
    Complete training pipeline for VQC Alzheimer's detection
    
    Args:
        adni_path: Path to ADNI data folder
        diagnosis_path: Path to diagnosis CSV
        mri_key_path: Path to MRI key CSV
        n_features: Number of features for VQC
        n_classes: Number of classes (2 for binary, 3 for multi-class)
        scan_type: MRI scan type to use
        max_subjects: Maximum subjects to load (None for all)
        test_size: Test set proportion
        reps: Number of ansatz repetitions
        optimizer: Optimizer name
        max_iter: Maximum optimization iterations
        output_dir: Directory for saving results
        random_seed: Random seed
    """
    print("\n" + "="*80)
    print("ADNI VARIATIONAL QUANTUM CLASSIFIER - ALZHEIMER'S DETECTION")
    print("="*80)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # =========================================================================
    # STEP 1: Load and Preprocess Data
    # =========================================================================
    print("\n[STEP 1/4] Loading and preprocessing ADNI data...")
    print("-" * 80)
    
    loader = ADNIDataLoader(
        adni_path=adni_path,
        diagnosis_path=diagnosis_path,
        mri_key_path=mri_key_path,
        target_size=(128, 128),
        max_slices=10
    )
    
    # Load dataset
    images, labels, subject_ids = loader.load_dataset(
        scan_type=scan_type,
        max_subjects=max_subjects
    )
    
    if len(images) == 0:
        print("Error: No images loaded. Please check your data paths.")
        return
    
    # Prepare features for VQC
    features = loader.prepare_for_vqc(images, n_features=n_features)
    
    # Split data
    X_train, X_test, y_train, y_test = loader.split_data(
        features, labels, test_size=test_size, random_state=random_seed
    )
    
    print(f"\nDataset Summary:")
    print(f"  Total samples: {len(features)}")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Number of features: {n_features}")
    print(f"  Number of classes: {n_classes}")
    print(f"  Class distribution (train): {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"  Class distribution (test): {dict(zip(*np.unique(y_test, return_counts=True)))}")
    
    # =========================================================================
    # STEP 2: Initialize VQC Model
    # =========================================================================
    print("\n[STEP 2/4] Initializing Variational Quantum Classifier...")
    print("-" * 80)
    
    if n_classes == 2:
        vqc = BinaryVQC(
            n_features=n_features,
            reps=reps,
            optimizer=optimizer,
            max_iter=max_iter,
            random_seed=random_seed
        )
        class_names = ['Normal', 'AD']
    else:
        vqc = MultiClassVQC(
            n_features=n_features,
            reps=reps,
            optimizer=optimizer,
            max_iter=max_iter,
            random_seed=random_seed
        )
        class_names = ['Normal', 'MCI', 'AD']
    
    # Visualize circuit
    circuit_path = os.path.join(output_dir, f'vqc_circuit_{timestamp}.png')
    try:
        vqc.visualize_circuit(save_path=circuit_path)
    except Exception as e:
        print(f"Warning: Could not visualize circuit: {e}")
    
    # =========================================================================
    # STEP 3: Train Model
    # =========================================================================
    print("\n[STEP 3/4] Training VQC model...")
    print("-" * 80)
    
    vqc.fit(X_train, y_train)
    
    # Plot training history
    training_plot_path = os.path.join(output_dir, f'training_history_{timestamp}.png')
    try:
        vqc.plot_training_history(save_path=training_plot_path)
    except Exception as e:
        print(f"Warning: Could not plot training history: {e}")
    
    # =========================================================================
    # STEP 4: Evaluate Model
    # =========================================================================
    print("\n[STEP 4/4] Evaluating model performance...")
    print("-" * 80)
    
    # Make predictions
    y_pred_train = vqc.predict(X_train)
    y_pred_test = vqc.predict(X_test)
    
    # Calculate scores
    train_accuracy = vqc.score(X_train, y_train)
    test_accuracy = vqc.score(X_test, y_test)
    
    print(f"\nAccuracy Scores:")
    print(f"  Training accuracy: {train_accuracy:.4f}")
    print(f"  Test accuracy: {test_accuracy:.4f}")
    
    # Evaluate with comprehensive metrics
    evaluator = VQCEvaluator(class_names=class_names)
    
    # Generate reports
    print("\nGenerating evaluation reports...")
    
    # Test set evaluation
    test_metrics = evaluator.generate_report(
        y_true=y_test,
        y_pred=y_pred_test,
        output_dir=output_dir,
        model_name=f'VQC_test_{timestamp}',
        metadata={
            'n_features': n_features,
            'n_classes': n_classes,
            'reps': reps,
            'optimizer': optimizer,
            'max_iter': max_iter,
            'scan_type': scan_type,
            'train_samples': len(X_train),
            'test_samples': len(X_test)
        }
    )
    
    # Training set evaluation (for comparison)
    train_metrics = evaluator.calculate_metrics(y_train, y_pred_train)
    
    # =========================================================================
    # STEP 5: Log Experiment
    # =========================================================================
    print("\n[STEP 5/5] Logging experiment...")
    print("-" * 80)
    
    tracker = ExperimentTracker(experiment_name='ADNI_VQC_Alzheimers')
    
    tracker.log_experiment(
        model_name=f'VQC_{n_classes}class_{timestamp}',
        metrics=test_metrics,
        hyperparameters={
            'n_features': n_features,
            'n_classes': n_classes,
            'n_qubits': vqc.n_qubits,
            'reps': reps,
            'optimizer': optimizer,
            'max_iter': max_iter,
            'scan_type': scan_type,
            'test_size': test_size,
            'random_seed': random_seed
        },
        notes=f'Train acc: {train_accuracy:.4f}, Test acc: {test_accuracy:.4f}'
    )
    
    experiments_path = os.path.join(output_dir, f'experiments_{timestamp}.json')
    tracker.save_experiments(experiments_path)
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*80)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\nResults saved in: {output_dir}")
    print(f"\nKey Files:")
    print(f"  - Quantum circuit: {circuit_path}")
    print(f"  - Training history: {training_plot_path}")
    print(f"  - Experiments log: {experiments_path}")
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print("="*80 + "\n")
    
    return vqc, test_metrics


def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(
        description='Train VQC for ADNI Alzheimer\'s Detection'
    )
    
    # Data arguments
    parser.add_argument('--adni-path', type=str, default='ADNI',
                       help='Path to ADNI data folder')
    parser.add_argument('--diagnosis-path', type=str, 
                       default='arc table/Diagnosis/DXSUM_28Jan2026.csv',
                       help='Path to diagnosis CSV')
    parser.add_argument('--mri-key-path', type=str,
                       default='arc table/Tables_28Jan2026/All_Subjects_Key_MRI_28Jan2026.csv',
                       help='Path to MRI key CSV')
    parser.add_argument('--scan-type', type=str, default='MP-RAGE',
                       help='MRI scan type to use')
    parser.add_argument('--max-subjects', type=int, default=None,
                       help='Maximum number of subjects to load')
    
    # Model arguments
    parser.add_argument('--n-features', type=int, default=8,
                       help='Number of features for VQC input')
    parser.add_argument('--n-classes', type=int, default=3,
                       help='Number of classes (2 or 3)')
    parser.add_argument('--reps', type=int, default=2,
                       help='Number of ansatz repetitions')
    parser.add_argument('--optimizer', type=str, default='COBYLA',
                       choices=['COBYLA', 'SPSA', 'L_BFGS_B'],
                       help='Optimizer to use')
    parser.add_argument('--max-iter', type=int, default=100,
                       help='Maximum optimization iterations')
    
    # Training arguments
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set proportion')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Run training
    train_vqc_alzheimer_detection(
        adni_path=args.adni_path,
        diagnosis_path=args.diagnosis_path,
        mri_key_path=args.mri_key_path,
        n_features=args.n_features,
        n_classes=args.n_classes,
        scan_type=args.scan_type,
        max_subjects=args.max_subjects,
        test_size=args.test_size,
        reps=args.reps,
        optimizer=args.optimizer,
        max_iter=args.max_iter,
        output_dir=args.output_dir,
        random_seed=args.random_seed
    )


if __name__ == "__main__":
    main()
