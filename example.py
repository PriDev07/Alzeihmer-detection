"""
Quick example: Train a simple VQC model with limited data
"""

from data_loader import ADNIDataLoader
from vqc_model import MultiClassVQC
from evaluation import VQCEvaluator

def quick_example():
    """Train a quick VQC model with limited subjects"""
    
    print("="*60)
    print("Quick Example: ADNI VQC Training")
    print("="*60)
    
    # Paths
    adni_path = "ADNI"
    diagnosis_path = "arc table/Diagnosis/DXSUM_28Jan2026.csv"
    mri_key_path = "arc table/Tables_28Jan2026/All_Subjects_Key_MRI_28Jan2026.csv"
    
    # Initialize loader
    print("\n1. Loading data...")
    loader = ADNIDataLoader(adni_path, diagnosis_path, mri_key_path)
    
    # Load limited dataset
    images, labels, subject_ids = loader.load_dataset(
        scan_type='MP-RAGE', 
        max_subjects=10  # Use only 10 subjects for quick demo
    )
    
    if len(images) == 0:
        print("No images loaded. Please check your data paths.")
        return
    
    # Prepare features
    features = loader.prepare_for_vqc(images, n_features=8)
    
    # Split data
    X_train, X_test, y_train, y_test = loader.split_data(features, labels)
    
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Initialize VQC
    print("\n2. Initializing VQC model...")
    vqc = MultiClassVQC(
        n_features=8,
        reps=1,  # Use 1 rep for faster training
        optimizer='COBYLA',
        max_iter=50,  # Limited iterations for demo
        random_seed=42
    )
    
    # Train
    print("\n3. Training model...")
    vqc.fit(X_train, y_train)
    
    # Evaluate
    print("\n4. Evaluating model...")
    y_pred = vqc.predict(X_test)
    accuracy = vqc.score(X_test, y_test)
    
    print(f"\nTest Accuracy: {accuracy:.4f}")
    
    # Detailed metrics
    evaluator = VQCEvaluator(class_names=['Normal', 'MCI', 'AD'])
    metrics = evaluator.calculate_metrics(y_test, y_pred)
    evaluator.print_metrics(metrics)
    
    print("\n" + "="*60)
    print("Example completed!")
    print("="*60)


if __name__ == "__main__":
    quick_example()
