"""
ADNI Data Loader for Alzheimer's Detection using VQC
Handles DICOM image loading and preprocessing
"""

import os
import numpy as np
import pandas as pd
import pydicom
from pathlib import Path
from typing import List, Tuple, Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import cv2
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class ADNIDataLoader:
    """Load and preprocess ADNI DICOM images with diagnosis labels"""
    
    def __init__(self, 
                 adni_path: str,
                 diagnosis_path: str,
                 mri_key_path: str,
                 target_size: Tuple[int, int] = (128, 128),
                 max_slices: int = 10):
        """
        Initialize ADNI data loader
        
        Args:
            adni_path: Path to ADNI folder with DICOM images
            diagnosis_path: Path to DXSUM CSV file
            mri_key_path: Path to MRI key CSV file
            target_size: Target image dimensions for resizing
            max_slices: Maximum number of slices to use per scan
        """
        self.adni_path = Path(adni_path)
        self.diagnosis_path = diagnosis_path
        self.mri_key_path = mri_key_path
        self.target_size = target_size
        self.max_slices = max_slices
        
        # Load metadata
        self.diagnosis_df = pd.read_csv(diagnosis_path)
        self.mri_key_df = pd.read_csv(mri_key_path)
        
        # Diagnosis mapping: 1=Normal, 2=MCI, 3=AD
        self.diagnosis_map = {1: 0, 2: 1, 3: 2}  # 0=Normal, 1=MCI, 2=AD
        self.class_names = ['Normal', 'MCI', 'AD']
        
    def extract_subject_id(self, folder_name: str) -> str:
        """Extract subject ID from folder name (e.g., '002_S_0413')"""
        return folder_name
    
    def get_diagnosis_for_subject(self, subject_id: str, visit_code: str = 'bl') -> int:
        """
        Get diagnosis label for a subject
        
        Args:
            subject_id: Subject ID (e.g., '002_S_0413')
            visit_code: Visit code (default 'bl' for baseline)
        
        Returns:
            Diagnosis code: 0=Normal, 1=MCI, 2=AD, -1=Unknown
        """
        subject_data = self.diagnosis_df[
            (self.diagnosis_df['PTID'] == subject_id) & 
            (self.diagnosis_df['VISCODE'] == visit_code)
        ]
        
        if len(subject_data) == 0:
            return -1
        
        diagnosis = subject_data.iloc[0]['DIAGNOSIS']
        return self.diagnosis_map.get(diagnosis, -1)
    
    def load_dicom_slice(self, dicom_path: str) -> np.ndarray:
        """
        Load and preprocess a single DICOM slice
        
        Args:
            dicom_path: Path to DICOM file
            
        Returns:
            Preprocessed image array
        """
        try:
            dicom_data = pydicom.dcmread(dicom_path)
            image = dicom_data.pixel_array.astype(float)
            
            # Normalize to 0-1
            image = (image - image.min()) / (image.max() - image.min() + 1e-8)
            
            # Resize
            image = cv2.resize(image, self.target_size)
            
            return image
        except Exception as e:
            print(f"Error loading {dicom_path}: {e}")
            return None
    
    def load_scan_volume(self, scan_path: Path) -> np.ndarray:
        """
        Load multiple DICOM slices from a scan folder
        
        Args:
            scan_path: Path to folder containing DICOM files
            
        Returns:
            3D numpy array of shape (slices, height, width)
        """
        dicom_files = sorted(list(scan_path.glob('*.dcm')))
        
        if len(dicom_files) == 0:
            return None
        
        # Sample slices evenly across the volume
        if len(dicom_files) > self.max_slices:
            indices = np.linspace(0, len(dicom_files)-1, self.max_slices, dtype=int)
            dicom_files = [dicom_files[i] for i in indices]
        
        slices = []
        for dicom_file in dicom_files:
            slice_img = self.load_dicom_slice(str(dicom_file))
            if slice_img is not None:
                slices.append(slice_img)
        
        if len(slices) == 0:
            return None
        
        # Pad if needed
        while len(slices) < self.max_slices:
            slices.append(np.zeros(self.target_size))
        
        return np.array(slices[:self.max_slices])
    
    def aggregate_volume_to_2d(self, volume: np.ndarray) -> np.ndarray:
        """
        Aggregate 3D volume to 2D by taking max intensity projection
        
        Args:
            volume: 3D array (slices, height, width)
            
        Returns:
            2D array (height, width)
        """
        return np.max(volume, axis=0)
    
    def load_dataset(self, 
                     scan_type: str = 'MP-RAGE',
                     max_subjects: int = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load complete dataset
        
        Args:
            scan_type: Type of MRI scan to load (e.g., 'MP-RAGE')
            max_subjects: Maximum number of subjects to load (None for all)
            
        Returns:
            Tuple of (images, labels, subject_ids)
        """
        images = []
        labels = []
        subject_ids = []
        
        # Iterate through subject folders
        subject_folders = sorted([f for f in self.adni_path.iterdir() if f.is_dir()])
        
        if max_subjects:
            subject_folders = subject_folders[:max_subjects]
        
        print(f"Loading {scan_type} scans from {len(subject_folders)} subjects...")
        
        for subject_folder in tqdm(subject_folders):
            subject_id = self.extract_subject_id(subject_folder.name)
            
            # Get diagnosis
            diagnosis = self.get_diagnosis_for_subject(subject_id)
            if diagnosis == -1:
                continue  # Skip subjects without diagnosis
            
            # Find scan folder
            scan_folder = subject_folder / scan_type
            if not scan_folder.exists():
                continue
            
            # Find first timestamped scan
            scan_sessions = sorted([f for f in scan_folder.iterdir() if f.is_dir()])
            if len(scan_sessions) == 0:
                continue
            
            # Use first session (baseline)
            first_session = scan_sessions[0]
            
            # Find image folder (usually one level deeper)
            image_folders = [f for f in first_session.iterdir() if f.is_dir()]
            if len(image_folders) == 0:
                continue
            
            image_folder = image_folders[0]
            
            # Load volume
            volume = self.load_scan_volume(image_folder)
            if volume is None:
                continue
            
            # Aggregate to 2D
            image_2d = self.aggregate_volume_to_2d(volume)
            
            images.append(image_2d)
            labels.append(diagnosis)
            subject_ids.append(subject_id)
        
        print(f"Loaded {len(images)} images")
        print(f"Class distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
        
        return np.array(images), np.array(labels), subject_ids
    
    def prepare_for_vqc(self, images: np.ndarray, n_features: int = 8) -> np.ndarray:
        """
        Prepare images for VQC by extracting features and reducing dimensionality
        
        Args:
            images: Array of images (n_samples, height, width)
            n_features: Number of features for VQC input
            
        Returns:
            Feature array (n_samples, n_features)
        """
        n_samples = len(images)
        
        # Extract statistical features from each image
        features = []
        for img in images:
            img_features = [
                np.mean(img),
                np.std(img),
                np.median(img),
                np.max(img),
                np.min(img),
                np.percentile(img, 25),
                np.percentile(img, 75),
                np.sum(img > 0.5) / img.size,  # Proportion of high-intensity pixels
            ]
            features.append(img_features[:n_features])
        
        features = np.array(features)
        
        # Normalize features
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        
        return features
    
    def split_data(self, 
                   X: np.ndarray, 
                   y: np.ndarray,
                   test_size: float = 0.2,
                   random_state: int = 42) -> Tuple:
        """
        Split data into train and test sets
        
        Args:
            X: Feature array
            y: Label array
            test_size: Proportion of test set
            random_state: Random seed
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        return train_test_split(X, y, test_size=test_size, 
                              random_state=random_state, stratify=y)


def main():
    """Demo usage"""
    # Paths
    adni_path = "ADNI"
    diagnosis_path = "arc table/Diagnosis/DXSUM_28Jan2026.csv"
    mri_key_path = "arc table/Tables_28Jan2026/All_Subjects_Key_MRI_28Jan2026.csv"
    
    # Initialize loader
    loader = ADNIDataLoader(adni_path, diagnosis_path, mri_key_path)
    
    # Load dataset
    images, labels, subject_ids = loader.load_dataset(scan_type='MP-RAGE', max_subjects=10)
    
    # Prepare for VQC
    features = loader.prepare_for_vqc(images, n_features=8)
    
    # Split data
    X_train, X_test, y_train, y_test = loader.split_data(features, labels)
    
    print(f"\nDataset prepared:")
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Feature range: [{features.min():.3f}, {features.max():.3f}]")


if __name__ == "__main__":
    main()
