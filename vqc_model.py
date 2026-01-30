"""
Variational Quantum Classifier (VQC) for Alzheimer's Detection
Implements quantum circuits for binary and multi-class classification
"""

import numpy as np
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt

# Qiskit imports
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_algorithms.optimizers import COBYLA, SPSA, L_BFGS_B
from qiskit_algorithms.utils import algorithm_globals
from qiskit.primitives import Sampler

from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.circuit.library import RawFeatureVector

import warnings
warnings.filterwarnings('ignore')


class ADNIQuantumClassifier:
    """Variational Quantum Classifier for Alzheimer's Disease classification"""
    
    def __init__(self, 
                 n_features: int = 8,
                 n_classes: int = 3,
                 reps: int = 2,
                 optimizer: str = 'COBYLA',
                 max_iter: int = 100,
                 random_seed: int = 42):
        """
        Initialize VQC model
        
        Args:
            n_features: Number of input features (must be power of 2 for quantum encoding)
            n_classes: Number of output classes (2 for binary, 3 for multi-class)
            reps: Number of repetitions in ansatz
            optimizer: Optimizer name ('COBYLA', 'SPSA', 'L_BFGS_B')
            max_iter: Maximum iterations for optimization
            random_seed: Random seed for reproducibility
        """
        self.n_features = n_features
        self.n_classes = n_classes
        self.reps = reps
        self.max_iter = max_iter
        self.random_seed = random_seed
        
        # Set random seed
        algorithm_globals.random_seed = random_seed
        
        # Calculate number of qubits (must accommodate features)
        self.n_qubits = int(np.ceil(np.log2(n_features)))
        if 2**self.n_qubits < n_features:
            self.n_qubits += 1
        
        # Initialize optimizer
        self.optimizer = self._get_optimizer(optimizer)
        
        # Build quantum circuit
        self.feature_map = self._build_feature_map()
        self.ansatz = self._build_ansatz()
        
        # VQC model (will be initialized during training)
        self.vqc = None
        self.training_history = []
        
    def _get_optimizer(self, optimizer_name: str):
        """Get optimizer instance"""
        if optimizer_name == 'COBYLA':
            return COBYLA(maxiter=self.max_iter)
        elif optimizer_name == 'SPSA':
            return SPSA(maxiter=self.max_iter)
        elif optimizer_name == 'L_BFGS_B':
            return L_BFGS_B(maxiter=self.max_iter)
        else:
            return COBYLA(maxiter=self.max_iter)
    
    def _build_feature_map(self) -> QuantumCircuit:
        """
        Build feature map for encoding classical data into quantum states
        Uses ZZFeatureMap with entanglement
        """
        feature_map = ZZFeatureMap(
            feature_dimension=self.n_qubits,
            reps=1,
            entanglement='linear'
        )
        return feature_map
    
    def _build_ansatz(self) -> QuantumCircuit:
        """
        Build variational ansatz (parameterized quantum circuit)
        Uses RealAmplitudes ansatz with linear entanglement
        """
        ansatz = RealAmplitudes(
            num_qubits=self.n_qubits,
            reps=self.reps,
            entanglement='linear'
        )
        return ansatz
    
    def build_full_circuit(self) -> QuantumCircuit:
        """Build complete quantum circuit (feature map + ansatz)"""
        circuit = QuantumCircuit(self.n_qubits)
        circuit.compose(self.feature_map, inplace=True)
        circuit.compose(self.ansatz, inplace=True)
        return circuit
    
    def visualize_circuit(self, save_path: Optional[str] = None):
        """Visualize the quantum circuit"""
        circuit = self.build_full_circuit()
        fig = circuit.decompose().draw(output='mpl', style='iqp', fold=-1)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Circuit saved to {save_path}")
        else:
            plt.show()
        
        return fig
    
    def pad_features(self, X: np.ndarray) -> np.ndarray:
        """
        Pad features to match number of qubits
        
        Args:
            X: Feature array (n_samples, n_features)
            
        Returns:
            Padded features (n_samples, n_qubits)
        """
        n_samples = X.shape[0]
        if X.shape[1] < self.n_qubits:
            padding = np.zeros((n_samples, self.n_qubits - X.shape[1]))
            X_padded = np.hstack([X, padding])
        else:
            X_padded = X[:, :self.n_qubits]
        
        return X_padded
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            callback: Optional[callable] = None) -> 'ADNIQuantumClassifier':
        """
        Train the VQC model
        
        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training labels (n_samples,)
            callback: Optional callback function for monitoring training
            
        Returns:
            self
        """
        # Pad features
        X_train_padded = self.pad_features(X_train)
        
        print(f"\n{'='*60}")
        print(f"Training Variational Quantum Classifier")
        print(f"{'='*60}")
        print(f"Number of qubits: {self.n_qubits}")
        print(f"Number of features: {self.n_features} (padded to {self.n_qubits})")
        print(f"Number of classes: {self.n_classes}")
        print(f"Training samples: {len(X_train)}")
        print(f"Ansatz reps: {self.reps}")
        print(f"Optimizer: {self.optimizer.__class__.__name__}")
        print(f"Max iterations: {self.max_iter}")
        print(f"{'='*60}\n")
        
        # Create VQC instance
        sampler = Sampler()
        
        self.vqc = VQC(
            sampler=sampler,
            feature_map=self.feature_map,
            ansatz=self.ansatz,
            optimizer=self.optimizer,
            callback=callback or self._default_callback
        )
        
        # Train model
        print("Starting training...\n")
        self.vqc.fit(X_train_padded, y_train)
        
        print(f"\n{'='*60}")
        print("Training completed!")
        print(f"{'='*60}\n")
        
        return self
    
    def _default_callback(self, eval_count, parameters, cost, stepsize):
        """Default callback for monitoring training"""
        self.training_history.append(cost)
        if eval_count % 10 == 0:
            print(f"Iteration {eval_count}: Cost = {cost:.4f}")
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict class labels
        
        Args:
            X_test: Test features (n_samples, n_features)
            
        Returns:
            Predicted labels (n_samples,)
        """
        if self.vqc is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        X_test_padded = self.pad_features(X_test)
        return self.vqc.predict(X_test_padded)
    
    def score(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """
        Calculate accuracy score
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Accuracy score
        """
        if self.vqc is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        X_test_padded = self.pad_features(X_test)
        return self.vqc.score(X_test_padded, y_test)
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training loss curve"""
        if len(self.training_history) == 0:
            print("No training history available")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.training_history, linewidth=2)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Cost Function', fontsize=12)
        plt.title('VQC Training Progress', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        else:
            plt.show()


class BinaryVQC(ADNIQuantumClassifier):
    """Binary VQC for Normal vs AD classification"""
    
    def __init__(self, **kwargs):
        kwargs['n_classes'] = 2
        super().__init__(**kwargs)


class MultiClassVQC(ADNIQuantumClassifier):
    """Multi-class VQC for Normal vs MCI vs AD classification"""
    
    def __init__(self, **kwargs):
        kwargs['n_classes'] = 3
        super().__init__(**kwargs)


def demo_circuit():
    """Demo: Visualize quantum circuit"""
    print("Creating demo VQC circuit...")
    vqc = ADNIQuantumClassifier(n_features=8, n_classes=3, reps=2)
    
    print(f"Number of qubits: {vqc.n_qubits}")
    print(f"Feature map: {vqc.feature_map}")
    print(f"Ansatz: {vqc.ansatz}")
    
    # Visualize circuit
    vqc.visualize_circuit(save_path="vqc_circuit.png")


if __name__ == "__main__":
    demo_circuit()
