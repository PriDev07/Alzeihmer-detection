"""
Training and Evaluation Pipeline for ADNI VQC
Includes metrics, visualization, and model comparison
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from typing import Dict, List, Tuple, Optional
import json
import os
from datetime import datetime


class VQCEvaluator:
    """Evaluate and visualize VQC model performance"""
    
    def __init__(self, class_names: List[str] = ['Normal', 'MCI', 'AD']):
        """
        Initialize evaluator
        
        Args:
            class_names: List of class names
        """
        self.class_names = class_names
        self.n_classes = len(class_names)
        self.metrics_history = []
        
    def calculate_metrics(self, 
                         y_true: np.ndarray, 
                         y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }
        
        # Per-class metrics
        if self.n_classes > 2:
            precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
            recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
            f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
            
            for i, class_name in enumerate(self.class_names):
                if i < len(precision_per_class):
                    metrics[f'precision_{class_name}'] = precision_per_class[i]
                    metrics[f'recall_{class_name}'] = recall_per_class[i]
                    metrics[f'f1_{class_name}'] = f1_per_class[i]
        
        return metrics
    
    def print_metrics(self, metrics: Dict[str, float]):
        """Print metrics in a formatted way"""
        print("\n" + "="*60)
        print("MODEL EVALUATION METRICS")
        print("="*60)
        print(f"Accuracy:           {metrics['accuracy']:.4f}")
        print(f"Precision (macro):  {metrics['precision_macro']:.4f}")
        print(f"Recall (macro):     {metrics['recall_macro']:.4f}")
        print(f"F1-score (macro):   {metrics['f1_macro']:.4f}")
        print("-"*60)
        print(f"Precision (weighted): {metrics['precision_weighted']:.4f}")
        print(f"Recall (weighted):    {metrics['recall_weighted']:.4f}")
        print(f"F1-score (weighted):  {metrics['f1_weighted']:.4f}")
        
        if self.n_classes > 2:
            print("-"*60)
            print("Per-Class Metrics:")
            for class_name in self.class_names:
                if f'f1_{class_name}' in metrics:
                    print(f"\n{class_name}:")
                    print(f"  Precision: {metrics[f'precision_{class_name}']:.4f}")
                    print(f"  Recall:    {metrics[f'recall_{class_name}']:.4f}")
                    print(f"  F1-score:  {metrics[f'f1_{class_name}']:.4f}")
        
        print("="*60 + "\n")
    
    def plot_confusion_matrix(self, 
                            y_true: np.ndarray, 
                            y_pred: np.ndarray,
                            save_path: Optional[str] = None):
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save figure
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, 
                   yticklabels=self.class_names,
                   cbar_kws={'label': 'Count'})
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        plt.title('Confusion Matrix - VQC Alzheimer Detection', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        else:
            plt.show()
    
    def plot_classification_report(self,
                                  y_true: np.ndarray,
                                  y_pred: np.ndarray,
                                  save_path: Optional[str] = None):
        """
        Visualize classification report as heatmap
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save figure
        """
        report = classification_report(y_true, y_pred, 
                                      target_names=self.class_names,
                                      output_dict=True,
                                      zero_division=0)
        
        # Extract metrics for visualization
        metrics_data = []
        for class_name in self.class_names:
            if class_name in report:
                metrics_data.append([
                    report[class_name]['precision'],
                    report[class_name]['recall'],
                    report[class_name]['f1-score']
                ])
        
        metrics_df = pd.DataFrame(
            metrics_data,
            columns=['Precision', 'Recall', 'F1-Score'],
            index=self.class_names
        )
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(metrics_df, annot=True, fmt='.3f', cmap='RdYlGn',
                   vmin=0, vmax=1, cbar_kws={'label': 'Score'})
        plt.title('Classification Report - Per Class Metrics', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.ylabel('Class', fontsize=12, fontweight='bold')
        plt.xlabel('Metric', fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Classification report saved to {save_path}")
        else:
            plt.show()
    
    def plot_metrics_comparison(self,
                               metrics_list: List[Dict[str, float]],
                               model_names: List[str],
                               save_path: Optional[str] = None):
        """
        Compare metrics across different models
        
        Args:
            metrics_list: List of metric dictionaries
            model_names: List of model names
            save_path: Path to save figure
        """
        metric_keys = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.ravel()
        
        for idx, metric_key in enumerate(metric_keys):
            values = [metrics[metric_key] for metrics in metrics_list]
            
            axes[idx].bar(model_names, values, color='steelblue', alpha=0.7)
            axes[idx].set_ylabel('Score', fontsize=11)
            axes[idx].set_title(metric_key.replace('_', ' ').title(), 
                              fontsize=12, fontweight='bold')
            axes[idx].set_ylim([0, 1])
            axes[idx].grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for i, v in enumerate(values):
                axes[idx].text(i, v + 0.02, f'{v:.3f}', 
                             ha='center', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Metrics comparison saved to {save_path}")
        else:
            plt.show()
    
    def save_metrics(self, 
                    metrics: Dict[str, float], 
                    save_path: str,
                    metadata: Optional[Dict] = None):
        """
        Save metrics to JSON file
        
        Args:
            metrics: Dictionary of metrics
            save_path: Path to save JSON file
            metadata: Optional metadata to include
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
        }
        
        if metadata:
            results['metadata'] = metadata
        
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"Metrics saved to {save_path}")
    
    def generate_report(self,
                       y_true: np.ndarray,
                       y_pred: np.ndarray,
                       output_dir: str = 'results',
                       model_name: str = 'VQC',
                       metadata: Optional[Dict] = None):
        """
        Generate comprehensive evaluation report
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            output_dir: Directory to save results
            model_name: Name of the model
            metadata: Optional metadata
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_true, y_pred)
        
        # Print metrics
        self.print_metrics(metrics)
        
        # Save metrics
        metrics_path = os.path.join(output_dir, f'{model_name}_metrics_{timestamp}.json')
        self.save_metrics(metrics, metrics_path, metadata)
        
        # Plot confusion matrix
        cm_path = os.path.join(output_dir, f'{model_name}_confusion_matrix_{timestamp}.png')
        self.plot_confusion_matrix(y_true, y_pred, save_path=cm_path)
        
        # Plot classification report
        report_path = os.path.join(output_dir, f'{model_name}_classification_report_{timestamp}.png')
        self.plot_classification_report(y_true, y_pred, save_path=report_path)
        
        print(f"\nEvaluation report generated in: {output_dir}")
        
        return metrics


class ExperimentTracker:
    """Track multiple experiments and compare results"""
    
    def __init__(self, experiment_name: str = 'ADNI_VQC'):
        """
        Initialize experiment tracker
        
        Args:
            experiment_name: Name of the experiment
        """
        self.experiment_name = experiment_name
        self.experiments = []
        
    def log_experiment(self,
                      model_name: str,
                      metrics: Dict[str, float],
                      hyperparameters: Dict,
                      notes: str = ''):
        """
        Log an experiment
        
        Args:
            model_name: Name of the model
            metrics: Dictionary of metrics
            hyperparameters: Dictionary of hyperparameters
            notes: Optional notes
        """
        experiment = {
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'metrics': metrics,
            'hyperparameters': hyperparameters,
            'notes': notes
        }
        
        self.experiments.append(experiment)
        print(f"Logged experiment: {model_name}")
    
    def save_experiments(self, save_path: str):
        """Save all experiments to JSON"""
        results = {
            'experiment_name': self.experiment_name,
            'experiments': self.experiments
        }
        
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"Experiments saved to {save_path}")
    
    def load_experiments(self, load_path: str):
        """Load experiments from JSON"""
        with open(load_path, 'r') as f:
            results = json.load(f)
        
        self.experiment_name = results.get('experiment_name', 'Unknown')
        self.experiments = results.get('experiments', [])
        
        print(f"Loaded {len(self.experiments)} experiments from {load_path}")
    
    def get_best_experiment(self, metric: str = 'accuracy') -> Dict:
        """Get experiment with best metric"""
        if not self.experiments:
            return None
        
        best_exp = max(self.experiments, 
                      key=lambda x: x['metrics'].get(metric, 0))
        
        return best_exp
    
    def compare_experiments(self, save_path: Optional[str] = None):
        """Compare all experiments"""
        if not self.experiments:
            print("No experiments to compare")
            return
        
        # Create comparison DataFrame
        data = []
        for exp in self.experiments:
            row = {
                'model': exp['model_name'],
                'timestamp': exp['timestamp'][:10],  # Date only
            }
            row.update(exp['metrics'])
            data.append(row)
        
        df = pd.DataFrame(data)
        
        print("\n" + "="*80)
        print("EXPERIMENT COMPARISON")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80 + "\n")
        
        if save_path:
            df.to_csv(save_path, index=False)
            print(f"Comparison saved to {save_path}")


def main():
    """Demo usage"""
    # Simulated predictions
    np.random.seed(42)
    y_true = np.random.randint(0, 3, 50)
    y_pred = y_true.copy()
    # Add some errors
    error_indices = np.random.choice(len(y_true), 10, replace=False)
    y_pred[error_indices] = (y_pred[error_indices] + 1) % 3
    
    # Evaluate
    evaluator = VQCEvaluator(class_names=['Normal', 'MCI', 'AD'])
    
    metrics = evaluator.calculate_metrics(y_true, y_pred)
    evaluator.print_metrics(metrics)
    
    # Generate report
    evaluator.generate_report(y_true, y_pred, output_dir='demo_results')


if __name__ == "__main__":
    main()
