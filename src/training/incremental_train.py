"""
Incremental Learning Pipeline - 23 Model Training
Splits CSV into chunks of 1000 rows and trains models incrementally
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import settings
from src.utils.io import ensure_feature_cross
from src.features.hashed_features import HashedCategoricalTransformer
def _make_pipeline_light(numeric_cols: list[str], categorical_cols: list[str], hash_buckets: int):
    """Lightweight pipeline for incremental training"""
    hashed = HashedCategoricalTransformer(cols=categorical_cols, n_features=hash_buckets)

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("hashcat", hashed, categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    # Faster ensemble - reduced trees for speed
    clf1 = LogisticRegression(max_iter=1000)
    clf2 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10)

    ensemble = VotingClassifier(
        estimators=[("lr", clf1), ("rf", clf2)],
        voting="soft",
    )

    # Rebalancing with SMOTE
    pipe = ImbPipeline(
        steps=[
            ("pre", pre),
            ("smote", SMOTE(random_state=42, k_neighbors=3)),
            ("model", ensemble),
        ]
    )
    return pipe


class IncrementalTrainer:
    def __init__(self, csv_path: str, chunk_size: int = 1000):
        self.csv_path = Path(csv_path)
        self.chunk_size = chunk_size
        self.models_dir = settings.artifacts_dir / "incremental_models"
        self.incremental_report = settings.reports_dir / "incremental_training_report.json"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Load full data
        print(f"Loading data from {self.csv_path}...")
        self.df = pd.read_csv(self.csv_path)
        print(f"Total samples loaded: {len(self.df)}")
        
        # Calculate number of chunks
        self.num_chunks = len(self.df) // self.chunk_size
        print(f"Number of {self.chunk_size}-row chunks: {self.num_chunks}")
        
        # Setup columns
        self.target = settings.target_col
        self.cat_cols = [c for c in settings.categorical_cols if c in self.df.columns]
        self.numeric_cols = [c for c in self.df.columns if c not in self.cat_cols + [self.target]]
        
        # Results tracking
        self.results = []
        self.model_scores = {}
        
    def train_incremental_models(self):
        """Train models with incremental data"""
        print("\n" + "="*80)
        print("STARTING INCREMENTAL TRAINING PIPELINE")
        print("="*80 + "\n")
        
        cumulative_df = pd.DataFrame()
        previous_accuracy = 0.0
        
        for chunk_idx in range(self.num_chunks):
            chunk_start = chunk_idx * self.chunk_size
            chunk_end = chunk_start + self.chunk_size
            
            current_chunk = self.df.iloc[chunk_start:chunk_end].copy()
            
            # Append to cumulative data
            cumulative_df = pd.concat([cumulative_df, current_chunk], ignore_index=True)
            
            print(f"\n{'='*80}")
            print(f"CHUNK {chunk_idx + 1}/{self.num_chunks}")
            print(f"{'='*80}")
            print(f"Chunk rows: {chunk_start} - {chunk_end}")
            print(f"Cumulative training samples: {len(cumulative_df)}")
            
            # Prepare data
            df_processed = ensure_feature_cross(cumulative_df.copy())
            
            X = df_processed[self.numeric_cols + self.cat_cols].copy()
            y = df_processed[self.target].astype(int)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Build and train model
            pipe = _make_pipeline_light(self.numeric_cols, self.cat_cols, settings.hash_buckets)
            
            print(f"Training model...")
            pipe.fit(X_train, y_train)
            
            # Evaluate
            prob = pipe.predict_proba(X_test)[:, 1]
            pred = (prob >= 0.5).astype(int)
            
            metrics = {
                "accuracy": float(accuracy_score(y_test, pred)),
                "precision": float(precision_score(y_test, pred, zero_division=0)),
                "recall": float(recall_score(y_test, pred, zero_division=0)),
                "f1": float(f1_score(y_test, pred, zero_division=0)),
            }
            
            current_accuracy = metrics["accuracy"]
            
            print(f"Accuracy: {current_accuracy:.4f}")
            print(f"F1-Score: {metrics['f1']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            
            # Determine if model should be pushed to production
            is_better = current_accuracy >= previous_accuracy
            status = "✓ PASSED - Will be pushed to GitHub" if is_better else "✗ FAILED - Stored locally only"
            
            print(f"Previous Accuracy: {previous_accuracy:.4f}")
            print(f"Status: {status}")
            
            # Save model
            model_dir = self.models_dir / f"model_{chunk_idx + 1:02d}"
            model_dir.mkdir(parents=True, exist_ok=True)
            
            model_path = model_dir / f"model_chunk_{chunk_idx + 1:02d}.joblib"
            dump(pipe, model_path)
            
            # Save metadata
            metadata = {
                "chunk_id": chunk_idx + 1,
                "total_chunks": self.num_chunks,
                "chunk_size": self.chunk_size,
                "cumulative_samples": len(cumulative_df),
                "test_samples": len(X_test),
                "metrics": metrics,
                "previous_accuracy": previous_accuracy,
                "is_improvement": is_better,
                "status": "production" if is_better else "local_only",
            }
            
            metadata_path = model_dir / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            
            self.results.append(metadata)
            self.model_scores[chunk_idx + 1] = {
                "accuracy": current_accuracy,
                "f1": metrics["f1"],
                "model_path": str(model_path),
                "is_improvement": is_better,
            }
            
            if is_better:
                previous_accuracy = current_accuracy
            
            print(f"Model saved to: {model_path}")
            
        self._generate_report()
        self._find_best_model()
        
    def _generate_report(self):
        """Generate comprehensive report"""
        print("\n" + "="*80)
        print("GENERATING INCREMENTAL TRAINING REPORT")
        print("="*80 + "\n")
        
        report = {
            "project": "Manufacturing Defect Detection - Incremental Learning",
            "total_chunks": self.num_chunks,
            "chunk_size": self.chunk_size,
            "total_samples": len(self.df),
            "models": self.results,
            "summary": self._create_summary(),
        }
        
        with open(self.incremental_report, "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"Report saved: {self.incremental_report}")
        
        # Create visualization
        self._create_visualization()
        
    def _create_summary(self):
        """Create summary statistics"""
        accuracies = [r["metrics"]["accuracy"] for r in self.results]
        f1_scores = [r["metrics"]["f1"] for r in self.results]
        
        return {
            "best_accuracy": max(accuracies),
            "worst_accuracy": min(accuracies),
            "avg_accuracy": np.mean(accuracies),
            "best_f1": max(f1_scores),
            "worst_f1": min(f1_scores),
            "avg_f1": np.mean(f1_scores),
            "improvements_count": sum(1 for r in self.results if r["is_improvement"]),
            "local_only_count": sum(1 for r in self.results if not r["is_improvement"]),
        }
        
    def _create_visualization(self):
        """Create performance visualizations"""
        chunk_ids = [r["chunk_id"] for r in self.results]
        accuracies = [r["metrics"]["accuracy"] for r in self.results]
        f1_scores = [r["metrics"]["f1"] for r in self.results]
        improvements = [r["is_improvement"] for r in self.results]
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Incremental Learning Pipeline - Performance Analysis", fontsize=16, fontweight='bold')
        
        # Accuracy over chunks
        colors = ['green' if imp else 'red' for imp in improvements]
        axes[0, 0].bar(chunk_ids, accuracies, color=colors, alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('Chunk Number')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Accuracy by Chunk (Green=Improvement, Red=Degradation)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=np.mean(accuracies), color='blue', linestyle='--', label='Mean', linewidth=2)
        axes[0, 0].legend()
        
        # F1 Score over chunks
        axes[0, 1].plot(chunk_ids, f1_scores, marker='o', linewidth=2, markersize=8, color='purple')
        axes[0, 1].fill_between(chunk_ids, f1_scores, alpha=0.3, color='purple')
        axes[0, 1].set_xlabel('Chunk Number')
        axes[0, 1].set_ylabel('F1 Score')
        axes[0, 1].set_title('F1 Score Trend')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Accuracy vs F1
        axes[1, 0].scatter(accuracies, f1_scores, s=100, c=chunk_ids, cmap='viridis', edgecolor='black')
        axes[1, 0].set_xlabel('Accuracy')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].set_title('Accuracy vs F1 Score')
        axes[1, 0].grid(True, alpha=0.3)
        cbar = plt.colorbar(axes[1, 0].collections[0], ax=axes[1, 0])
        cbar.set_label('Chunk ID')
        
        # Summary statistics
        axes[1, 1].axis('off')
        summary_text = f"""
SUMMARY STATISTICS

Total Chunks: {self.num_chunks}
Total Samples: {len(self.df):,}

Best Accuracy: {max(accuracies):.4f}
Worst Accuracy: {min(accuracies):.4f}
Mean Accuracy: {np.mean(accuracies):.4f}

Best F1 Score: {max(f1_scores):.4f}
Worst F1 Score: {min(f1_scores):.4f}
Mean F1 Score: {np.mean(f1_scores):.4f}

Improvements: {sum(improvements)}
Degradations: {len(improvements) - sum(improvements)}
        """
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                       verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Save figure
        viz_path = settings.reports_dir / "incremental_learning_visualization.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved: {viz_path}")
        plt.close()
        
    def _find_best_model(self):
        """Find and highlight the best performing model"""
        best_idx = max(range(len(self.results)), 
                      key=lambda i: self.results[i]["metrics"]["accuracy"])
        best_result = self.results[best_idx]
        
        print("\n" + "="*80)
        print("BEST MODEL FOUND")
        print("="*80)
        print(f"Chunk: {best_result['chunk_id']}/{self.num_chunks}")
        print(f"Cumulative Samples: {best_result['cumulative_samples']:,}")
        print(f"Accuracy: {best_result['metrics']['accuracy']:.4f}")
        print(f"F1 Score: {best_result['metrics']['f1']:.4f}")
        print(f"Precision: {best_result['metrics']['precision']:.4f}")
        print(f"Recall: {best_result['metrics']['recall']:.4f}")
        print("="*80 + "\n")
        
        return best_result


if __name__ == "__main__":
    trainer = IncrementalTrainer(
        csv_path=str(settings.raw_csv),
        chunk_size=1000
    )
    trainer.train_incremental_models()
