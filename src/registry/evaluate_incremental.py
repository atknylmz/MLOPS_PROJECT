"""
Evaluate and compare all incremental models
Finds the best model and prepares it for deployment
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np
from src.config import settings


def evaluate_all_models():
    """Evaluate all trained models"""
    models_dir = settings.artifacts_dir / "incremental_models"
    report_path = settings.reports_dir / "incremental_training_report.json"
    
    if not report_path.exists():
        print("❌ Incremental training report not found!")
        return None
    
    with open(report_path) as f:
        report = json.load(f)
    
    print("\n" + "="*80)
    print("MODEL EVALUATION SUMMARY")
    print("="*80 + "\n")
    
    models_data = []
    
    for result in report["models"]:
        chunk_id = result["chunk_id"]
        metrics = result["metrics"]
        is_improvement = result["is_improvement"]
        
        models_data.append({
            "Chunk": chunk_id,
            "Samples": result["cumulative_samples"],
            "Accuracy": f"{metrics['accuracy']:.4f}",
            "F1-Score": f"{metrics['f1']:.4f}",
            "Precision": f"{metrics['precision']:.4f}",
            "Recall": f"{metrics['recall']:.4f}",
            "Status": "✓ PROD" if is_improvement else "✗ LOCAL",
        })
    
    df = pd.DataFrame(models_data)
    print(df.to_string(index=False))
    print("\n")
    
    # Find best model
    summary = report["summary"]
    best_idx = None
    best_acc = 0
    
    for i, result in enumerate(report["models"]):
        if result["metrics"]["accuracy"] > best_acc:
            best_acc = result["metrics"]["accuracy"]
            best_idx = i
    
    best_model = report["models"][best_idx]
    
    print("="*80)
    print("🏆 BEST MODEL")
    print("="*80)
    print(f"Chunk: {best_model['chunk_id']}")
    print(f"Cumulative Samples: {best_model['cumulative_samples']:,}")
    print(f"Accuracy: {best_model['metrics']['accuracy']:.4f}")
    print(f"F1-Score: {best_model['metrics']['f1']:.4f}")
    print(f"Precision: {best_model['metrics']['precision']:.4f}")
    print(f"Recall: {best_model['metrics']['recall']:.4f}")
    print(f"Status: {'Production Ready' if best_model['is_improvement'] else 'Not Recommended'}")
    print("="*80 + "\n")
    
    # Save evaluation
    evaluation = {
        "best_model_chunk": best_model["chunk_id"],
        "best_accuracy": best_model["metrics"]["accuracy"],
        "best_f1": best_model["metrics"]["f1"],
        "total_models_trained": len(report["models"]),
        "summary": summary,
        "recommendation": "Deploy" if best_model["metrics"]["accuracy"] >= 0.75 else "Review",
    }
    
    eval_path = settings.reports_dir / "model_evaluation.json"
    with open(eval_path, "w") as f:
        json.dump(evaluation, f, indent=2)
    
    print(f"Evaluation saved: {eval_path}")
    
    return best_model


if __name__ == "__main__":
    evaluate_all_models()
