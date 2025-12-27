from __future__ import annotations

import json
import subprocess
from pathlib import Path
from datetime import datetime
import mlflow
from mlflow.tracking import MlflowClient
from src.config import settings


def promote_latest_to_stage(stage: str = "Staging") -> None:
    """Promote latest model version to a stage."""
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{settings.model_name}'")
    if not versions:
        raise RuntimeError("No model versions found to promote.")

    # latest by version number (string)
    latest = max(versions, key=lambda v: int(v.version))
    client.transition_model_version_stage(
        name=settings.model_name,
        version=latest.version,
        stage=stage,
        archive_existing_versions=True,
    )


def rollback_production() -> None:
    """Rollback Production to previous version (if exists)."""
    client = MlflowClient()
    versions = sorted(
        client.search_model_versions(f"name='{settings.model_name}'"),
        key=lambda v: int(v.version),
        reverse=True
    )
    prod = [v for v in versions if v.current_stage == "Production"]
    if not prod:
        raise RuntimeError("No Production model to rollback.")

    # pick newest non-production as fallback (usually Staging or Archived)
    fallback = next((v for v in versions if v.current_stage in {"Staging", "Archived"}), None)
    if fallback is None:
        raise RuntimeError("No fallback model version found.")

    client.transition_model_version_stage(
        name=settings.model_name,
        version=fallback.version,
        stage="Production",
        archive_existing_versions=True,
    )


def generate_readme():
    """Generate comprehensive README with model details"""
    
    report_path = settings.reports_dir / "incremental_training_report.json"
    eval_path = settings.reports_dir / "model_evaluation.json"
    
    if not report_path.exists() or not eval_path.exists():
        print("[!] Report files not found. Skipping README generation.")
        return
    
    with open(report_path) as f:
        report = json.load(f)
    
    with open(eval_path) as f:
        evaluation = json.load(f)
    
    best_model = None
    for m in report["models"]:
        if m["chunk_id"] == evaluation["best_model_chunk"]:
            best_model = m
            break
    
    if not best_model:
        print("[!] Best model not found in report!")
        return
    
    summary = report["summary"]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    readme_content = f"""# Manufacturing Defect Detection - MLOps Project

**Status**: Active  
**Model Version**: v{best_model['chunk_id']}  
**Best Accuracy**: {best_model['metrics']['accuracy']:.4f}  
**Last Updated**: {timestamp}

## 📊 Project Overview

This is an **MLOps production-ready project** implementing an **Incremental Learning Pipeline** for manufacturing defect detection. The system automatically trains multiple models with expanding datasets, evaluates performance, and deploys the best performing model to production.

## 🎯 Best Model Performance

### Model Information
- **Model ID**: Model_{best_model['chunk_id']:02d}
- **Training Samples**: {best_model['cumulative_samples']:,}
- **Training Data Chunks**: {best_model['chunk_id']}/{report['total_chunks']}
- **Deployment Status**: {'Production Ready' if best_model['is_improvement'] else 'Review Required'}

### Key Metrics

| Metric | Value |
|--------|-------|
| **Accuracy** | {best_model['metrics']['accuracy']:.4f} |
| **F1-Score** | {best_model['metrics']['f1']:.4f} |
| **Precision** | {best_model['metrics']['precision']:.4f} |
| **Recall** | {best_model['metrics']['recall']:.4f} |

## 📈 Incremental Learning Pipeline

### Pipeline Configuration
- **Total Chunks Trained**: {report['total_chunks']}
- **Samples per Chunk**: {report['chunk_size']:,}
- **Total Training Samples**: {report['total_samples']:,}

### Performance Summary

| Metric | Value |
|--------|-------|
| Best Accuracy | {summary['best_accuracy']:.4f} |
| Worst Accuracy | {summary['worst_accuracy']:.4f} |
| Mean Accuracy | {summary['avg_accuracy']:.4f} |
| Best F1-Score | {summary['best_f1']:.4f} |
| Mean F1-Score | {summary['avg_f1']:.4f} |
| Improvements | {summary['improvements_count']} |
| Local-Only Models | {summary['local_only_count']} |

### Learning Curve

![Incremental Learning Results](artifacts/reports/incremental_learning_visualization.png)

## 🏗️ Architecture

### Project Structure
```
mlops_defect_project/
├── src/
│   ├── training/
│   │   ├── train.py                 # Base training module
│   │   └── incremental_train.py      # Incremental learning pipeline
│   ├── registry/
│   │   ├── promote.py               # Model promotion script
│   │   └── evaluate_incremental.py  # Model evaluation
│   ├── features/
│   │   └── hashed_features.py       # Feature engineering
│   ├── serving/
│   │   └── app.py                   # FastAPI serving
│   └── config.py                    # Configuration
├── artifacts/
│   ├── incremental_models/          # All trained models
│   ├── checkpoints/                 # Model checkpoints
│   └── reports/                     # Training reports
├── data/
│   └── numeric_data.csv             # Training data
├── .gitlab-ci.yml                   # GitLab CI/CD
├── Jenkinsfile                      # Jenkins CI/CD
├── .circleci/config.yml            # CircleCI config
└── requirements.txt                 # Dependencies
```

## 🚀 CI/CD Pipelines

This project includes three enterprise-grade CI/CD implementations:

### 1️⃣ **GitLab CI/CD** (.gitlab-ci.yml)
- **Stages**: Build → Test → Train → Evaluate → Deploy
- **Key Features**:
  - Automated dependency installation
  - Python syntax validation
  - Unit and integration testing
  - Incremental model training
  - Automatic model comparison
  - GitHub push on success

### 2️⃣ **Jenkins** (Jenkinsfile)
- **Pipelines**: Declarative pipeline with multiple stages
- **Key Features**:
  - Build environment setup with venv
  - Comprehensive test suite
  - Incremental learning with timeout handling
  - Artifact archiving
  - GitHub integration
  - Clean workspace management

### 3️⃣ **CircleCI** (.circleci/config.yml)
- **Workflows**: Multi-job orchestration with caching
- **Key Features**:
  - Dependency caching for faster builds
  - Parallel job execution
  - Workspace persistence
  - Large resource class for training
  - GitHub push with commit messages

### Pipeline Stages

#### 1. **Commit Stage** ✓
- Code checkout and syntax validation
- Dependency management
- Import checks
- Quality gates

#### 2. **Test Stage** ✓
- Unit tests execution
- Feature engineering validation
- Test coverage reporting

#### 3. **Training Stage** 🚀
- Splits CSV into 1000-sample chunks
- Trains 23 models incrementally
- Each model adds new data to previous
- Compares accuracy with previous model
- Saves models and metadata

#### 4. **Acceptance Test Stage** ✅
- Validates model quality (min 0.70 accuracy)
- Checks acceptance criteria
- Generates evaluation reports

#### 5. **Deploy Stage** 📤
- Promotes best model to production
- Updates README with results
- Commits and pushes to GitHub
- Only runs on main branch

## 📊 Model Training Details

### Incremental Learning Strategy

**Data Split**: CSV with {report['total_samples']:,} rows → {report['total_chunks']} chunks of {report['chunk_size']:,} rows each

**Training Process**:
```
Chunk 1:    Train Model 1 with 1,000 samples (Accuracy: baseline)
Chunk 2:    Train Model 2 with 2,000 samples (Compare with Model 1)
Chunk 3:    Train Model 3 with 3,000 samples (Compare with Model 2)
...
Chunk 23:   Train Model 23 with 23,000 samples (Final best model)
```

**Promotion Logic**:
- ✅ **IF** accuracy ≥ previous → Push to production
- ❌ **ELSE** → Store locally only (no push)

### Feature Engineering

- **Numeric Features**: StandardScaler normalization
- **Categorical Features**: Hashed embeddings (4096 dimensions)
- **Target**: DefectStatus (binary classification)
- **Imbalance Handling**: SMOTE resampling

### Model Ensemble

Voting Classifier combining:
- **Logistic Regression** (linear patterns)
- **Random Forest** (non-linear patterns, 250 trees)
- **Gradient Boosting** (adaptive learning)

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.11+
- Git
- Docker (optional)

### Installation

```bash
# Clone repository
git clone https://github.com/atknylmz/MLOPS_PROJECT.git
cd MLOPS_PROJECT

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-cov black pylint
```

### Running Training Locally

```bash
# Run incremental training pipeline
python -m src.training.incremental_train

# Evaluate models
python src/registry/evaluate_incremental.py

# Promote best model
python src/registry/promote.py
```

## 📝 Configuration

Edit `src/config.py` to customize:

```python
hash_buckets: int = 2**12        # Feature hashing dimensions
target_col: str = "DefectStatus" # Target variable
categorical_cols: tuple[str, ...]  # Categorical features
```

## 🔗 API Serving

FastAPI server for model serving:

```bash
# Start API server
python -m src.serving.app

# API will be available at http://localhost:8000
# Swagger docs: http://localhost:8000/docs
```

## 📦 Deployment

### Docker

```bash
docker build -t mlops-defect-model .
docker run -p 8000:8000 mlops-defect-model
```

### Cloud Platforms

- **AWS**: SageMaker, Lambda, EC2
- **Google Cloud**: Vertex AI, Cloud Run
- **Azure**: ML Studio, Container Instances

## 📊 Monitoring & MLflow

All models are tracked in MLflow:

```bash
# Start MLflow UI
mlflow ui

# View all experiments at http://localhost:5000
```

## 🔄 Continuous Integration

### GitLab CI
```bash
git push origin main  # Automatically triggers pipeline
```

### Jenkins
- Configure GitHub webhook in Jenkins
- Set up Jenkins job from Jenkinsfile
- Auto-triggers on commit to main

### CircleCI
- Connect GitHub repo to CircleCI
- Auto-triggers on commit
- View progress at circleci.com

## 📊 Results & Artifacts

All results are stored in `artifacts/`:

```
artifacts/
├── incremental_models/
│   ├── model_01/
│   │   ├── model_chunk_01.joblib
│   │   └── metadata.json
│   ├── model_02/
│   │   ├── model_chunk_02.joblib
│   │   └── metadata.json
│   ...
│   └── model_23/
│       ├── model_chunk_23.joblib
│       └── metadata.json
│
├── checkpoints/
│   ├── model_*.joblib
│   └── acceptance_*.json
│
└── reports/
    ├── incremental_training_report.json
    ├── model_evaluation.json
    └── incremental_learning_visualization.png
```

## 🎓 Learning Resources

- [MLflow Documentation](https://mlflow.org)
- [Scikit-learn Ensemble Methods](https://scikit-learn.org/stable/modules/ensemble.html)
- [GitLab CI/CD](https://docs.gitlab.com/ee/ci/)
- [Jenkins Documentation](https://www.jenkins.io/doc/)
- [CircleCI Documentation](https://circleci.com/docs/)

## 🤝 Contributing

1. Create feature branch: `git checkout -b feature/improvement`
2. Make changes and test locally
3. Commit: `git commit -m "feat: description"`
4. Push: `git push origin feature/improvement`
5. Create Pull Request

## 📄 License

MIT License - See LICENSE file for details

## 👤 Author

**MLOps Team**  
- GitHub: [@atknylmz](https://github.com/atknylmz)
- Email: atknylmz@email.com

## 📞 Support

For issues and questions:
- Open GitHub Issues
- Contact: atknylmz@email.com
- Documentation: See `/docs` folder

---

**Last Updated**: {timestamp}  
**Best Model Accuracy**: {best_model['metrics']['accuracy']:.4f}  
**Total Models Trained**: {report['total_chunks']}  
**Project Status**: ✅ Active & Maintained
"""
    
    readme_path = settings.project_root / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    print(f"[+] README.md generated: {readme_path}")


def push_to_github():
    """Commit and push to GitHub"""
    
    try:
        print("\n" + "="*80)
        print("PUSHING TO GITHUB")
        print("="*80 + "\n")
        
        # Configure git
        subprocess.run(["git", "config", "user.email", "mlops@automated.local"], check=True)
        subprocess.run(["git", "config", "user.name", "MLOps CI/CD"], check=True)
        
        print("[+] Git configured")
        
        # Add changes
        subprocess.run(["git", "add", "artifacts/incremental_models/", "artifacts/reports/", "README.md"], check=True)
        print("[+] Changes staged")
        
        # Commit
        try:
            subprocess.run(
                ["git", "commit", "-m", "ci: incremental training results - best model deployed"],
                check=True
            )
            print("[+] Changes committed")
        except subprocess.CalledProcessError:
            print("[!] No changes to commit")
        
        # Push
        subprocess.run(["git", "push", "origin", "main"], check=True)
        print("[+] Pushed to GitHub main branch")
        
        print("\n" + "="*80)
        print("GITHUB PUSH COMPLETED SUCCESSFULLY")
        print("="*80 + "\n")
        
    except subprocess.CalledProcessError as e:
        print(f"\n[!] Git operation failed: {e}")
        print("[!] Make sure you have GitHub credentials configured")
        raise

if __name__ == "__main__":
    import sys
    import os
    os.chdir(str(settings.project_root))
    sys.stdout.reconfigure(encoding='utf-8')
    
    print("\n[*] Preparing deployment...")
    
    # Generate README
    generate_readme()
    
    # Push to GitHub
    push_to_github()
    
    print("\n[+] Deployment completed!")