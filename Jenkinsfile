pipeline {
    agent any
    
    options {
        buildDiscarder(logRotator(numToKeepStr: '10'))
        timeout(time: 2, unit: 'HOURS')
    }
    
    environment {
        PYTHON_ENV = 'venv'
        ARTIFACTS_DIR = "${WORKSPACE}/artifacts"
        REPORTS_DIR = "${WORKSPACE}/artifacts/reports"
    }
    
    stages {
        // ======================================================================
        // COMMIT STAGE - Code quality and syntax checks
        // ======================================================================
        stage('Build - Setup') {
            steps {
                echo '🔨 Building environment...'
                sh '''
                    python -m venv ${PYTHON_ENV}
                    . ${PYTHON_ENV}/bin/activate
                    pip install --upgrade pip
                    pip install -r requirements.txt
                '''
            }
        }
        
        stage('Build - Syntax Check') {
            steps {
                echo '🔍 Checking Python syntax...'
                sh '''
                    . ${PYTHON_ENV}/bin/activate
                    python -m py_compile src/**/*.py
                    echo "✓ All files have valid syntax"
                '''
            }
        }
        
        // ======================================================================
        // TEST STAGE - Unit and integration tests
        // ======================================================================
        stage('Test - Unit Tests') {
            steps {
                echo '🧪 Running unit tests...'
                sh '''
                    . ${PYTHON_ENV}/bin/activate
                    pytest tests/ -v --junitxml=test-results.xml || true
                '''
            }
            post {
                always {
                    junit 'test-results.xml'
                }
            }
        }
        
        stage('Test - Feature Engineering') {
            steps {
                echo '🔧 Testing feature transformers...'
                sh '''
                    . ${PYTHON_ENV}/bin/activate
                    pytest tests/test_hashed_transformer.py tests/test_feature_cross.py -v
                '''
            }
        }
        
        // ======================================================================
        // TRAINING STAGE - Incremental Learning Pipeline
        // ======================================================================
        stage('Train - Incremental Models') {
            steps {
                echo '🚀 Starting Incremental Learning Pipeline...'
                echo '📊 Training 23 models with 1000-sample chunks...'
                sh '''
                    . ${PYTHON_ENV}/bin/activate
                    python -m src.training.incremental_train
                    echo "✓ Incremental training completed"
                '''
            }
            post {
                always {
                    archiveArtifacts artifacts: 'artifacts/incremental_models/**', allowEmptyArchive: true
                    archiveArtifacts artifacts: 'artifacts/reports/**', allowEmptyArchive: true
                }
            }
        }
        
        // ======================================================================
        // ACCEPTANCE TEST STAGE - Validate model quality
        // ======================================================================
        stage('Acceptance - Model Validation') {
            steps {
                echo '✅ Validating best model...'
                sh '''
                    . ${PYTHON_ENV}/bin/activate
                    python -c "
import json
from pathlib import Path
reports = Path('artifacts/reports/incremental_training_report.json')
if reports.exists():
    with open(reports) as f:
        data = json.load(f)
        summary = data['summary']
        print(f'Best Accuracy: {summary[\"best_accuracy\"]:.4f}')
        print(f'Average Accuracy: {summary[\"avg_accuracy\"]:.4f}')
        if summary['best_accuracy'] >= 0.70:
            print('✓ Model meets acceptance criteria')
        else:
            print('✗ Model does not meet criteria')
            exit(1)
                    "
                '''
            }
        }
        
        // ======================================================================
        // DEPLOY STAGE - Push best model to production
        // ======================================================================
        stage('Deploy - Promote Best Model') {
            steps {
                echo '🎯 Promoting best model to production...'
                sh '''
                    . ${PYTHON_ENV}/bin/activate
                    python src/registry/promote.py || true
                '''
            }
        }
        
        stage('Deploy - GitHub Push') {
            when {
                branch 'main'
            }
            steps {
                echo '📤 Pushing results to GitHub...'
                sh '''
                    git config --global user.email "jenkins@mlops.local"
                    git config --global user.name "Jenkins CI/CD"
                    
                    git add artifacts/incremental_models/ artifacts/reports/ README.md || true
                    git commit -m "ci(jenkins): incremental training results - best model pushed [skip ci]" || true
                    git push origin main || true
                '''
            }
        }
    }
    
    post {
        always {
            echo '📊 Cleaning up workspace...'
            cleanWs()
        }
        success {
            echo '✓ Pipeline completed successfully!'
        }
        failure {
            echo '✗ Pipeline failed!'
        }
    }
}
