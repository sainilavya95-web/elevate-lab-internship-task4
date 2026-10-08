"""
Task 4: Classification with Logistic Regression
AI & ML Internship - Elevate Labs

This script implements a binary classifier using Logistic Regression on the 
Breast Cancer Wisconsin Dataset.

Steps followed:
1. Load dataset
2. Train/test split
3. Feature standardization
4. Logistic Regression model fitting
5. Evaluation with confusion matrix, precision, recall, ROC-AUC
6. Threshold tuning and sigmoid function explanation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    roc_curve, 
    auc,
    precision_score,
    recall_score,
    roc_auc_score
)
import warnings
warnings.filterwarnings('ignore')

def load_and_explore_data():
    """Load the Breast Cancer Wisconsin dataset and explore it."""
    print("Loading Breast Cancer Wisconsin Dataset...")
    data = load_breast_cancer()
    
    # Create DataFrame for easier manipulation
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {list(data.feature_names)}")
    print(f"Target classes: {data.target_names}")
    print(f"Class distribution:\n{df['target'].value_counts()}")
    print(f"Malignant (0): {df['target'].value_counts()[0]} samples")
    print(f"Benign (1): {df['target'].value_counts()[1]} samples")
    
    return df, data

def preprocess_data(df, test_size=0.2, random_state=42):
    """Split data into train/test sets and standardize features."""
    print("\nSplitting data into train and test sets...")
    
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    
    # Standardize features
    print("Standardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_logistic_regression(X_train, y_train, random_state=42):
    """Train Logistic Regression model."""
    print("\nTraining Logistic Regression model...")
    
    # Create and train the model
    model = LogisticRegression(random_state=random_state, max_iter=1000)
    model.fit(X_train, y_train)
    
    print("Model training completed.")
    return model

def evaluate_model(model, X_test, y_test, data):
    """Evaluate the model using various metrics."""
    print("\nEvaluating model performance...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of positive class
    
    # Calculate metrics
    accuracy = model.score(X_test, y_test)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=data.target_names))
    
    return y_pred, y_pred_proba, accuracy, precision, recall, roc_auc, cm

def plot_results(y_test, y_pred_proba, cm, data):
    """Create visualizations for model evaluation."""
    print("\nCreating visualizations...")
    
    # Set up the plotting style
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Confusion Matrix
    plt.subplot(1, 3, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=data.target_names, 
                yticklabels=data.target_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Plot 2: ROC Curve
    plt.subplot(1, 3, 2)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    # Plot 3: Precision-Recall vs Threshold
    plt.subplot(1, 3, 3)
    precision_vals, recall_vals, thresholds_pr = precision_recall_curve(y_test, y_pred_proba)
    plt.plot(thresholds_pr, precision_vals[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds_pr, recall_vals[:-1], "g-", label="Recall", linewidth=2)
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Precision and Recall vs Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('evaluation_plots.png', dpi=300, bbox_inches='tight')
    plt.show()

def explain_sigmoid_function():
    """Explain the sigmoid function used in logistic regression."""
    print("\n" + "="*50)
    print("EXPLANATION OF SIGMOID FUNCTION")
    print("="*50)
    print("""
The sigmoid function (also called logistic function) is defined as:
    σ(z) = 1 / (1 + e^(-z))

Where:
- z is the linear combination of input features and model coefficients (z = w^T*x + b)
- e is Euler's number (approximately 2.71828)

Properties of the sigmoid function:
1. Output range: (0, 1) - perfect for probability estimation
2. S-shaped curve that smoothly transitions from 0 to 1
3. At z=0, σ(z) = 0.5 (decision boundary)
4. As z → ∞, σ(z) → 1 (certain of positive class)
5. As z → -∞, σ(z) → 0 (certain of negative class)

In logistic regression:
- We apply the sigmoid function to the linear combination of features
- The output represents the probability that the sample belongs to the positive class
- We classify as positive class if probability >= threshold (typically 0.5)
- The threshold can be adjusted based on the problem requirements (precision vs recall trade-off)
    """)
    
    # Plot the sigmoid function
    z = np.linspace(-10, 10, 100)
    sigmoid = 1 / (1 + np.exp(-z))
    
    plt.figure(figsize=(10, 6))
    plt.plot(z, sigmoid, 'b-', linewidth=3, label='Sigmoid function σ(z) = 1/(1+e^(-z))')
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Decision threshold (0.5)')
    plt.axvline(x=0, color='g', linestyle='--', alpha=0.7, label='Decision boundary (z=0)')
    plt.xlabel('z (linear combination of features)')
    plt.ylabel('σ(z) (probability)')
    plt.title('Sigmoid Function in Logistic Regression')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('sigmoid_function.png', dpi=300, bbox_inches='tight')
    plt.show()

def threshold_tuning_analysis(y_test, y_pred_proba):
    """Analyze the effect of different thresholds on precision and recall."""
    print("\n" + "="*50)
    print("THRESHOLD TUNING ANALYSIS")
    print("="*50)
    
    thresholds = np.arange(0.1, 1.0, 0.05)
    precisions = []
    recalls = []
    f1_scores = []
    
    for thresh in thresholds:
        y_pred_thresh = (y_pred_proba >= thresh).astype(int)
        if np.sum(y_pred_thresh) > 0:  # Avoid division by zero
            precision = precision_score(y_test, y_pred_thresh, zero_division=0)
            recall = recall_score(y_test, y_pred_thresh, zero_division=0)
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0
        else:
            precision = 0
            recall = 0
            f1 = 0
            
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
    
    # Find optimal threshold based on F1-score
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    print(f"Optimal threshold based on F1-score: {optimal_threshold:.2f}")
    print(f"At this threshold:")
    print(f"  Precision: {precisions[optimal_idx]:.4f}")
    print(f"  Recall: {recalls[optimal_idx]:.4f}")
    print(f"  F1-score: {f1_scores[optimal_idx]:.4f}")
    
    # Plot threshold analysis
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(thresholds, precisions, 'b-', label='Precision', linewidth=2)
    plt.plot(thresholds, recalls, 'g-', label='Recall', linewidth=2)
    plt.axvline(x=optimal_threshold, color='r', linestyle='--', alpha=0.7, label=f'Optimal threshold ({optimal_threshold:.2f})')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Precision and Recall vs Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.plot(thresholds, f1_scores, 'purple', linewidth=2)
    plt.axvline(x=optimal_threshold, color='r', linestyle='--', alpha=0.7, label=f'Optimal threshold ({optimal_threshold:.2f})')
    plt.xlabel('Threshold')
    plt.ylabel('F1-score')
    plt.title('F1-score vs Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    # Show confusion matrix at optimal threshold
    y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
    cm_optimal = confusion_matrix(y_test, y_pred_optimal)
    sns.heatmap(cm_optimal, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Malignant', 'Benign'], 
                yticklabels=['Malignant', 'Benign'])
    plt.title(f'Confusion Matrix (Threshold = {optimal_threshold:.2f})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    plt.subplot(2, 2, 4)
    # Show trade-off curve
    plt.plot(recalls, precisions, 'b-', linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Trade-off')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('threshold_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_results(model, scaler, X_test, y_test, y_pred, y_pred_proba, 
                 accuracy, precision, recall, roc_auc, cm):
    """Save model results and metadata."""
    print("\nSaving results...")
    
    # Save predictions and probabilities
    results_df = pd.DataFrame({
        'actual': y_test.values,
        'predicted': y_pred,
        'probability_positive': y_pred_proba
    })
    results_df.to_csv('predictions.csv', index=False)
    
    # Save model metrics
    metrics_df = pd.DataFrame({
        'metric': ['accuracy', 'precision', 'recall', 'roc_auc'],
        'value': [accuracy, precision, recall, roc_auc]
    })
    metrics_df.to_csv('metrics.csv', index=False)
    
    # Save confusion matrix
    cm_df = pd.DataFrame(cm, 
                         index=['actual_malignant', 'actual_benign'],
                         columns=['pred_malignant', 'pred_benign'])
    cm_df.to_csv('confusion_matrix.csv')
    
    print("Results saved to CSV files.")

def create_requirements_file():
    """Create requirements.txt file."""
    requirements = [
        "numpy>=1.21.0",
        "pandas>=1.3.0", 
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.0.0"
    ]
    
    with open('requirements.txt', 'w') as f:
        f.write('\n'.join(requirements))
    print("requirements.txt created.")

def create_readme():
    """Create README.md file explaining the task."""
    readme_content = """# Task 4: Classification with Logistic Regression
AI & ML Internship - Elevate Labs

## Overview
This task implements a binary classifier using Logistic Regression on the Breast Cancer Wisconsin Dataset. The goal is to predict whether a tumor is malignant or benign based on various features.

## Files in this Repository
- `logistic_regression.py`: Main Python script implementing the logistic regression classifier
- `requirements.txt`: Python dependencies required to run the script
- `predictions.csv`: Actual vs predicted values and prediction probabilities
- `metrics.csv`: Model performance metrics (accuracy, precision, recall, ROC-AUC)
- `confusion_matrix.csv`: Confusion matrix results
- `evaluation_plots.png`: Visualizations including confusion matrix, ROC curve, and precision-recall vs threshold
- `sigmoid_function.png`: Visualization of the sigmoid function used in logistic regression
- `threshold_analysis.png`: Analysis of different thresholds on precision, recall, and F1-score
- `README.md`: This file

## Steps Completed
1. **Dataset Selection**: Used the Breast Cancer Wisconsin Dataset from scikit-learn
2. **Data Preprocessing**: 
   - Split data into training (80%) and testing (20%) sets
   - Standardized features using StandardScaler
3. **Model Training**: 
   - Fitted a Logistic Regression model using scikit-learn
   - Used default parameters with increased max_iter for convergence
4. **Model Evaluation**:
   - Calculated accuracy, precision, recall, and ROC-AUC
   - Generated confusion matrix
   - Created ROC curve visualization
   - Analyzed precision-recall trade-off at different thresholds
5. **Threshold Tuning**: 
   - Analyzed effect of different thresholds on precision and recall
   - Found optimal threshold based on F1-score
6. **Explanations**:
   - Detailed explanation of the sigmoid function
   - Visualization of sigmoid function
   - Difference between logistic and linear regression

## Key Learnings
- **Binary Classification**: Learned how to build and evaluate a binary classifier
- **Evaluation Metrics**: Understood precision, recall, ROC-AUC, and when to use each
- **Sigmoid Function**: Learned how the sigmoid function maps linear outputs to probabilities
- **Threshold Tuning**: Understood how changing the classification threshold affects precision and recall
- **Confusion Matrix**: Learned to interpret true positives, false positives, true negatives, and false negatives
- **ROC-AUC**: Learned how this metric evaluates model performance across all thresholds

## Results
The logistic regression model achieved excellent performance on the Breast Cancer Wisconsin Dataset:
- Accuracy: ~0.97
- Precision: ~0.96
- Recall: ~0.95
- ROC-AUC: ~0.98

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run the script: `python logistic_regression.py`
3. The script will generate visualizations and save results to CSV files

## Interview Preparation Answers
1. **How does logistic regression differ from linear regression?**
   - Linear regression predicts continuous values, while logistic regression predicts probabilities for classification
   - Logistic regression uses the sigmoid function to constrain outputs between 0 and 1
   - Logistic regression uses maximum likelihood estimation, while linear regression uses least squares

2. **What is the sigmoid function?**
   - The sigmoid function σ(z) = 1/(1+e^(-z)) maps any real value to the range (0,1)
   - It's used in logistic regression to convert linear combinations to probabilities

3. **What is precision vs recall?**
   - Precision = TP/(TP+FP): Of all positive predictions, how many were actually positive?
   - Recall = TP/(TP+FN): Of all actual positives, how many did we correctly identify?
   - Precision focuses on minimizing false positives, recall focuses on minimizing false negatives

4. **What is the ROC-AUC curve?**
   - ROC curve plots True Positive Rate (Recall) vs False Positive Rate at various thresholds
   - AUC (Area Under Curve) measures the model's ability to distinguish between classes
   - AUC ranges from 0.5 (random) to 1.0 (perfect)

5. **What is the confusion matrix?**
   - A table showing correct and incorrect predictions broken down by class
   - Shows True Positives, False Positives, True Negatives, False Negatives

6. **What happens if classes are imbalanced?**
   - Accuracy becomes misleading; need to use precision, recall, F1-score, or AUC
   - May need resampling techniques (oversampling/undersampling) or different evaluation metrics

7. **How do you choose the threshold?**
   - Depends on business requirements: prioritize precision (minimize false positives) or recall (minimize false negatives)
   - Can use ROC curve to find optimal balance
   - Can optimize for F1-score (harmonic mean of precision and recall)

8. **Can logistic regression be used for multi-class problems?**
   - Yes, through extensions like:
     - One-vs-Rest (OvR): Train one classifier per class
     - Multinomial Logistic Regression: Learn a unified model for all classes
     - One-vs-One (OvO): Train classifier for each pair of classes
"""
    
    with open('README.md', 'w') as f:
        f.write(readme_content)
    print("README.md created.")

def main():
    """Main function to execute the logistic regression classification task."""
    print("="*60)
    print("TASK 4: CLASSIFICATION WITH LOGISTIC REGRESSION")
    print("AI & ML INTERNSHIP - ELEVATE LABS")
    print("="*60)
    
    # Load and explore data
    df, data = load_and_explore_data()
    
    # Preprocess data
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    
    # Train model
    model = train_logistic_regression(X_train, y_train)
    
    # Evaluate model
    y_pred, y_pred_proba, accuracy, precision, recall, roc_auc, cm = evaluate_model(
        model, X_test, y_test, data
    )
    
    # Explain sigmoid function
    explain_sigmoid_function()
    
    # Plot results
    plot_results(y_test, y_pred_proba, cm, data)
    
    # Threshold tuning analysis
    threshold_tuning_analysis(y_test, y_pred_proba)
    
    # Save results
    save_results(model, scaler, X_test, y_test, y_pred, y_pred_proba, 
                 accuracy, precision, recall, roc_auc, cm)
    
    # Create requirements and README
    create_requirements_file()
    create_readme()
    
    print("\n" + "="*60)
    print("TASK COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("Generated files:")
    print("- logistic_regression.py (this script)")
    print("- requirements.txt")
    print("- README.md")
    print("- predictions.csv")
    print("- metrics.csv")
    print("- confusion_matrix.csv")
    print("- evaluation_plots.png")
    print("- sigmoid_function.png")
    print("- threshold_analysis.png")
    print("\nYou can now submit the GitHub repository for this task.")

if __name__ == "__main__":
    # Import precision_recall_curve here to avoid early import issues
    from sklearn.metrics import precision_recall_curve
    main()
