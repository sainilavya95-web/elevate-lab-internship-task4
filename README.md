# Task 4: Classification with Logistic Regression
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
