# Multi-Class Model Selection using ROC and Precision-Recall Curves

- Name : Saranath P
- Roll : DA25E003

## Overview
This repository contains a comprehensive analysis of multi-class classification techniques applied to the UCI Landsat Satellite dataset. The project demonstrates advanced model selection methodologies using Receiver Operating Characteristic (ROC) curves and Precision-Recall Curves (PRC), particularly focusing on the One-vs-Rest (OvR) approach for multi-class problems.

## Dataset Information
The analysis uses the **UCI Landsat Satellite dataset**, which contains multi-spectral satellite image data classified into 6 distinct land cover types (classes 1, 2, 3, 4, 5, and 7, with class 6 ignored as per standard practice). Each instance consists of 36 features derived from 4 spectral bands in a 3x3 pixel neighborhood.

## Project Structure
- `da25e003.ipynb`: Main Jupyter notebook containing all code, analysis, and visualizations
- `statlog+landsat+satellite/`: Directory containing the original dataset files
  - `sat.trn`: Training dataset
  - `sat.tst`: Test dataset
  - `sat.doc`: Dataset documentation
  - `Index`: Index file

## Methodology

### Data Preparation
- Loaded pre-split training and testing data from `sat.trn` and `sat.tst`
- Removed unused class label 6 (as per standard practice)
- Applied standardization using `StandardScaler` to normalize feature scales
- Verified scaling effectiveness through visualization of feature distributions

### Model Evaluation Strategy
The project employs a comprehensive model evaluation approach:
1. **Base metrics**: Accuracy and Weighted F1-Score for initial assessment
2. **ROC Analysis**: Using One-vs-Rest (OvR) approach with macro-averaging to evaluate model performance across all classes
3. **Precision-Recall Analysis**: Using OvR with macro-averaging to account for class imbalance
4. **Per-class diagnostics**: Examining confusion matrices and class-specific metrics

## Experimental Results

### 1. Baseline Model Comparison

The following models were evaluated on the test set:

| Model                  | Accuracy | Weighted F1-Score | Macro-AUC | Macro-AP |
|------------------------|----------|-------------------|-----------|----------|
| K-Nearest Neighbors    | 0.9045   | 0.9037            | 0.979     | 0.922    |
| Support Vector Machine | 0.8955   | 0.8925            | 0.985     | 0.918    |
| Decision Tree          | 0.8505   | 0.8509            | 0.900     | 0.737    |
| Logistic Regression    | 0.8395   | 0.8296            | 0.976     | 0.871    |
| Gaussian Naive Bayes   | 0.7965   | 0.8036            | 0.955     | 0.810    |
| Dummy Classifier       | 0.2305   | 0.0864            | 0.500     | 0.167    |

**Key Findings**:
- KNN achieved the highest accuracy and weighted F1-score, suggesting strong point prediction capabilities
- SVM had the highest macro-AUC, indicating superior class separation and threshold-independent performance
- All models significantly outperformed the dummy baseline, confirming that the models are learning meaningful patterns

### 2. ROC Analysis Results

The Macro-Averaged One-vs-Rest ROC curves demonstrated:
- SVM achieved the highest macro-AUC (0.985), indicating excellent class discrimination capabilities
- KNN followed closely with a macro-AUC of 0.979
- Even linear models like Logistic Regression performed well (macro-AUC 0.976)
- The Decision Tree had the lowest performance among real models (macro-AUC 0.900)

### 3. Precision-Recall Analysis

The PRC analysis, which is more sensitive to class imbalance, showed:
- KNN led with a macro-AP of 0.922
- SVM followed very closely at 0.918
- Logistic Regression achieved 0.871
- The no-skill baseline was approximately 0.167 (matching the Dummy classifier's performance)

### 4. Advanced Model Experiments

**Enhanced Models**:
- **Random Forest**: Achieved accuracy of 0.913, weighted F1-score of 0.912, macro-AUC of 0.990, and macro-AP of 0.923
- **XGBoost**: Achieved accuracy of 0.916, weighted F1-score of 0.916, macro-AUC of 0.991, and macro-AP of 0.924

**Contrarian Classifier Experiment**:
- Created a "Contrarian Logistic Regression" model by inverting the probabilities of a standard Logistic Regression model
- The model achieved a macro-AUC of approximately 0.025 (far below 0.5)
- This confirmed that AUC < 0.5 indicates a model making systematically incorrect predictions (worse than random guessing)

### 5. Deeper Diagnostic Analysis

**Per-Class Performance**:
- XGBoost showed consistently excellent performance across all classes (AUC > 0.99, AP > 0.96)
- Class 4 (grey soil) was identified as slightly more challenging but still had strong performance

**Confusion Matrix Analysis**:
- Tree-based ensembles (Random Forest and XGBoost) were superior at handling difficult classes (3, 4, and 7)
- SVM showed more errors on class 4 classification
- All models performed nearly perfectly on classes 1 and 2

**Probability Calibration**:
- XGBoost and Random Forest produced excellent classifications but poorly calibrated probabilities
- Random Forest exhibited a classic sigmoid curve in calibration plots
- SVM's probability estimates (using Platt scaling) were more stable but not perfectly calibrated

**Feature Importance**:
- XGBoost's feature importance analysis showed that a subset of the 36 features contributed most to model predictions
- This suggests potential for model simplification through feature selection

**Cross-Validation**:
- 5-fold cross-validation confirmed the stability and robustness of both XGBoost and Random Forest models
- Random Forest showed slightly higher mean F1-score with low standard deviation

## Conclusions and Recommendations

### Best Model Selection
Based on the comprehensive analysis:

1. **For highest point prediction accuracy**: K-Nearest Neighbors or ensemble models (XGBoost/Random Forest)
2. **For best overall performance with ranking capabilities**: XGBoost or Random Forest
3. **For well-calibrated probabilities**: SVM (after calibration) offers more reliable probability scores

### Key Insights
1. **Different metrics reveal different strengths**: Models can rank differently depending on whether you prioritize accuracy, AUC, or AP
2. **Ensemble advantages**: Tree-based ensembles consistently outperformed single models, especially on challenging classes
3. **Probability calibration matters**: High classification accuracy doesn't guarantee well-calibrated probabilities
4. **Class-specific performance**: Models that perform well on average may still struggle with specific classes

### Practical Recommendations
1. For deployment, use ensemble methods (XGBoost/Random Forest) with probability calibration
2. Consider feature selection based on importance analysis to simplify models
3. For applications requiring reliable probability estimates, apply calibration techniques to the chosen model
4. When class-specific performance is critical, evaluate per-class metrics rather than relying solely on macro-averages

## Methodological Contributions
1. Demonstrated the value of using both ROC and PRC curves for model evaluation
2. Illustrated how macro-averaging provides a fair assessment across all classes in imbalanced settings
3. Showed how detailed per-class analysis can reveal insights not visible in aggregate metrics
4. Confirmed the theoretical understanding of AUC < 0.5 through the contrarian classifier experiment