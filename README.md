# Predicting Diabetes Using Machine Learning

A supervised machine-learning project that compares classification models for predicting diabetes using the Pima Indians Diabetes Dataset.

## Project overview

The notebook explores clinical measurements such as pregnancies, glucose, blood pressure, skin thickness, insulin, BMI, diabetes pedigree function, and age. The target column is `Outcome` (`1` = diabetes, `0` = no diabetes).

The project includes:
- Data exploration and cleaning
- Exploratory data analysis (correlation heatmap and target distribution)
- Train/test splitting and feature scaling
- Classification with Random Forest, Logistic Regression, Support Vector Machine (SVM), K-Nearest Neighbors (KNN), and Gaussian Naive Bayes
- Random Forest and SVM hyperparameter tuning
- ROC curve comparison and evaluation with accuracy, precision, recall, F1-score, confusion matrix, and ROC-AUC
- K-Means clustering to explore patterns in the dataset

## Repository structure

```text
diabetes-prediction-ml/
├── diabetes_prediction.ipynb
├── diabetes.csv
├── requirements.txt
├── .gitignore
└── README.md
```

## Dataset

The project uses the Pima Indians Diabetes Dataset.

Place `diabetes.csv` in the repository root before running the notebook. The CSV should contain these columns:

`Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome`

The original report references Kaggle as the dataset source but does not include the CSV itself. Add the exact dataset URL and attribution for the version used, and ensure you have the right to use and redistribute it.

## Run the project

1. Clone or download this repository.
2. Install Python 3.10 or later.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   Alternatively, install the required packages directly:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```

4. Add `diabetes.csv` to the project root.
5. Launch Jupyter:

   ```bash
   jupyter notebook diabetes_prediction.ipynb
   ```

   Alternatively, upload the notebook to Google Colab and upload the dataset.

## Models and reported results

The following test-set accuracies are reported in the supplied notebook. The corresponding classification experiments use 154 test samples with an 80/20 train-test split and `random_state=42`.

| Model | Reported test accuracy |
|---|---:|
| Random Forest (baseline) | 72.73% |
| Random Forest (tuned) | 72.73% |
| Logistic Regression | 75.32% |
| Support Vector Machine (SVM) | 72.73% |
| K-Nearest Neighbors (KNN) | 68.83% |
| Gaussian Naive Bayes | 76.62% |

The tuned Random Forest model reports a best cross-validation accuracy of approximately **78.34%**. Its reported test accuracy remains approximately 72.73%.

The ROC curve comparison reports the following approximate AUC values:

| Model | Reported ROC-AUC |
|---|---:|
| Logistic Regression | 0.81 |
| Random Forest | 0.82 |
| Support Vector Machine | 0.81 |

These results are transcribed from the supplied notebook export and have not been independently reproduced. Results may vary if the dataset, preprocessing, model configuration, or random seed changes.

The report contains a discrepancy regarding the tuned Random Forest configuration: the displayed grid-search output selects `max_depth=None`, while a separate reference mentions `max_depth=5`. Verify the final configuration when rerunning the notebook.

## Important notes on methodology

- The notebook replaces zero values in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` with missing values and discusses median imputation.
- Some preprocessing code scales the features before splitting the data, which can introduce data leakage. For a robust implementation, split the data first and fit imputation and scaling only on the training set, ideally using a scikit-learn `Pipeline`. Correcting this may change the reported results.
- K-Means is an unsupervised clustering method, not a supervised classifier. Its cluster labels are arbitrary and should not be interpreted directly as diabetes predictions without an appropriate evaluation procedure.
- Accuracy should be considered alongside precision, recall, F1-score, and ROC-AUC. Recall is particularly important when evaluating models intended to identify potential diabetes cases.

## Disclaimer

This is an educational machine-learning project, not a medical diagnostic tool. Its predictions should not be used to diagnose diabetes or replace advice from a qualified healthcare professional.

## References

- Pima Indians Diabetes Dataset — add the exact Kaggle dataset URL used in the project.
- Géron, A. (2019). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (2nd ed.). O'Reilly Media.
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
