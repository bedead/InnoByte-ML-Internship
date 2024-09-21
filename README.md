# Customer Churn Prediction Project

This repository contains the code, datasets, and models used for predicting customer churn. The project follows a structured approach, starting with data preprocessing, followed by exploratory data analysis (EDA), model development, evaluation, and interpretation of results.

## Directory Structure

```
├── Data
│   ├── ML Dataset after transformation.csv
│   ├── ML Dataset cleaned.csv
│   └── Original ML Datasets.csv
├── Final_Model
│   └── Final_model.pkl
├── Models
│   ├── Ada_B.pkl
│   ├── B_bagging.pkl
│   ├── B_RF.pkl
│   ├── DT.pkl
│   ├── Extra tree.pkl
│   ├── Extra Tress (E).pkl
│   ├── Gradient B.pkl
│   ├── KNN.pkl
│   ├── MLP.pkl
│   ├── NB.pkl
│   └── RF.pkl
│   └── SVM.pkl
├── Notebooks
│   ├── Data Preprocessing.ipynb
│   ├── EDA.ipynb
│   ├── Model_development & evaluation.ipynb
│   └── Model_interpretation.ipynb
```

## Data folder

**Original ML Datasets.csv:**
The original dataset collected for customer churn prediction.

**ML Dataset cleaned.csv:**
The dataset after initial cleaning processes such as handling missing values, removing redundant columns, and correcting data types.

**ML Dataset after transformation.csv:**
The dataset after transformations such as encoding categorical variables and feature scaling.

## Final_Model folder

Final_model.pkl:
This file contains the final model trained using the AdaBoost algorithm. It achieved the highest accuracy in comparison to other models and is saved in pkl format for future predictions.

## Models folder

This directory includes the serialized trained models (in pkl format) for various machine learning algorithms. Each model has been trained, tested, and evaluated using the dataset.

1. Ada_B.pkl: AdaBoost model.
2. B_bagging.pkl: Bagging ensemble model.
3. B_RF.pkl: Random Forest with bagging.
4. DT.pkl: Decision Tree classifier model.
5. Extra tree.pkl: Extra Tree classifier model (Singular tree).
6. Extra Tress (E).pkl: Another Extra Trees classifier model (Multple trees).
7. Gradient B.pkl: Gradient Boosting classifier model.
8. KNN.pkl: K-Nearest Neighbors model.
9. MLP.pkl: Multi-layer Perceptron (Neural Network) model.
10. NB.pkl: Naive Bayes model.
11. RF.pkl: Random Forest model.
12. SVM.pkl: Support Vector Machine model.

## Notebooks folder

Contains Jupyter notebooks for different stages of the project.
Each notebook includes most of the required explaination and documentation.

### Data Preprocessing.ipynb

Handles missing data, removes unwanted columns, and corrects errors in data types.
Prepares the dataset for further steps by performing initial cleaning and transformations.

### EDA.ipynb

Performs Exploratory Data Analysis (EDA) on the dataset.
Provides insights into the data through visualizations and descriptive statistics.
Includes feature engineering, such as converting categorical features to numeric form.

### Model_development & evaluation.ipynb

Contains the implementation of various machine learning models, including splitting the data into training and test sets.
Defines algorithms, trains models, and evaluates them based on accuracy and other performance metrics.

### Model_interpretation.ipynb

Interprets the performance and behavior of models using techniques like SHAP and feature importance.
Provides insights into which features contribute most to churn prediction.
Includes recommendations for the company based on the results of the analysis.

## Requirements

To run the notebooks and models, the following Python packages are required:

1. pandas
2. numpy
3. scikit-learn
4. matplotlib
5. seaborn
6. SHAP
7. Jupyter Notebook

Install them using the following command:

```
pip install pandas numpy scikit-learn matplotlib seaborn shap
```

## How to Use

#### Data Preprocessing

Begin with Data Preprocessing.ipynb to clean and preprocess the dataset.
The output will be a transformed dataset that can be used for training machine learning models.

#### Exploratory Data Analysis (EDA)

Run the EDA.ipynb notebook to explore the data, understand feature distributions, and perform feature engineering.

#### Model Development & Evaluation

In Model_development & evaluation.ipynb, various machine learning models are defined, trained, and evaluated.
You can experiment with different models and parameters for churn prediction.

#### Model Interpretation

Use the Model_interpretation.ipynb to interpret the trained models using techniques such as SHAP to understand feature importance.
This notebook also provides actionable insights and recommendations for reducing customer churn.

# Insights and Recommendations

Based on the analysis from the Model_interpretation.ipynb notebook, you will find insights on key factors that contribute to customer churn.
Recommendations will be provided to help reduce churn based on the model's predictions.
