# Happiness Survey Analysis & Prediction

This project analyzes the **ACME Happiness Survey 2020** dataset to explore the factors influencing happiness and build machine learning models to predict survey responses.

## 📌 Project Overview
- **Exploratory Data Analysis (EDA):**
  - Univariate, bivariate, and multivariate analysis
  - Visualization of feature distributions and relationships
  - Correlation heatmaps

- **Feature Engineering:**
  - Derived interaction feature `X5*X6`
  - Handled categorical and numerical preprocessing

- **Modeling:**
  - Logistic Regression
  - Decision Tree Classifier
  - Random Forest Classifier
  - Gradient Boosting Classifier
  - Stacking Classifier (ensemble)

- **Evaluation:**
  - Accuracy
  - Comparative performance of models

## 📊 Data Description
- **Y**: Target attribute  
  - `0` → Unhappy customer  
  - `1` → Happy customer  
- **X1**: My order was delivered on time  
- **X2**: Contents of my order were as I expected  
- **X3**: I ordered everything I wanted to order  
- **X4**: I paid a good price for my order  
- **X5**: I am satisfied with my courier  
- **X6**: The app makes ordering easy for me  

All attributes **X1–X6** are rated on a scale from **1 to 5**, where smaller values indicate lower satisfaction and higher values indicate greater satisfaction.


## 📂 Project Structure
```
├── full_code.ipynb   # Jupyter Notebook with full workflow
├── README.md         # Project documentation
└── requirements.txt  # Dependencies
```

⚠️ **Note on Data Access**  
The dataset used in this project is **internal and cannot be shared**.  
If you wish to reproduce the workflow, you can replace the dataset with your own CSV file, ensuring it has a similar structure.

## ⚙️ Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/AishwaryaBalajee/happiness-survey-analysis.git
cd happiness-survey-analysis
pip install -r requirements.txt
```

## ▶️ Usage
Open the notebook and run all cells:

```bash
jupyter notebook full_code.ipynb
```

## 📊 Results
- Gradient Boosting and Stacking classifiers showed strong performance.
- Feature interaction (`X5*X6`) improved predictive power.
- Insights from EDA highlight which features contribute most to survey responses.

## 🚀 Future Work
- Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)
- Cross-validation for model robustness
- Deployment with a simple web app (Flask/Streamlit)
