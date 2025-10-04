
# California Housing Price Analysis

## Project Overview

This project analyzes the California housing dataset to explore patterns, perform feature engineering, and build machine learning models to predict house prices. It demonstrates end-to-end data science workflow including data preprocessing, EDA, model training, evaluation, and prediction output.

---

## Dataset

* **Input File:** `data/housing.csv`

* **Source:** [Kaggle – California Housing Prices](https://www.kaggle.com/datasets/camnugent/california-housing-prices)

* **Rows / Columns:** 20,640 × 10

* **Columns:**

  * `longitude`, `latitude` – Geographic coordinates
  * `housing_median_age` – Median age of houses
  * `total_rooms`, `total_bedrooms` – House details
  * `population`, `households` – Demographics
  * `median_income` – Median income
  * `median_house_value` – Target variable
  * `ocean_proximity` – Categorical feature

* **Output File:** `data/predictions.csv`

  * Contains predicted vs actual median house values from trained models

---

## Project Steps

### 1. Exploratory Data Analysis (EDA)

* Checked dataset structure and summary statistics using `head()`, `tail()`, `info()`, `describe()`
* Visualized distributions of numerical features using histograms
* Explored `ocean_proximity` distribution
* Plotted scatter plots of latitude vs longitude to visualize geographic patterns
* Colored scatter plots by `median_house_value` to observe pricing trends
* Created scatter matrix for key attributes (`median_income`, `median_house_value`, `total_rooms`, `housing_median_age`) to explore correlations

### 2. Feature Engineering

* Created `income_cat` by binning `median_income` into 5 categories
* Performed **stratified train-test split** to preserve income distribution across datasets

### 3. Data Preprocessing

* Imputed missing values with **median** for numerical features
* Encoded categorical features (`ocean_proximity`) using **One-Hot Encoding**
* Standardized numerical features using **StandardScaler**
* Built **pipelines** for numerical and categorical transformations and combined them with `ColumnTransformer`

### 4. Modeling

* Trained three machine learning models:

  * **Linear Regression**
  * **Decision Tree Regressor**
  * **Random Forest Regressor**
* Evaluated models using **RMSE**
* Performed **10-fold cross-validation** for Decision Tree to check stability

### 5. Key Findings

* `median_income` strongly correlates with `median_house_value`
* Houses near the ocean tend to have higher prices
* Random Forest performed best with the lowest RMSE on training data
* Stratified sampling ensured consistent distribution across train and test sets

---

## Tools & Libraries

* **Python Libraries:** Pandas, NumPy, Matplotlib, Scikit-Learn
* **Machine Learning Models:** Linear Regression, Decision Tree, Random Forest
* **Data Preprocessing:** StandardScaler, OneHotEncoder, SimpleImputer, ColumnTransformer
* **Visualization:** Histograms, Scatter plots, Scatter matrix

---

## How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/<your-username>/California_Housing_Analysis.git
   ```
2. Install required libraries:

   ```bash
   pip install pandas numpy matplotlib scikit-learn
   ```
3. Run either:

   * **Notebook:** `notebooks/housing_analysis.ipynb` for step-by-step exploration and visualizations
   * **Script:** `scripts/housing_analysis.py` for final model training and predictions

---

## Folder Structure

```
California_Housing_Analysis/
│
├── data/
│   ├── housing.csv           # Input dataset
│   └── predictions.csv       # Output predictions
│
├── notebooks/
│   └── housing_analysis.ipynb  # Exploratory analysis & visualizations
│
├── scripts/
│   └── housing_analysis.py     # Production-ready Python script
│
└── README.md                   # Project description and instructions
```

---

## Future Work

* Hyperparameter tuning for Random Forest to improve accuracy
* Feature selection or dimensionality reduction for faster training
* Deploy the trained model as a web app for predicting house prices

---
