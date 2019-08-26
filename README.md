# House Prices: Advanced Regression Techniques

[Kaggle challenge link](https://www.kaggle.com/c/data)

With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this Kaggle competition challenges to predict the final price of each home.

![alt text](https://olegleyz.github.io/images/header.jpg)

## Files

* ***Final_FDS_Kaggle.ipynb*** : Jupyter notebook which shows all the steps performed to tackle the challenge

## Report

* **Data tidying**:
    + Removed outliers with ‘*GrLivArea*’>4000 and ‘*SalePrice*’<300000.
    + Normalization:
    
        + Target variable ('*Sale Price*') with Log Transformation
        + Box-Cox transformation (α = 0.15) to variables with skewness > 0.75.
        + Fill missing values. Depending on the feature, the missing value has been substituted by:
            * '*None*'
            * 0
            * Median of the feature
            * Mode of the feature

* **Feature engineering**:
    + Make 'object' type categorical variables which were set as numerical.
    + Label encoding of categorical variables.
    + Create dummy variables for categorical variables.
    + Drop variable '*Utilities*'.
    + Drop variable ID for the training process.
    + Create a variable '*TotalSF*' as result of ‘*Total BsmtSF*’ + ’*1stFlrSf*’ + ’*2ndFlrSF*’ that encompasses the total surface of a house.

* **Modelization**:
    + Lasso Regression + Robust Scaler.
    + Elastic Net Regression + Robust Scaler.
    + Kernel Ridge Regression.
    + Gradient Boosting Regression + “Huber” Loss.
    + XGBoost Regression.
    + LightGBM Regression.
    + Stacked regression: GBoost as meta-regressor, all the other models as regressors.
    
* **Training**:
    + Average of 5-folds cross validation
    + Final model:
        - Stacked regression + other models, with the following weights:
        
        stack*0.14 + xgb*0.14 + lgb*0.14 + gboost*0.16 + krr*0.14 + lasso*0.14 + e_net*0.14
