# 📊 Sales Prediction Using Machine Learning

This project explores how advertising expenditure impacts sales by building machine learning models to predict future sales. Through different regression techniques and hyperparameter tuning, the project aims to identify the best model to accurately forecast sales performance based on advertising data.

## 📖 Project Summary

The project utilizes the following machine learning techniques to model and predict sales:
- **Linear Regression**: A baseline model to establish a linear relationship between advertising spend and sales.
- **Lasso Regression**: An enhanced model with L1 regularization to prevent overfitting by reducing the impact of irrelevant features.
- **Grid Search with Cross-Validation**: A method for optimizing model hyperparameters to improve prediction accuracy.

## 📊 Dataset

The dataset consists of advertising spend across different media channels (TV, radio, and newspaper) and the corresponding sales figures. This data is stored in the `Advertising.csv` file.

## 🚀 Getting Started

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-repo/SalesPrediction.git
   cd SalesPrediction
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🛠️ How to Use

1. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook SalesPrediction.ipynb
   ```

2. **Run the notebook**: Execute all cells to train and evaluate the models.

## 🧩 Machine Learning Models

- **Linear Regression**: A simple model to predict sales using the linear relationship between advertising expenditures and sales.
- **Lasso Regression**: Adds L1 regularization to penalize large coefficients, helping the model generalize better and reduce overfitting.

## ⚙️ Hyperparameter Optimization

The **GridSearchCV** method is used to fine-tune model parameters for the best performance. Key hyperparameters tuned include:
- `alpha` (for Lasso Regression)
- `fit_intercept`
- `max_iter`

## 📈 Model Performance

The models are evaluated using the R-squared score to measure the goodness of fit. After tuning, the Lasso Regression model outperforms Linear Regression, achieving an R-squared score of approximately **0.92**, indicating strong predictive accuracy.

### Performance Overview:
- **Linear Regression R² Score**: 0.89
- **Lasso Regression R² Score**: 0.92

## 🧪 Libraries Used

- **`scikit-learn`**: For model building and hyperparameter tuning.
- **`pandas`**: For data processing and manipulation.
- **`numpy`**: For numerical operations.

## 🔍 Conclusion

This project demonstrates how advertising spend can be used to predict sales through machine learning models. Lasso Regression, with hyperparameter tuning, provided the best results. Future enhancements could include:
- Testing additional models, such as Ridge Regression.
- Applying feature engineering to further improve model accuracy.

