import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

# Create directory for figures
os.makedirs('figures', exist_ok=True)

# 1. Load Data
df = pd.read_csv('used_car_data.csv')

# 2. EDA & Figures
# Missing Values Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.savefig('figures/fig1_missing_values.png')
plt.close()

# Feature Engineering
df['Car_Age'] = 2026 - df['Year'] # Current year is 2026 according to system prompt
# Drop Year and Car_Name (too many categories for a simple report, though original used it, we'll focus on others)
df_model = df.drop(['Car_Name', 'Year'], axis=1)

# Encode Categorical
le = LabelEncoder()
for col in ['Fuel_Type', 'Seller_Type', 'Transmission']:
    df_model[col] = le.fit_transform(df_model[col])

# Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df_model.corr(), annot=True, cmap='RdBu_r', fmt='.2f')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('figures/fig2_correlation_heatmap.png')
plt.close()

# Feature vs Selling_Price (Present_Price vs Selling_Price)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Present_Price', y='Selling_Price', hue='Fuel_Type')
plt.title('Present Price vs Selling Price')
plt.savefig('figures/fig3_features_vs_price.png')
plt.close()

# Categorical vs Selling_Price (Fuel_Type)
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Fuel_Type', y='Selling_Price')
plt.title('Selling Price by Fuel Type')
plt.savefig('figures/fig4_categorical_vs_price.png')
plt.close()

# Outlier Detection (Kms_Driven vs Selling_Price)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Kms_Driven', y='Selling_Price')
plt.title('Kms Driven vs Selling Price (Outlier Detection)')
plt.savefig('figures/fig5_outlier_detection.png')
plt.close()

# 3. Modeling
X = df_model.drop('Selling_Price', axis=1)
y = df_model['Selling_Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
}

results = []

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    results.append({
        'Model': name,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    })
    
    if name == 'XGBoost':
        # Residual Analysis
        plt.figure(figsize=(10, 6))
        sns.residplot(x=preds, y=y_test - preds, lowess=True, line_kws={'color': 'red'})
        plt.title('Residual Analysis (XGBoost)')
        plt.xlabel('Predicted')
        plt.ylabel('Residuals')
        plt.savefig('figures/fig6_residual_analysis.png')
        plt.close()
        
        # Feature Importance
        plt.figure(figsize=(10, 6))
        importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        importances.plot(kind='bar')
        plt.title('Feature Importance (XGBoost)')
        plt.savefig('figures/fig7_feature_importance.png')
        plt.close()
        
        # Actual vs Predicted
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, preds, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.title('Actual vs Predicted (XGBoost)')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.savefig('figures/fig8_actual_vs_predicted.png')
        plt.close()

# Learning Curves for XGBoost
train_sizes, train_scores, test_scores = learning_curve(
    XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42), X, y, cv=5, scoring='r2', 
    train_sizes=np.linspace(0.1, 1.0, 5)
)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Training score')
plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', label='Cross-validation score')
plt.title('Learning Curves (XGBoost)')
plt.xlabel('Training examples')
plt.ylabel('R2 Score')
plt.legend(loc='best')
plt.savefig('figures/fig9_learning_curves.png')
plt.close()

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv('model_results.csv', index=False)

print("Analysis complete. Figures and results saved.")
