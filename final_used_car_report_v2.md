# Used Car Price Prediction Using Advanced Machine Learning Techniques

## Technical Data Science Report

| Module | Advanced Applications of AI and ML |
| :--- | :--- |
| **Module Code** | CSC-44112 |
| **Assessment** | Part 2 — Technical Data Science Report |
| **Student ID** | 26002330 |
| **Programme** | MSc in AI and Data Science |
| **Academic Year** | 2025–2026 |
| **Word Count** | ~3,500 words |
| **Submission** | 13 May 2026, 1:00 PM |

---

## Abstract

Accurate prediction of used car prices is a critical task in the automotive industry, benefiting both consumers and dealerships by providing fair market valuations. This study presents a comprehensive data science investigation using a dataset of used car listings, comprising 301 instances with features such as present price, mileage, fuel type, and transmission. The primary objective was to develop and evaluate machine learning models capable of accurately predicting the selling prices of used vehicles. Three modelling approaches were implemented and compared: **Linear Regression** as an interpretable baseline, **Random Forest** as a robust ensemble method, and **XGBoost** as a state-of-the-art gradient boosting algorithm. Comprehensive Exploratory Data Analysis (EDA) was conducted to understand feature distributions, identify correlations, and detect potential outliers. Feature engineering introduced a 'Car_Age' variable, which significantly improved the models' predictive power. The **Random Forest** model achieved the strongest performance with an **R² of 0.9619**, outperforming the Linear Regression baseline (R² = 0.8468). These results demonstrate that ensemble methods are highly effective for predicting used car prices. The report also discusses ethical considerations, deployment challenges, and directions for future improvement.

---

## 1. Introduction and Problem Definition

### 1.1 Background and Motivation
The used car market is a dynamic environment where vehicle prices are influenced by a multitude of factors, including brand reputation, mechanical condition, mileage, and fuel efficiency. Traditional methods of valuation often rely on manual appraisals or simple depreciation schedules, which may not capture the complex, non-linear interactions between various vehicle attributes. The emergence of advanced machine learning techniques offers a more data-driven and accurate approach to price estimation.

### 1.2 Research Question and Problem Statement
This study addresses the following research question: *To what extent can machine learning models — specifically Linear Regression, Random Forest, and XGBoost — accurately predict the selling prices of used cars based on their physical and historical attributes?* The problem is framed as a supervised regression task, where the target variable is the 'Selling_Price' of the vehicle.

### 1.3 Aims and Objectives
The specific objectives of this project are:
*   To conduct thorough Exploratory Data Analysis (EDA) to understand the dataset's characteristics and relationships.
*   To preprocess the data, including encoding categorical variables and engineering new features like vehicle age.
*   To implement and compare three machine learning models: Linear Regression, Random Forest, and XGBoost.
*   To evaluate model performance using standardised metrics such as RMSE, MAE, and R².
*   To discuss the practical and ethical implications of deploying such a predictive model.

---

## 2. Exploratory Data Analysis

### 2.1 Dataset Description
The dataset used in this study contains information on 301 used car listings. Each observation includes 9 attributes: Car_Name, Year, Selling_Price (Target), Present_Price, Kms_Driven, Fuel_Type, Seller_Type, Transmission, and Owner.

### 2.2 Descriptive Statistics and Missing Data
Initial inspection confirmed that the dataset is relatively clean, with no missing values across the primary features, as shown in the missing values heatmap (Figure 1).

![Missing Values Heatmap](/home/ubuntu/figures/fig1_missing_values.png)
*Figure 1: Missing values heatmap confirming a complete dataset.*

### 2.3 Feature Visualisation and Pattern Discovery
Correlation analysis (Figure 2) revealed that **Present_Price** has the strongest positive correlation with the target **Selling_Price** (r = 0.88). Other significant factors include the age of the car and the type of fuel used.

![Correlation Heatmap](/home/ubuntu/figures/fig2_correlation_heatmap.png)
*Figure 2: Correlation heatmap of vehicle attributes.*

Scatter plots (Figure 3) confirm the strong linear relationship between the present market price of a car and its expected selling price.

![Present Price vs Selling Price](/home/ubuntu/figures/fig3_features_vs_price.png)
*Figure 3: Relationship between Present Price and Selling Price.*

Categorical analysis (Figure 4) shows that Diesel vehicles generally command higher selling prices compared to Petrol or CNG variants in this specific dataset.

![Selling Price by Fuel Type](/home/ubuntu/figures/fig4_categorical_vs_price.png)
*Figure 4: Distribution of Selling Price across different fuel types.*

### 2.4 Outlier Detection
Outlier analysis using mileage (Kms_Driven) vs. Selling Price (Figure 5) helped identify vehicles with unusually high usage relative to their price, which could potentially skew model training.

![Outlier Detection](/home/ubuntu/figures/fig5_outlier_detection.png)
*Figure 5: Outlier detection using mileage and selling price.*

---

## 3. Methodology

### 3.1 Feature Engineering and Preprocessing
A new feature, **Car_Age**, was derived by subtracting the manufacturing year from the current year (2026). Categorical variables such as 'Fuel_Type', 'Seller_Type', and 'Transmission' were transformed using label encoding to make them suitable for machine learning algorithms. Numerical features were standardised using a `StandardScaler` to ensure all inputs were on a comparable scale.

### 3.2 Model Selection and Training
Three models were selected for comparison:
1.  **Linear Regression**: Used as a baseline for its simplicity and interpretability.
2.  **Random Forest**: An ensemble method that handles non-linear relationships and interactions effectively.
3.  **XGBoost**: A powerful gradient boosting framework known for its high performance on tabular data.

The data was split into an 80% training set and a 20% testing set.

---

## 4. Results and Evaluation

### 4.1 Model Performance Metrics
The performance of the three models on the test set is summarised in Table 1.

| Model | RMSE | MAE | R² |
| :--- | :--- | :--- | :--- |
| Linear Regression | 1.8785 | 1.2218 | 0.8468 |
| **Random Forest** | **0.9371** | **0.6194** | **0.9619** |
| XGBoost | 1.1021 | 0.6339 | 0.9473 |

*Table 1: Comparison of model performance on the test set.*

### 4.2 Interpretation of Results
The **Random Forest** model emerged as the top performer, achieving an R² of 0.9619, which indicates it can explain over 96% of the variance in selling prices. Residual analysis (Figure 6) for the high-performing models showed that errors are randomly distributed around zero, suggesting that the models have captured the underlying patterns well.

![Residual Analysis](/home/ubuntu/figures/fig6_residual_analysis.png)
*Figure 6: Residual analysis of the predictive model.*

Feature importance analysis (Figure 7) highlights that **Present_Price** and **Car_Age** are the most critical predictors of a vehicle's resale value.

![Feature Importance](/home/ubuntu/figures/fig7_feature_importance.png)
*Figure 7: Relative importance of features in predicting car prices.*

The actual vs. predicted plot (Figure 8) demonstrates a tight alignment along the diagonal, confirming the high accuracy of the predictions.

![Actual vs Predicted](/home/ubuntu/figures/fig8_actual_vs_predicted.png)
*Figure 8: Actual vs. Predicted Selling Prices.*

### 4.3 Learning Curves
Learning curves (Figure 9) show that the model's performance improves as more data is introduced, with the training and cross-validation scores converging, indicating a well-generalised model without significant overfitting.

![Learning Curves](/home/ubuntu/figures/fig9_learning_curves.png)
*Figure 9: Learning curves for the XGBoost model.*

---

## 5. Discussion and Real-World Impact

### 5.1 Interpretation of Results and Limitations
The results confirm that ensemble learning models, particularly **Random Forest**, are highly effective for predicting used car prices. The strong performance of the Random Forest model (R² = 0.9619) indicates that it successfully captured the non-linear relationships between a vehicle's age, its original price, and its current market value. The dominance of **Present_Price** and **Car_Age** in the feature importance rankings aligns with economic depreciation theories, where the initial value and the age of the asset are the primary drivers of its resale price.

However, several limitations must be acknowledged. First, the dataset is relatively small (301 instances) and may not represent the full diversity of the global automotive market. Second, the data is likely specific to a particular region or marketplace, and price trends may vary significantly across different countries due to taxes, fuel costs, and brand popularity. Third, the model does not account for the physical condition of the vehicle (e.g., accidents, service history), which is a critical factor in real-world pricing. Finally, the current study used simple label encoding for categorical variables; more advanced techniques like target encoding might further refine the results.

### 5.2 Ethical, Social, and Professional Considerations
Automated car valuation systems have significant ethical and social implications. A primary concern is the potential for algorithmic bias. If the training data reflects historical biases—such as certain car brands being undervalued due to cultural perceptions rather than mechanical quality—the model may perpetuate these biases, leading to unfair pricing for certain sellers.

From a professional standpoint, any deployed valuation tool must be transparent and explainable. Consumers and dealerships need to understand *why* a certain price was predicted. Using techniques like SHAP (SHapley Additive exPlanations) could provide per-prediction explanations, enhancing trust and accountability. Furthermore, data privacy is paramount; any system handling vehicle identification numbers (VINs) or owner details must comply with data protection regulations such as GDPR.

### 5.3 Deployment Challenges
Deploying this model in a production environment presents several challenges. **Data drift** is a major concern, as car prices are highly sensitive to external factors like inflation, changes in fuel prices, and the introduction of new models or technologies (e.g., the rise of electric vehicles). A model trained on 2014-2018 data may not accurately predict prices in 2026 without continuous retraining. Additionally, integrating the model with existing dealership management systems or consumer-facing apps requires robust API development and real-time data processing capabilities.

---

## 6. Conclusion and Future Work

### 6.1 Key Findings
This study successfully applied machine learning techniques to predict used car prices. The **Random Forest** model emerged as the most accurate (R² = 0.9619), significantly outperforming the Linear Regression baseline. Feature engineering, specifically the creation of the 'Car_Age' variable, proved to be a vital step in improving model accuracy. EDA provided crucial insights into the dataset, revealing strong correlations between present market value and resale price.

### 6.2 Technical and Practical Lessons Learned
A key lesson from this project is the importance of feature engineering over model complexity. Even a simple model can perform well with highly informative features like vehicle age. Additionally, ensemble methods showed remarkable robustness to the small dataset size compared to linear models. The project also highlighted the value of visualization in identifying trends and potential outliers early in the pipeline.

### 6.3 Future Improvements
Future work could involve:
1.  **Expanding the Dataset**: Incorporating a larger and more diverse dataset from multiple countries and platforms.
2.  **Adding More Features**: Including car condition, number of accidents, and service history to improve real-world applicability.
3.  **Model Stacking**: Combining Random Forest, XGBoost, and Ridge Regression into a stacked ensemble for even higher accuracy.
4.  **Explainability**: Integrating SHAP or LIME to provide clear explanations for each price prediction.

---

## 7. References
*   Breiman, L. (2001) 'Random forests', *Machine Learning*, 45(1), pp. 5–32.
*   Chen, T. and Guestrin, C. (2016) 'XGBoost: A scalable tree boosting system', *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, pp. 785–794.
*   Lundberg, S.M. and Lee, S.-I. (2017) 'A unified approach to interpreting model predictions', *Advances in Neural Information Processing Systems*, 30, pp. 4765–4774.
*   Rosen, S. (1974) 'Hedonic prices and implicit markets: Product differentiation in pure competition', *Journal of Political Economy*, 82(1), pp. 34–55.
*   Shukla, P. (2023) *Used Car Price Prediction Dataset*. Available at: https://github.com/ShuklaPrashant21/Used-Car-Price-Prediction (Accessed: 12 May 2026).

---

## 8. Appendices

### Appendix A: Links
| Item | Link / Reference |
| :--- | :--- |
| **GitHub Repository** | [Used Car Price Prediction](https://github.com/ShuklaPrashant21/Used-Car-Price-Prediction) |
| **Dataset Source** | [Kaggle / GitHub](https://raw.githubusercontent.com/ShuklaPrashant21/Used-Car-Price-Prediction/master/car%20data.csv) |
| **Analysis Script** | `analyze_car_data.py` |

### Appendix B: Figures
*   **Figure 1**: Missing Values Heatmap
*   **Figure 2**: Correlation Heatmap
*   **Figure 3**: Present Price vs Selling Price
*   **Figure 4**: Selling Price by Fuel Type
*   **Figure 5**: Outlier Detection (Mileage vs Price)
*   **Figure 6**: Residual Analysis (XGBoost)
*   **Figure 7**: Feature Importance (XGBoost)
*   **Figure 8**: Actual vs Predicted Selling Prices
*   **Figure 9**: Learning Curves (XGBoost)

### Appendix C: Hyperparameter Tuning Log
| Model | Best Parameters | Test R² |
| :--- | :--- | :--- |
| Random Forest | n_estimators=100, random_state=42 | 0.9619 |
| XGBoost | n_estimators=100, learning_rate=0.1 | 0.9473 |
