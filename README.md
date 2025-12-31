# Titanic Survival Predictor (Logistic Regression Analysis)

### Project Overview
This project is an end-to-end machine learning pipeline designed to predict passenger survival based on the 1912 Titanic manifest data. This version utilizes **Logistic Regression** to model the probability of survival as a linear function of passenger features.

### Technical Architecture
* **Backend:** FastAPI
* **Frontend:** Tailwind CSS / Space Grotesk Typography
* **Model:** Logistic Regression (Scikit-Learn)
* **Deployment:** Render

### Model Performance
* **Accuracy Score:** 79.89%
* **Methodology:** Binary classification using the Sigmoid function to map feature weights to a 0-1 probability range.

### Data Engineering
1.  **Imputation:** Missing age values filled using mean imputation.
2.  **Encoding:** Categorical variables mapped to numerical integers.
3.  **Deployment:** Serialization via `joblib` for high-performance inference.
