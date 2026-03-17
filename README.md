# Bank-Loan-Default-Prediction
Predicting bank loan defaults using the LendingClub dataset. Features comprehensive EDA, data preprocessing, and K-Nearest Neighbors (KNN) classification with hyperparameter tuning.
---

## Project Overview
The goal of this project is to build a machine learning model that can predict whether a borrower will pay back their loan in full or "Charge Off" (default). By identifying high-risk borrowers, financial institutions can minimize financial losses and optimize their lending strategies.

## Data Source
The dataset used is a subset of the **LendingClub Loan Data** from Kaggle.
* **Original Source:** [LendingClub Dataset on Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
* **Size:** 390,000+ records.
* **Primary Dataset:** `accepted_2007_to_2018Q4.csv` (Used for model training and testing).
  **Secondary Dataset:** `rejected_2007_to_2018Q4.csv` (Contains data on rejected loan applications).
* **Target Variable:** `loan_status` (Fully Paid vs. Charged Off).

> **Note:** Due to the large file size (>100MB), the raw dataset is not stored in this repository. Please download it from the link above and place it in the `data/` folder for local execution.

## Technical Stack
* **Language:** Python
* **Environment:** Google Colab / Jupyter Notebook
* **Libraries:** `Pandas`, `NumPy`, `Matplotlib`, `Seaborn`, `Scikit-learn`.

## Key Features & Workflow
1.  **Exploratory Data Analysis (EDA):** * Visualizing the distribution of loan status.
    * Analyzing correlations between features (Income, Loan Amount, DTI, etc.).
    * Handling missing values and identifying outliers.
2.  **Data Preprocessing:**
    * Feature Engineering (converting categorical strings to numeric).
    * Scaling features using `StandardScaler`.
    * Handling imbalanced data.
3.  **Modeling:**
    * Implementing **K-Nearest Neighbors (KNN)**.
    * Fine-tuning hyperparameters (k-value, distance metrics) using `GridSearchCV`.
4.  **Evaluation:**
    * Assessing performance using Confusion Matrix, Precision, Recall, and F1-score.

## Results
*(Updating as the project progresses...)*
- Found optimal `k` value for the KNN model.
- Achieved an accuracy of X% and F1-score of Y%.

## How to Run
1. Clone the repository:
   ```bash
   git clone [https://github.com/PhcPh4m/Bank-Loan-Default-Prediction.git](https://github.com/your-username/Bank-Loan-Default-Prediction.git)
2. Install dependencies: 
   ```bash
   pip install -r requirements.txt
3. Run the notebook in notebooks or upload it to Google Colab.
   
   
