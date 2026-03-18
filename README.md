# Bank Loan Default Prediction
Predicting bank loan defaults using the LendingClub dataset. Features comprehensive EDA, big data preprocessing, and K-Nearest Neighbors (KNN) classification with hyperparameter tuning.

---

## Project Overview
The goal of this project is to build a machine learning model that can predict whether a borrower will pay back their loan in full or "Charge Off" (default). By identifying high-risk borrowers, financial institutions can minimize financial losses and optimize their lending strategies.

## Data Source & Big Data Handling
The dataset used is a subset of the **LendingClub Loan Data** from Kaggle.
* **Original Source:** [LendingClub Dataset on Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
* **Dataset Size:** 2.2 million+ records (~1.55GB CSV file).
* **Big Data Optimization:** Due to hardware constraints (RAM limits), I implemented **Linux `!head` command** and **Pandas Chunking** to efficiently sample and process the data without crashing the environment.

## Technical Stack
* **Language:** Python
* **Libraries:** `Pandas`, `NumPy`, `Matplotlib`, `Seaborn`, `Scikit-learn`.
* **Tools:** Google Colab, Linux Command Line.

## Key Features & Workflow
### 1. Exploratory Data Analysis (EDA)
* Analyzed the distribution of `loan_status` to identify class imbalance.
* Visualized correlations between features like `annual_inc`, `loan_amnt`, and `int_rate`.

### 2. Advanced Data Preprocessing
* **Outlier Removal:** Used the **Interquartile Range (IQR)** method to filter extreme values in income and debt-to-income (DTI) ratios, ensuring the KNN model isn't skewed.
* **Feature Engineering:** * Converted `term` (e.g., "36 months") into numeric format.
    * Encoded categorical `grade` (A-G) into ordinal numeric values.
* **Scaling:** Applied **StandardScaler** to normalize features, which is mandatory for distance-based algorithms like KNN.

### 3. Modeling & Hyperparameter Tuning
* Implemented **K-Nearest Neighbors (KNN)**.
* Performed **Hyperparameter Tuning** using the **Elbow Method** (testing K from 1 to 20) to find the optimal balance between bias and variance.

## Results
* **Optimal K-Value:** **17** (found via Error Rate visualization).
* **Final Accuracy:** **82%** (Improved from 79% after tuning).
* **Conclusion:** The model performs exceptionally well at identifying "Fully Paid" loans (High Recall). The results reflect real-world credit risk challenges where data is naturally imbalanced.
## How to Run
1. Clone the repository:
   ```bash
   git clone [https://github.com/PhcPh4m/Bank-Loan-Default-Prediction.git](https://github.com/your-username/Bank-Loan-Default-Prediction.git)
2. Install dependencies: 
   ```bash
   pip install -r requirements.txt
3. Run the notebook in notebooks or upload it to Google Colab.
   
   
