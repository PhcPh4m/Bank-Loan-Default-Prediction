# Bank Loan Default Prediction (Big Data Edition)

Predicting bank loan defaults using the LendingClub dataset. This project evolved from a local-scale KNN model to a Scalable Big Data Pipeline using Apache Spark and Hadoop HDFS, handling over 2.2 million records with advanced Class Imbalance Handling.

---

## Project Evolution: From Local to Big Data
Initially, a KNN model on Pandas could only handle 8k rows. To process the full **1.55GB dataset**, I built a distributed architecture:
* **Storage:** Hadoop Distributed File System (HDFS) for reliable storage.
* **Processing Engine:** Apache Spark (PySpark) for parallelized data transformation.
* **Model:** Random Forest Classifier (Optimized for large-scale financial data).
* **Problem Solving**: Applied Undersampling to resolve the 80/20 class imbalance, transforming the model from "guessing" to "accurately detecting" risky loans.

## Dataset & Infrastructure
* **Source:** [LendingClub Dataset on Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
* **Raw Scale:** 2,260,668 records (~1.55GB CSV file).
* **Infrastructure:** * **HDFS:** Managed distributed data ingestion.
    * **PySpark:** Parallelized the training process across the entire cluster.
* **Balanced Dataset**: After cleaning and Undersampling, the model trains on ~537k records (50/50 ratio of Good vs. Bad loans) to ensure unbiased predictions.

## Why KNN and Random Forest?
I selected these two algorithms to demonstrate the transition from a prototype baseline to a scalable production model.
1. **K-Nearest Neighbors (KNN) - The Baseline**
   * Purpose: Used in the initial phase to establish a baseline.
   * Limitation: KNN is an instance-based learner with $O(n^2)$ complexity. It is computationally impossible to scale to 2.2 million rows on a single machine.
   * Lesson Learned: This scalability bottleneck was the primary driver for migrating to Apache Spark.
2. **Random Forest - The Scalable Solution**
   * Distributed Power: Naturally parallelizable. Spark trains different trees on different executors simultaneously.
   * Handling Non-Linearity: Lending data has complex interactions (e.g., Interest Rate vs. Credit Grade) that Random Forest captures effectively.

## Technical Stack
* **Big Data:** Apache Spark 3.1.2, Hadoop 3.2.
* **Machine Learning:** `pyspark.ml` (VectorAssembler, RandomForestClassifier, StringIndexer).
* **Visualization:** `Seaborn`, `Matplotlib`.
* **Environment:** Google Colab with Java 8 & HDFS configured.

## Key Insights & Visualizations (EDA)
Spark allowed  to capture the full variance of the data, revealing critical "red flags":

### 1. The Imbalance Challenge & Interest Rate "Red Flag"
* The dataset was heavily skewed toward "Fully Paid" loans. EDA confirmed that "Charged Off" (Default) loans carry significantly higher interest rates.

<p align="center">
<img src="images/loan_status_dist.png" width="400" title="Loan Status Distribution">
<img src="images/interest_rate_boxplot.png" width="450" title="Interest Rate Boxplot">
</p>

### 2. Feature Correlations
Spark's MLlib calculated correlations across 2.2M rows, identifying Interest Rate and Grade as the most influential features.

<p align="center">
<img src="images/correlation_heatmap.png" width="500" title="Correlation Heatmap">
</p>

---

## Model Performance: Spark Random Forest
By transitioning to a Balanced Random Forest (100 Trees), the model now actively identifies defaults instead of just guessing the majority class.

### **Final Metrics:**
*  Accuracy: 64.05% (Realistic performance on balanced 50/50 data).
*  AUC (Area Under ROC): 0.6952 (Strong predictive power for financial risk).
*  Default Detection: Successfully captured 37,455 actual default cases in the test set.

### **Evaluation & Importance:**
The model confirms Interest Rate as the primary predictor of default risk.

<p align="center">
<img src="images/feature_importance.png" width="450" title="Feature Importance">
<img src="images/confusion_matrix.png" width="450" title="Confusion Matrix">
</p>
<p align="center">
<img src="images/ROC_curve.png" width="500" title="ROC Curve">
</p>

---

## Data-Driven Insights & Business Recommendations
After analyzing 2.2 million records and evaluating the model, I have identified several key insights for credit risk management:
1. **The "Interest Rate" Risk Threshold**
   * Insight: Our Interest Rate Boxplot and Feature Importance confirm that Interest Rate is the #1 predictor of default. Loans that eventually "Charged Off" typically have a median interest rate significantly higher than "Fully Paid" loans.
   * Recommendation: The bank should implement a stricter manual review process for any loan application where the calculated interest rate exceeds 15%, as this group falls into the high-risk "Default Zone".
2. **Credit Grade vs. Default Probability**
   * Insight: The Correlation Heatmap shows a strong relationship between credit grade and loan status. As the grade drops (from A to G), the correlation with "Charged Off" status increases.
   * Recommendation: Automate a "Red Flag" system for applicants with Grades E, F, and G, regardless of their income level, as historical data shows these grades are highly prone to default under economic stress.
3. **Model Reliability in Production**
   * Insight: By achieving an AUC of 0.70 on a balanced dataset, the model demonstrates a solid ability to distinguish between "Good" and "Bad" borrowers, far outperforming random chance or biased baseline models.
   * Recommendation: This Spark-based pipeline is ready for Production Integration. It can be deployed within a Loan Origination System (LOS) to provide real-time risk scoring, allowing for proactive rejection of high-risk profiles and potentially saving millions in unrecoverable debt.
   
## Big Data Optimization
Processing 2.2M rows is computationally expensive. I optimized the Spark pipeline to run the entire notebook in ~18 minutes:
* Strategic Caching: Used .cache() to store intermediate results, preventing redundant full-data scans.
* Training Efficiency: Configured subsamplingRate=0.7 for the Random Forest, reducing training time by 75% while maintaining model stability.
* Parallel Execution: Leveraged Spark's distributed architecture to handle 160x more data than the legacy KNN version

## Model Comparison: KNN vs. Spark Random Forest

| Feature | Legacy KNN Model | Spark Random Forest (Final) |
| :--- | :--- | :--- |
| **Data Volume** | Sampled (**8,344 rows**) | **2.2M rows (Full )** |
| **Infrastructure** | Local CPU / Pandas | **Hadoop HDFS & Apache Spark** |
| **Class Handling** | None (Biased) | **Undersampling (Balanced 50/50)** |
|**Complexity** | $O(n^2)$ - Not scalable | $O(Trees \times n \log n)$ - **Distributed** |
| **Accuracy** | 82.00% (Small sample) | **64.05% (On balanced data)** |
| **Scalability** | Low (Memory limited) | **High (Production-ready)** |

---

## Repository Structure
* `notebook/`: 
    * `Bank_Loan_Classification_Large_Scale_System.ipynb` (Final Spark Pipeline).
    * `Legacy_KNN_Model.ipynb` (Initial approach on 8k rows).
* `images/`: Visualization charts for EDA and Model Evaluation.
* `README.md`: Project documentation.

## How to Run
1. Navigate to Notebooks: Go to the notebook/ folder in this repository.
2. Open the Core Pipeline: Click on Bank_Loan_Classification_Large_Scale_System.ipynb.
3. Launch in Colab: Click the "Open in Colab" .
4. Execute the Pipeline
       * Go to the menu Runtime > Run all (or press Ctrl + F9).
       * No Kaggle API Required: The system automatically fetches the 1.55GB dataset from a secure public mirror to bypass authentication hurdles.
5. Automated Workflow:
   * Environment Setup: The notebook silently installs Java 8, Spark 3.1.2, and Hadoop, then configures a virtual HDFS environment.
   * Data Ingestion: The 2.2M raw records are downloaded and ingested into HDFS at /user/Bank-Loan-Default-Prediction/data/loan_data.csv.
   * Distributed Processing: PySpark parallelizes the cleaning and training process across the virtual cluster.
6. After ~18 minutes (depending on Colab's network speed), the pipeline will complete. You will witness a model trained on a balanced dataset of ~537k rows (achieved through Undersampling).
---
**Author:** Phuc Pham  
