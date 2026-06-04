# Task 1: Data Cleaning & Preprocessing
### AI & ML Internship — ElevateLabs

---

## Objective
Clean and prepare the raw Titanic dataset for Machine Learning by handling missing values, encoding categorical features, detecting/removing outliers, and scaling numerical features.

---

## Tools & Libraries
| Tool | Purpose |
|------|---------|
| Python 3 | Core language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib | Plotting |
| Seaborn | Statistical visualizations |
| Scikit-learn | Encoding & Scaling |

---

## Dataset
**Titanic Dataset** — 891 rows × 12 columns  
Source: [Kaggle / datasciencedojo GitHub](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv)

---

## Steps Performed

### 1. Load & Explore
- Loaded CSV, checked shape, dtypes, and missing values
- Found 3 columns with nulls: `Age` (19.9%), `Cabin` (77.1%), `Embarked` (0.2%)

### 2. Handle Missing Values
| Column | Strategy | Reason |
|--------|----------|--------|
| `Age` | Filled with **median** (28.0) | Median is robust to outliers |
| `Embarked` | Filled with **mode** ('S') | Categorical, only 2 nulls |
| `Cabin` | **Dropped** entirely | 77% missing — not recoverable |

### 3. Drop Irrelevant Columns
Dropped `PassengerId`, `Name`, `Ticket` — these are identifiers with no predictive value.

### 4. Encode Categorical Features
- **`Sex`** → Label Encoding (female=0, male=1) — binary, so label encoding is safe
- **`Embarked`** → One-Hot Encoding (C as reference; added `Embarked_Q`, `Embarked_S`) — avoids ordinal assumptions

### 5. Outlier Detection & Removal
- Visualized with **boxplots** (before & after)
- Used **IQR method** (1.5 × IQR fence) on `Fare` and `Age`
- Removed 95 Fare outliers + 15 Age outliers → 602 clean rows remain

### 6. Feature Scaling
- **StandardScaler** (Z-score): mean=0, std=1 — best for algorithms like SVM, Logistic Regression
- **MinMaxScaler** (Normalization): range [0,1] — best for neural networks, KNN

---

## Output Files
| File | Description |
|------|-------------|
| `titanic_preprocessing.py` | Main Python script |
| `titanic_cleaned.csv` | Cleaned & preprocessed dataset |
| `boxplots_before.png` | Outlier visualization before removal |
| `boxplots_after.png` | Outlier visualization after removal |
| `correlation_heatmap.png` | Feature correlation matrix |

---

## Interview Questions — Answers

**1. What are the different types of missing data?**  
- **MCAR** (Missing Completely At Random): No pattern, e.g., sensor glitch  
- **MAR** (Missing At Random): Missingness depends on other observed data  
- **MNAR** (Missing Not At Random): Missingness depends on the missing value itself (e.g., high earners skip income field)

**2. How do you handle categorical variables?**  
Label Encoding for binary/ordinal features; One-Hot Encoding for nominal features with no order.

**3. Normalization vs Standardization?**  
- **Normalization** (Min-Max): scales to [0,1]; sensitive to outliers; good for neural networks  
- **Standardization** (Z-score): zero mean, unit variance; robust to outliers; good for SVM, linear models

**4. How do you detect outliers?**  
IQR method, Z-score, boxplots, scatter plots, or isolation forests.

**5. Why is preprocessing important in ML?**  
Raw data is messy. Models assume clean, numeric, properly scaled input. Bad data → bad model (Garbage In, Garbage Out).

**6. One-Hot Encoding vs Label Encoding?**  
- **Label Encoding**: assigns integer ranks (0,1,2…) — can imply false ordinal relationships  
- **One-Hot Encoding**: creates binary columns — safe for nominal (unordered) categories

**7. How do you handle data imbalance?**  
Oversampling (SMOTE), undersampling, class weights in the model, or generating synthetic data.

**8. Can preprocessing affect model accuracy?**  
Yes, significantly. Proper scaling, encoding, and outlier removal can improve accuracy by 10–30% depending on the algorithm.

---

## Results
- Started with 891 rows, 12 columns  
- Ended with 602 rows, 9 clean features ready for ML  
