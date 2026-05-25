# Feature Engineering: Custom Transformations via FunctionTransformer
[![Machine Learning](https://img.shields.io/badge/Domain-Feature%20Engineering-blue)](https://scikit-learn.org/)
[![Preprocessing](https://img.shields.io/badge/Transformer-FunctionTransformer-orange)](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html)
[![Dataset](https://img.shields.io/badge/Dataset-Titanic%20Survival-green)](./Titanic-Dataset.csv)

## 🏗️ Project Overview
Many machine learning algorithms—especially parametric models like Linear Regression, Logistic Regression, and Artificial Neural Networks—assume that numeric features are normally distributed. In real-world data like the **Titanic Dataset**, features such as `Fare` display extreme right-skewness, while others like `Age` may require custom mathematical boundaries. Leaving these distributions unscaled forces estimators to contend with high-variance bias and compressed data structures.

This repository demonstrates how to build and inject custom stateless mathematical transformations into production pipelines using Scikit-Learn's **`FunctionTransformer`**. Rather than manually modifying columns using standalone Pandas operations, this project wraps custom Python logic and NumPy mathematical functions into stateful, pipeline-compliant objects that ensure consistent data manipulation across training and inference data boundaries.

---

## 🛠️ Advanced Engineering Mechanics

### 1. Mathematical Feature Reshaping
When a dataset contains heavily skewed distributions, applying non-linear mathematical operations pulls extreme outliers closer to the mean, stabilizing variance and stabilizing gradient descent convergence. This notebook explores core functional transformations:
* **Logarithmic Transform ($x \to \ln(x + 1)$):** Perfect for resolving highly right-skewed variables (e.g., ticket `Fare`). It expands tightly packed lower-range values and compresses massive outlier scales.
* **Reciprocal Transform ($x \to 1/x$):** Flips values such that larger numbers become infinitely small, restructuring ratio behaviors.
* **Power/Square Root Transforms ($x \to \sqrt{x}$):** Applies a moderate variance-stabilizing compression, ideal for moderately skewed count fields.

### 2. The Operational Flow
The `FunctionTransformer` works by converting an arbitrary python callable or numpy mathematical function into a Scikit-Learn transformer object. This allows it to natively expose `.fit()`, `.transform()`, and `.fit_transform()` methods, granting it seamless entry into pipeline orchestrations.


                         ┌──> [Fare] ──> np.log1p ───> FunctionTransformer  -──┐
                         │                                                     │
Raw Skewed Features ─────┼──> [Age] ───> np.sqrt ────> FunctionTransformer   ──┼──> Balanced Training Matrix
                         │                                                     │
                         └──> [Custom] ──> Python Fnc ─> FunctionTransformer ──┘


---

## 🔬 Implementation Steps

The implementation in `FunctionTransformer-Using-ML.ipynb` follows a modular workflow:

1.  **Exploratory Distribution Check:** Plotting feature distributions utilizing Seaborn distribution grids and QQ-plots to verify skewness coefficients before running operations.
2.  **Custom Function Mapping:** Writing clean Python callables and binding them directly with NumPy operations (`np.log1p`, `np.sqrt`).
3.  **Transformer Packaging:** Wrapping functions inside `FunctionTransformer()` to create scikit-learn compatible object parameters.
4.  **Distribution Audit Pipeline:** Plotting **Before vs. After** visualization comparisons to confirm that features have transitioned to an approximately normal, Gaussian distribution profile.
5.  **Pipeline Integration Ready:** Validating that the serialized objects maintain structural compatibility inside automated `ColumnTransformer` frameworks.

---

## 📊 Comparison Matrix: Core Mathematical Transformations

| Transformation Type | Mathematical Formula | Primary Target Use-Case | Impact on Skewed Data |
| :--- | :--- | :--- | :--- |
| **Log Transform** | $y = \log(x + 1)$ | Extreme Right Skew (e.g., `Fare`) | Maximally compresses trailing positive variance |
| **Square Root** | $y = \sqrt{x}$ | Moderate Skew (e.g., Age/Counts) | Smoothly shifts distributions toward normality |
| **Reciprocal** | $y = 1 / (x + c)$ | Ratio Rates / Non-Zero Continuities | Inverts magnitude hierarchies completely |

---

## 💻 Tech Stack & Dependencies
* **Language Environment:** Python 3.9+
* **Data Structures & Math Stack:** NumPy, Pandas
* **Pipeline & Transformation Layer:** Scikit-Learn (`preprocessing.FunctionTransformer`)
* **Statistical Plotting Engine:** Matplotlib, Seaborn, SciPy (for QQ-plots)
* **Development Workspace:** Jupyter Notebook

---

## 🚀 Getting Started

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/Function-Transformer-Using-ML.git](https://github.com/your-username/Function-Transformer-Using-ML.git)
   cd Function-Transformer-Using-ML

2. **Install Production Core Dependencies:**
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn scipy jupyter

3. **Run the Analysis Pipeline:**
   ```bash
   jupyter notebook

Open Function-Transformer-Using-ML.ipynb to step through the custom mathematical mappings and inspect the distribution shifts.   
