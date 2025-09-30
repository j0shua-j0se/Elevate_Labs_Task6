# Task 6 — K-Nearest Neighbors (KNN) Classification

This repository implements a complete KNN classification workflow with feature normalization, K selection, accuracy/confusion-matrix evaluation, and decision boundary visualization.

## Objective

Understand and implement KNN for classification by:
- Normalizing features and preparing a leak-free pipeline.
- Training and evaluating KNeighborsClassifier with different K values.
- Comparing accuracy across K and visualizing decision regions.

## Dataset

- Source file (local): C:\Users\OMEN\.cache\kagglehub\datasets\uciml\iris\versions\2\Iris.csv  
- Schema: typically includes [Id, SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm, Species].
- Target: Species (multi-class: setosa, versicolor, virginica).
- Note: Id is dropped if present; Species is label-encoded to numeric classes for modeling.

## Environment

- Python 3.8+
- Libraries: pandas, numpy, scikit-learn, seaborn, matplotlib

Install:
```bash
pip install pandas numpy scikit-learn seaborn matplotlib
```

## How to run

Option A: Notebook
- Open the notebook.
- Ensure INPUT_CSV is set to C:\Users\OMEN\.cache\kagglehub\datasets\uciml\iris\versions\2\Iris.csv.
- Run cells top-to-bottom to train KNN, sweep K, evaluate accuracy, and generate decision boundary plots.

Option B: Script
- Place the provided script in src/.
- Edit the INPUT_CSV constant at the top of the file to the path above.
- Run:
```bash
python src/task6_knn.py
```

## Workflow

### Data loading and target encoding
- Load Iris.csv and drop Id if present.
- Label-encode Species into numeric classes to train KNN.

### Preprocessing
- Numeric features: median imputation and scaling (StandardScaler or MinMaxScaler).
- Categorical (if any): most_frequent imputation and one-hot encoding.
- Implemented via a ColumnTransformer wrapped in a Pipeline to avoid leakage.

### Train/test split
- Stratified 80/20 split to preserve class balance and ensure a fair evaluation.

### Baseline KNN
- Train a baseline KNeighborsClassifier (e.g., K=5, weights="distance").
- Report accuracy and plot a confusion matrix for the test split.

### K sweep and selection
- Sweep odd K values (e.g., 1 to 29) and plot test accuracy vs K.
- Select best K by accuracy stability and interpretability.

### Decision boundary visualization
- Plot decision boundaries using the 2D feature pair PetalLengthCm and PetalWidthCm (if both exist).
- If 2D pair not available, reduce to 2D via PCA and visualize decision regions in the projected space.

### Optional: distance metrics and CV
- Compare Euclidean (p=2) vs Manhattan (p=1) via Minkowski metric in KNN.
- Run 5-fold Stratified cross-validation to report mean ± std accuracy.

## Outputs

- Accuracy score and confusion matrix for the chosen K.
- K sweep plot: test accuracy vs K.
- Decision boundary plot (either on PetalLength/PetalWidth or PCA projection).
- Optional: CV accuracy summary and metric comparisons.

## Repository structure

```
.
├── notebooks/
│   └── task6_knn.ipynb
├── src/
│   └── task6_knn.py
├── data/
│   └── Iris.csv  (or reference to the KaggleHub path)
├── outputs/
│   ├── confusion_matrix_kNN.png
│   ├── k_sweep_accuracy.png
│   └── decision_boundary.png
└── README.md
```

## Configuration

- INPUT_CSV: set to C:\Users\OMEN\.cache\kagglehub\datasets\uciml\iris\versions\2\Iris.csv.
- TEST_SIZE, RANDOM_STATE: adjust for reproducibility and split control.
- KNeighborsClassifier hyperparameters: n_neighbors, weights, metric, p.

## Why normalization matters in KNN

KNN is distance-based; unscaled features can dominate distance computations, skewing nearest neighbor selection. Scaling (StandardScaler/MinMaxScaler) puts features on comparable ranges, improving performance and stability.

## Interview-ready notes

- How KNN works: predicts a class by majority vote among the k nearest neighbors according to a distance metric in feature space.
- Choosing K: small K can overfit (high variance); larger K smooths boundaries (higher bias). Use validation/K sweep to pick a stable K.
- Normalization: essential to ensure fair contribution of each feature in distance calculations.
- Complexity: naive query time is O(N·D) per prediction; tree/graph indices can help in low dimensions.
- Pros/cons: simple, non-parametric, effective with well-scaled features; slower at inference and sensitive to irrelevant/noisy features.
- Multi-class: handled natively by majority voting across classes.
- Distance metrics: Euclidean (p=2) and Manhattan (p=1) via Minkowski; the choice affects neighbor geometry and decision boundaries.

## Submission checklist

- Code runs end-to-end and generates the confusion matrix, K sweep plot, and decision boundary figure.
- README.md present at repository root.
- Dataset path configured correctly.
- Optional: CV scores and metric comparisons included.

## License and acknowledgments

- Educational use for internship Task 6.
- Dataset: Iris (local CSV from KaggleHub cache path as configured).

<img width="487" height="453" alt="image" src="https://github.com/user-attachments/assets/741135c4-1e7b-4cdb-9259-169656b02393" />
<img width="739" height="469" alt="image" src="https://github.com/user-attachments/assets/8662c45e-0b04-4834-b039-95ede9c1ca9a" />
<img width="729" height="414" alt="image" src="https://github.com/user-attachments/assets/3c6d3543-6b6b-4512-865c-1a31719ccfb6" />
<img width="413" height="77" alt="image" src="https://github.com/user-attachments/assets/4a04dca6-381e-4159-af87-c31f2fa7bf9d" />
