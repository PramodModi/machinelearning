# ML Algorithm Cheat Sheet
> Comprehensive reference with Python code (scikit-learn + PyTorch), loss functions, hyperparameters, Adam optimizer, industry usage, and algorithm comparison guides.

---

## Table of Contents
1. [Linear Regression](#1-linear-regression)
2. [Logistic Regression](#2-logistic-regression)
3. [Decision Tree](#3-decision-tree)
4. [Support Vector Machine (SVM)](#4-support-vector-machine-svm)
5. [K-Nearest Neighbors (KNN)](#5-k-nearest-neighbors-knn)
6. [Naive Bayes](#6-naive-bayes)
7. [K-Means Clustering](#7-k-means-clustering)
8. [DBSCAN](#8-dbscan)
9. [PCA](#9-pca-principal-component-analysis)
10. [Random Forest](#10-random-forest)
11. [Gradient Boosting](#11-gradient-boosting)
12. [XGBoost / LightGBM](#12-xgboost--lightgbm)
13. [Neural Network (MLP)](#13-neural-network-mlp)
14. [CNN](#14-convolutional-neural-network-cnn)
15. [RNN / LSTM](#15-rnn--lstm)
16. [Transformer](#16-transformer)
17. [Algorithm Comparison Guides](#17-algorithm-comparison-guides)

---

## 1. Linear Regression

### Overview
| Property | Detail |
|---|---|
| **Type** | Regression (continuous output) |
| **Classification** | Not applicable — predicts continuous values |
| **Core Idea** | Fit a hyperplane `ŷ = Xβ` by minimizing squared error |

### Use Cases & Why
| Use Case | Why Linear Regression |
|---|---|
| House price prediction | Target is continuous; strong linear correlation with area, rooms, location |
| Sales forecasting | Temporal trends are often linear over short horizons |
| Risk scoring in insurance | Actuarial factors relate linearly to claim amounts |
| Energy consumption forecasting | Load is linearly correlated with temperature, time-of-day features |

### Loss Function
- **MSE:** `L = (1/n) Σ(yᵢ - ŷᵢ)²`
- **MAE:** `L = (1/n) Σ|yᵢ - ŷᵢ|` — robust to outliers
- **Huber:** Combines MSE and MAE — quadratic for small errors, linear for large

### Optimization
- **OLS (closed-form):** `β = (XᵀX)⁻¹Xᵀy`
- **Gradient Descent:** `βₜ₊₁ = βₜ - η·∇L`
- **Adam:** Used when linear regression is a layer inside a neural network

### Hyperparameters
| Hyperparameter | Typical Range | Effect |
|---|---|---|
| Regularization λ (L1/L2) | 1e-4 – 10 | L1=Lasso (sparse), L2=Ridge (shrink) |
| Learning rate η | 0.001 – 0.1 | Gradient descent convergence speed |
| Max iterations | 100 – 10,000 | Convergence threshold |

### Pros & Cons
**Pros**
- Fast, interpretable coefficients
- No hyperparameter tuning for OLS
- Works well when true relationship is linear

**Cons**
- Assumes linearity — fails on complex patterns
- Sensitive to outliers (MSE amplifies them)
- Requires feature engineering for non-linear relationships

### Industry Usage
- **Finance:** Credit risk models, loan pricing (Basel III interpretability)
- **Retail:** Demand forecasting (Walmart, Amazon supply chain baseline)
- **Healthcare:** Dosage-response modeling
- **Real estate:** Zillow AVM as a baseline layer

### Optimization Tips
- Always `StandardScaler` before gradient descent
- Use Ridge (L2) when features correlate; Lasso (L1) for feature selection
- Plot residuals vs. fitted to diagnose non-linearity
- Polynomial features for curvature without switching models

### Python Code

```python
# ── scikit-learn ────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing

# Load dataset
data = fetch_california_housing()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# Plain OLS
ols = LinearRegression()
ols.fit(X_train_s, y_train)
y_pred = ols.predict(X_test_s)
print(f"OLS   RMSE={mean_squared_error(y_test,y_pred)**0.5:.4f}  R²={r2_score(y_test,y_pred):.4f}")

# Ridge (L2 regularization)
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_s, y_train)
y_pred_r = ridge.predict(X_test_s)
print(f"Ridge RMSE={mean_squared_error(y_test,y_pred_r)**0.5:.4f}  R²={r2_score(y_test,y_pred_r):.4f}")

# Lasso (L1 regularization – sparse coefficients)
lasso = Lasso(alpha=0.01)
lasso.fit(X_train_s, y_train)
y_pred_l = lasso.predict(X_test_s)
print(f"Lasso RMSE={mean_squared_error(y_test,y_pred_l)**0.5:.4f}  R²={r2_score(y_test,y_pred_l):.4f}")

# Feature importance via coefficients
for name, coef in zip(data.feature_names, ols.coef_):
    print(f"  {name:<20} {coef:+.4f}")
```

```python
# ── PyTorch — Linear Regression with Adam ──────────────────────────────────
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data = fetch_california_housing()
X, y = data.data.astype("float32"), data.target.astype("float32")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test  = sc.transform(X_test)

X_tr = torch.tensor(X_train); y_tr = torch.tensor(y_train).unsqueeze(1)
X_te = torch.tensor(X_test);  y_te = torch.tensor(y_test).unsqueeze(1)

loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=256, shuffle=True)

model     = nn.Linear(X_train.shape[1], 1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

for epoch in range(100):
    for xb, yb in loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()

with torch.no_grad():
    rmse = ((model(X_te) - y_te)**2).mean().sqrt().item()
    print(f"PyTorch Linear Regression  RMSE={rmse:.4f}")
```

---

## 2. Logistic Regression

### Overview
| Property | Detail |
|---|---|
| **Type** | Classification |
| **Classification** | **Binary** natively; **Multi-class** via OvR or Softmax (Multinomial LR) |
| **Core Idea** | Model log-odds as a linear function; output probability via sigmoid |

### Use Cases & Why
| Use Case | Why |
|---|---|
| Email spam detection | Binary label; probability output enables threshold tuning |
| Credit default prediction | Regulatory need for explainability (GDPR, FCRA) |
| Customer churn prediction | Fast, auditable baseline with solid AUC |
| Ad CTR prediction | Scales to billions of samples with SGD |

### Loss Function
- **Binary Cross-Entropy:** `L = -[y·log(p) + (1-y)·log(1-p)]`
- **Multinomial Cross-Entropy:** `L = -Σ yₖ·log(pₖ)`

### Optimization & Adam
- **L-BFGS:** Quasi-Newton; preferred for small-medium data
- **SGD / Adam:** When implemented as a 1-layer neural network
  ```
  mₜ = β₁·mₜ₋₁ + (1-β₁)·gₜ         # 1st moment
  vₜ = β₂·vₜ₋₁ + (1-β₂)·gₜ²        # 2nd moment
  θₜ₊₁ = θₜ - α·m̂ₜ / (√v̂ₜ + ε)     # update
  ```
  Default: α=0.001, β₁=0.9, β₂=0.999, ε=1e-8

### Hyperparameters
| Hyperparameter | Typical Range | Effect |
|---|---|---|
| C (inverse regularization) | 0.001 – 100 | Higher C = less regularization |
| solver | lbfgs, saga, liblinear | saga supports L1 + large datasets |
| max_iter | 100 – 1000 | Convergence iterations |
| multi_class | ovr, multinomial | OvR = independent binary; multinomial = softmax |
| class_weight | balanced / manual | Handles class imbalance |

### Pros & Cons
**Pros**
- Calibrated probability outputs
- Single coefficient per feature — highly interpretable
- Fast with SGD; scales to millions of rows

**Cons**
- Linear decision boundary only
- Requires feature engineering for non-linear patterns
- Underperforms tree-based models on structured tabular data

### Industry Usage
- **Google/Meta:** Final scoring layer in CTR prediction (on deep learned embeddings)
- **Banking:** Credit scorecards — regulators require monotonic interpretable models
- **Healthcare:** Mortality prediction in ICUs (APACHE score)

### Optimization Tips
- Normalize numeric features — not scale-invariant
- Use `class_weight='balanced'` or SMOTE for imbalanced labels
- Tune `C` over log scale: [0.001, 0.01, 0.1, 1, 10, 100]
- Use `saga` + L1 for high-dimensional text — implicit feature selection

### Python Code

```python
# ── scikit-learn ────────────────────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, roc_auc_score,
                             ConfusionMatrixDisplay)
import matplotlib.pyplot as plt

data = load_breast_cancer()
X, y = data.data, data.target                      # binary: 0=malignant, 1=benign

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                     random_state=42, stratify=y)
sc = StandardScaler()
X_train_s = sc.fit_transform(X_train)
X_test_s  = sc.transform(X_test)

# Binary
lr_bin = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, random_state=42)
lr_bin.fit(X_train_s, y_train)
y_prob = lr_bin.predict_proba(X_test_s)[:, 1]
print("Binary LR")
print(classification_report(y_test, lr_bin.predict(X_test_s),
                            target_names=data.target_names))
print(f"AUC-ROC: {roc_auc_score(y_test, y_prob):.4f}")

# Multi-class (using Iris for 3-class demo)
from sklearn.datasets import load_iris
Xi, yi = load_iris(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(Xi, yi, test_size=0.2, random_state=42)
sc2 = StandardScaler()
lr_multi = LogisticRegression(C=1.0, solver='lbfgs', multi_class='multinomial',
                              max_iter=200, random_state=42)
lr_multi.fit(sc2.fit_transform(Xtr), ytr)
print("\nMulti-class LR (Iris)")
print(classification_report(yte, lr_multi.predict(sc2.transform(Xte))))
```

```python
# ── PyTorch — Logistic Regression with Adam ────────────────────────────────
import torch, torch.nn as nn
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

data = load_breast_cancer()
X, y = data.data.astype("float32"), data.target.astype("float32")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                     random_state=42, stratify=y)
sc = StandardScaler()
X_train = torch.tensor(sc.fit_transform(X_train))
X_test  = torch.tensor(sc.transform(X_test))
y_train = torch.tensor(y_train).unsqueeze(1)
y_test_np = y_test

# Single-layer logistic regression = linear + sigmoid
model     = nn.Sequential(nn.Linear(X_train.shape[1], 1), nn.Sigmoid())
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999),
                             eps=1e-8, weight_decay=1e-4)

for epoch in range(200):
    optimizer.zero_grad()
    loss = criterion(model(X_train), y_train)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1:3d}  Loss={loss.item():.4f}")

with torch.no_grad():
    probs = model(X_test).squeeze().numpy()
    print(f"AUC-ROC: {roc_auc_score(y_test_np, probs):.4f}")
```

---

## 3. Decision Tree

### Overview
| Property | Detail |
|---|---|
| **Type** | Classification & Regression |
| **Classification** | **Binary** and **Multi-class** natively |
| **Core Idea** | Recursively split data on the feature that maximally reduces impurity |

### Use Cases & Why
| Use Case | Why |
|---|---|
| Customer churn rule extraction | Human-readable if-then rules align with business logic |
| Medical diagnosis decision support | Explainability is mandatory; clinicians validate path |
| Fraud detection rule mining | Auditable decisions for regulatory compliance |
| Loan approval systems | Transparent criteria map directly to lending policy |

### Loss / Impurity Functions
- **Gini:** `G = 1 - Σpₖ²` — fast, sklearn default
- **Entropy:** `H = -Σ pₖ·log₂(pₖ)` — theoretically grounded
- **MSE (regression):** Minimize variance within leaf nodes

### Optimization
- Greedy split search — no gradient descent
- **Cost-Complexity Pruning (ccp_alpha):** Post-hoc overfitting reduction
- Adam not applicable

### Hyperparameters
| Hyperparameter | Typical Range | Effect |
|---|---|---|
| max_depth | 3 – 20 | Primary control for overfitting |
| min_samples_split | 2 – 50 | Min samples to allow a split |
| min_samples_leaf | 1 – 20 | Min samples in a leaf |
| criterion | gini, entropy | Impurity measure |
| ccp_alpha | 0.0 – 0.05 | Pruning strength |

### Pros & Cons
**Pros**
- Fully interpretable — visualizable tree
- No feature scaling required
- Captures non-linear patterns and interactions

**Cons**
- High variance — tiny data changes = very different trees
- Prone to overfitting without pruning
- Poor extrapolation beyond training range

### Python Code

```python
# ── scikit-learn ────────────────────────────────────────────────────────────
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                     random_state=42, stratify=y)

# Binary classification tree
dt = DecisionTreeClassifier(max_depth=5, criterion='gini',
                            min_samples_leaf=5, random_state=42)
dt.fit(X_train, y_train)
print(classification_report(y_test, dt.predict(X_test),
                            target_names=data.target_names))

# Visualize text representation
print(export_text(dt, feature_names=list(data.feature_names), max_depth=3))

# Cost-complexity pruning — find optimal alpha
path = dt.cost_complexity_pruning_path(X_train, y_train)
alphas = path.ccp_alphas

scores = []
for alpha in alphas:
    t = DecisionTreeClassifier(ccp_alpha=alpha, random_state=42)
    t.fit(X_train, y_train)
    scores.append(t.score(X_test, y_test))

best_alpha = alphas[scores.index(max(scores))]
print(f"\nBest ccp_alpha={best_alpha:.5f}  Accuracy={max(scores):.4f}")

# Feature importance
for name, imp in sorted(zip(data.feature_names, dt.feature_importances_),
                        key=lambda x: -x[1])[:5]:
    print(f"  {name:<30} {imp:.4f}")
```

---

## 4. Support Vector Machine (SVM)

### Overview
| Property | Detail |
|---|---|
| **Type** | Classification & Regression |
| **Classification** | **Binary** natively; **Multi-class** via OvO or OvR |
| **Core Idea** | Find maximum-margin hyperplane; kernel trick enables non-linear boundaries |

### Loss Function
- **Hinge Loss:** `L = max(0, 1 - yᵢ·(wᵀxᵢ + b))`
- **ε-insensitive (SVR):** `L = max(0, |y - ŷ| - ε)`
- Regularization: `(1/2)‖w‖²`

### Optimization
- **SMO (Sequential Minimal Optimization):** Solves dual QP — default in sklearn
- **Sub-gradient methods** for large-scale linear SVMs
- Adam not applicable to classical SVM

### Hyperparameters
| Hyperparameter | Typical Range | Effect |
|---|---|---|
| C | 0.01 – 1000 | Low C = wide margin; high C = fit training data tight |
| kernel | linear, rbf, poly | Feature space transformation |
| gamma | scale, auto, 1e-4 – 1 | RBF width — controls boundary complexity |
| degree | 2 – 5 | Polynomial kernel degree |

### Pros & Cons
**Pros**
- Effective in high-dim spaces (d >> n)
- Kernel trick enables complex non-linear boundaries
- Only support vectors determine boundary — memory efficient

**Cons**
- O(n²–n³) training — does not scale to large n
- No calibrated probabilities without Platt scaling
- Sensitive to C and gamma tuning

### Python Code

```python
# ── scikit-learn ────────────────────────────────────────────────────────────
from sklearn.svm import SVC, LinearSVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np

data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                     random_state=42, stratify=y)
sc = StandardScaler()
X_train_s = sc.fit_transform(X_train)
X_test_s  = sc.transform(X_test)

# RBF kernel SVM
svm_rbf = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
svm_rbf.fit(X_train_s, y_train)
y_prob = svm_rbf.predict_proba(X_test_s)[:, 1]
print("RBF-SVM Binary Classification")
print(classification_report(y_test, svm_rbf.predict(X_test_s),
                            target_names=data.target_names))
print(f"AUC-ROC: {roc_auc_score(y_test, y_prob):.4f}")
print(f"Support vectors per class: {svm_rbf.n_support_}")

# Grid search for optimal C and gamma
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 0.01, 0.001]}
gs = GridSearchCV(SVC(kernel='rbf', probability=True), param_grid,
                  cv=5, scoring='roc_auc', n_jobs=-1)
gs.fit(X_train_s, y_train)
print(f"\nBest params: {gs.best_params_}  CV AUC: {gs.best_score_:.4f}")

# LinearSVC — scales to large datasets
linear_svm = LinearSVC(C=1.0, max_iter=2000, random_state=42)
linear_svm.fit(X_train_s, y_train)
print(f"\nLinearSVC Accuracy: {linear_svm.score(X_test_s, y_test):.4f}")
```

---

## 5. K-Nearest Neighbors (KNN)

### Overview
| Property | Detail |
|---|---|
| **Type** | Classification & Regression |
| **Classification** | **Binary** and **Multi-class** (majority vote among k neighbors) |
| **Core Idea** | At inference, find k most similar training examples and aggregate labels |

### Distance Functions
- **Euclidean:** `d = √Σ(xᵢ - yᵢ)²`
- **Manhattan:** `d = Σ|xᵢ - yᵢ|`
- **Cosine:** `cos(θ) = (x·y)/(‖x‖·‖y‖)` — preferred for text/embeddings

### Hyperparameters
| Hyperparameter | Typical Range | Effect |
|---|---|---|
| k (n_neighbors) | 1 – 50 | Low k = high variance; high k = high bias |
| metric | euclidean, manhattan, cosine | Distance space |
| weights | uniform, distance | Distance-weighted voting |
| algorithm | ball_tree, kd_tree, brute | Data structure for search |

### Pros & Cons
**Pros**
- Zero training time
- Non-parametric — no distribution assumption
- Naturally multi-class; predictions backed by real examples

**Cons**
- O(n·d) inference — slow at scale without ANN indexing
- Heavily hurt by irrelevant/noisy features (curse of dimensionality)
- Entire training set must be stored in memory

### Python Code

```python
# ── scikit-learn ────────────────────────────────────────────────────────────
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import numpy as np

# Multi-class example (Iris — 3 classes)
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                     random_state=42, stratify=y)
sc = StandardScaler()
X_train_s = sc.fit_transform(X_train)
X_test_s  = sc.transform(X_test)

# Find optimal k using cross-validation
k_range = range(1, 31)
cv_scores = [cross_val_score(KNeighborsClassifier(n_neighbors=k, weights='distance'),
                              X_train_s, y_train, cv=5).mean() for k in k_range]
best_k = k_range[np.argmax(cv_scores)]
print(f"Best k={best_k}  CV Accuracy={max(cv_scores):.4f}")

knn = KNeighborsClassifier(n_neighbors=best_k, weights='distance',
                            metric='euclidean', algorithm='ball_tree')
knn.fit(X_train_s, y_train)
print(classification_report(y_test, knn.predict(X_test_s),
                            target_names=['setosa','versicolor','virginica']))

# Inspect neighbors for a test sample
distances, indices = knn.kneighbors(X_test_s[:1])
print(f"\nNearest {best_k} neighbors (indices): {indices[0]}")
print(f"Distances: {distances[0].round(3)}")
print(f"Their labels: {y_train[indices[0]]}")
```

---

## 6. Naive Bayes

### Overview
| Property | Detail |
|---|---|
| **Type** | Classification |
| **Classification** | **Binary** and **Multi-class** simultaneously (probabilistic across all classes) |
| **Core Idea** | Bayes theorem with strong conditional independence assumption |

### Loss Function
- **Negative Log-Likelihood:** `L = -log P(y|X)`
- Parameters via MLE: `P(xᵢ|y) = count(xᵢ,y) / count(y)` + Laplace smoothing

### Hyperparameters
| Hyperparameter | Typical Range | Effect |
|---|---|---|
| alpha (Laplace) | 0.0 – 2.0 | Prevents zero-frequency problem |
| var_smoothing (Gaussian NB) | 1e-9 – 1e-3 | Numerical stability |
| fit_prior | True / False | Use empirical class priors |

### Pros & Cons
**Pros**
- O(n·d) training — extremely fast
- Works well with small data and high-dim (text)
- Natural probability output

**Cons**
- Independence assumption frequently violated
- Cannot capture feature interactions
- Poorly calibrated probabilities

### Python Code

```python
# ── scikit-learn — three Naive Bayes variants ──────────────────────────────
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.datasets import load_iris, fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

# 1. Gaussian NB — continuous features (Iris, multi-class)
X, y = load_iris(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
gnb = GaussianNB(var_smoothing=1e-9)
gnb.fit(Xtr, ytr)
print("Gaussian NB (Iris, 3-class):")
print(classification_report(yte, gnb.predict(Xte)))

# 2. Multinomial NB — text classification (multi-class)
cats = ['sci.med', 'sci.space', 'talk.politics.guns']
train_data = fetch_20newsgroups(subset='train', categories=cats, remove=('headers','footers'))
test_data  = fetch_20newsgroups(subset='test',  categories=cats, remove=('headers','footers'))

cv = CountVectorizer(max_features=10000, ngram_range=(1,2))
X_tr_cv = cv.fit_transform(train_data.data)
X_te_cv = cv.transform(test_data.data)

mnb = MultinomialNB(alpha=0.5)          # Laplace smoothing
mnb.fit(X_tr_cv, train_data.target)
print("\nMultinomial NB (20newsgroups, 3-class):")
print(classification_report(test_data.target, mnb.predict(X_te_cv),
                            target_names=cats))

# 3. Bernoulli NB — binary feature presence (short text, binary)
bnb = BernoulliNB(alpha=1.0, binarize=0.0)
bnb.fit(X_tr_cv, train_data.target)
print("Bernoulli NB accuracy:", accuracy_score(test_data.target, bnb.predict(X_te_cv)))

# Inspect top features per class
log_probs = mnb.feature_log_prob_       # shape: (n_classes, n_features)
feature_names = np.array(cv.get_feature_names_out())
for cls_idx, cls_name in enumerate(cats):
    top10 = feature_names[np.argsort(log_probs[cls_idx])[-10:][::-1]]
    print(f"\nTop features for '{cls_name}': {', '.join(top10)}")
```

---

## 7. K-Means Clustering

### Overview
| Property | Detail |
|---|---|
| **Type** | Unsupervised — Clustering |
| **Classification** | Not applicable (unsupervised) |
| **Core Idea** | Iteratively assign points to nearest centroid; recompute centroids to minimize within-cluster variance |

### Loss / Objective Function
- **WCSS (Inertia):** `J = Σᵢ Σₓ∈Cᵢ ‖x - μᵢ‖²`
- **Silhouette Score:** `s = (b - a) / max(a, b)` — post-hoc quality metric

### Hyperparameters
| Hyperparameter | Typical Range | Effect |
|---|---|---|
| k (n_clusters) | 2 – 50+ | Use elbow method + silhouette score |
| init | k-means++, random | k-means++ dramatically improves convergence |
| n_init | 10 – 50 | Restarts; take best WCSS |
| max_iter | 100 – 500 | Iterations per run |

### Python Code

```python
# ── scikit-learn ────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs

# Generate synthetic clustered data
X, y_true = make_blobs(n_samples=1000, centers=4, cluster_std=0.8, random_state=42)
X_s = StandardScaler().fit_transform(X)

# Elbow method + Silhouette to find optimal k
inertias, silhouettes = [], []
k_range = range(2, 11)
for k in k_range:
    km = KMeans(n_clusters=k, init='k-means++', n_init=20, random_state=42)
    labels = km.fit_predict(X_s)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_s, labels, sample_size=500))

best_k = k_range[np.argmax(silhouettes)]
print(f"Optimal k by silhouette: {best_k}  score={max(silhouettes):.4f}")

# Final model
km_final = KMeans(n_clusters=best_k, init='k-means++', n_init=50, random_state=42)
km_final.fit(X_s)
print(f"WCSS (inertia): {km_final.inertia_:.2f}")
print(f"Cluster sizes: {np.bincount(km_final.labels_)}")

# Mini-Batch K-Means — for large datasets
mb_km = MiniBatchKMeans(n_clusters=best_k, batch_size=256, n_init=10, random_state=42)
mb_km.fit(X_s)
print(f"MiniBatch inertia: {mb_km.inertia_:.2f}")

# Assign new points to clusters
new_points = np.array([[1.0, 2.0], [-1.0, -1.5]])
new_points_s = StandardScaler().fit(X).transform(new_points)
print(f"New point cluster assignments: {km_final.predict(new_points_s)}")
```

---

## 8. DBSCAN

### Overview
| Property | Detail |
|---|---|
| **Type** | Unsupervised — Density-based Clustering |
| **Classification** | Not applicable; noise points labeled as -1 |
| **Core Idea** | Connect density-reachable points; sparse regions labeled as noise |

### Hyperparameters
| Hyperparameter | Typical Range | Effect |
|---|---|---|
| eps (ε) | Data-dependent | Neighborhood radius — use k-distance plot |
| min_samples | 2×d – 20 | Min points to form a dense region |
| metric | euclidean, cosine, haversine | Distance measure |

### Pros & Cons
**Pros**
- Arbitrary cluster shapes; auto-detects outliers; no k required

**Cons**
- eps and min_samples difficult to set; struggles with varying density

### Python Code

```python
# ── scikit-learn ────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import make_moons, make_blobs

# make_moons — non-spherical clusters that K-Means fails on
X, _ = make_moons(n_samples=500, noise=0.05, random_state=42)
X_s  = StandardScaler().fit_transform(X)

# Step 1: k-distance plot to estimate optimal eps
nn = NearestNeighbors(n_neighbors=5)
nn.fit(X_s)
distances, _ = nn.kneighbors(X_s)
k_distances = np.sort(distances[:, -1])       # distance to 5th nearest neighbor
# Look for the "knee" in k_distances plot — that's your eps

# Step 2: DBSCAN
db = DBSCAN(eps=0.15, min_samples=5, metric='euclidean')
labels = db.fit_predict(X_s)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise    = np.sum(labels == -1)
print(f"Clusters found: {n_clusters}  Noise points: {n_noise}")
print(f"Cluster sizes: {np.bincount(labels[labels >= 0])}")

# Compare: DBSCAN vs K-Means on non-spherical data
from sklearn.cluster import KMeans
km_labels = KMeans(n_clusters=2, random_state=42).fit_predict(X_s)
print(f"\nK-Means would split this into 2 artificial spherical clusters")
print(f"DBSCAN correctly finds the 2 moon shapes without being told k=2")

# HDBSCAN (more robust — varying density)
try:
    import hdbscan
    hdb = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5)
    hdb_labels = hdb.fit_predict(X_s)
    print(f"\nHDBSCAN clusters: {len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0)}")
except ImportError:
    print("\nInstall hdbscan: pip install hdbscan")
```

---

## 9. PCA (Principal Component Analysis)

### Overview
| Property | Detail |
|---|---|
| **Type** | Unsupervised — Dimensionality Reduction |
| **Classification** | Not applicable — transforms feature space |
| **Core Idea** | Project to orthogonal axes of maximum variance via SVD of covariance matrix |

### Loss / Objective
- **Reconstruction Error:** `L = ‖X - X_reconstructed‖²_F`
- Solved via **SVD:** `X = UΣVᵀ`

### Hyperparameters
| Hyperparameter | Typical Range | Effect |
|---|---|---|
| n_components | 2 – 0.95 (float) | Components count or % variance retained |
| svd_solver | full, randomized, arpack | Algorithm by data size |
| whiten | True / False | Normalize component scales |

### Python Code

```python
# ── scikit-learn ────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load digits dataset (64 features → reduce)
X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sc = StandardScaler()
X_train_s = sc.fit_transform(X_train)
X_test_s  = sc.transform(X_test)

# Find components explaining 95% variance
pca_full = PCA(n_components=0.95, svd_solver='full')
X_pca    = pca_full.fit_transform(X_train_s)
print(f"Original dims: {X_train_s.shape[1]}  →  PCA dims: {X_pca.shape[1]}")
print(f"Variance explained: {pca_full.explained_variance_ratio_.sum():.3f}")

# Cumulative explained variance
cumvar = np.cumsum(PCA().fit(X_train_s).explained_variance_ratio_)
n_90   = np.searchsorted(cumvar, 0.90) + 1
n_95   = np.searchsorted(cumvar, 0.95) + 1
print(f"Components for 90% variance: {n_90},  95% variance: {n_95}")

# Pipeline: PCA → Logistic Regression
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca',    PCA(n_components=n_95)),
    ('clf',    LogisticRegression(max_iter=1000, random_state=42))
])
pipe.fit(X_train, y_train)
print(f"\nPCA+LR accuracy on digits: {pipe.score(X_test, y_test):.4f}")

# Visualize 2D projection
pca2d = PCA(n_components=2)
X_2d  = pca2d.fit_transform(X_train_s)
print(f"\n2D PCA variance explained: {pca2d.explained_variance_ratio_.sum():.3f}")

# Reconstruct a digit and compute reconstruction error
pca_r  = PCA(n_components=20).fit(X_train_s)
X_rec  = pca_r.inverse_transform(pca_r.transform(X_test_s))
rec_err = np.mean((X_test_s - X_rec)**2)
print(f"Reconstruction MSE (20 components): {rec_err:.4f}")
```

---

## 10. Random Forest

### Overview
| Property | Detail |
|---|---|
| **Type** | Ensemble — Classification & Regression |
| **Classification** | **Binary** and **Multi-class** (majority vote) |
| **Core Idea** | Bootstrap-aggregated ensemble of decision trees, each on random feature subset — reduces variance |

### Loss / Impurity Functions
- **Gini** or **Entropy** for classification
- **MSE** for regression
- **OOB error** — free internal validation using samples not in each bootstrap

### Hyperparameters
| Hyperparameter | Typical Range | Effect |
|---|---|---|
| n_estimators | 100 – 2000 | More trees = lower variance; diminishing returns after ~500 |
| max_depth | None, 5 – 30 | Individual tree complexity |
| max_features | sqrt, log2, 0.3 | Feature subsampling — key for diversity |
| min_samples_leaf | 1 – 20 | Regularization |
| class_weight | balanced | Class imbalance handling |

### Python Code

```python
# ── scikit-learn ────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.inspection import permutation_importance

data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                     random_state=42, stratify=y)

# Train Random Forest
rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    max_features='sqrt',
    min_samples_leaf=2,
    class_weight='balanced',
    oob_score=True,
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train, y_train)

y_prob = rf.predict_proba(X_test)[:, 1]
print(classification_report(y_test, rf.predict(X_test), target_names=data.target_names))
print(f"AUC-ROC: {roc_auc_score(y_test, y_prob):.4f}")
print(f"OOB Score: {rf.oob_score_:.4f}")

# Permutation importance (unbiased — use on test set)
perm_imp = permutation_importance(rf, X_test, y_test, n_repeats=10,
                                  random_state=42, n_jobs=-1)
for idx in np.argsort(perm_imp.importances_mean)[::-1][:5]:
    print(f"  {data.feature_names[idx]:<35} {perm_imp.importances_mean[idx]:.4f}"
          f" ± {perm_imp.importances_std[idx]:.4f}")

# OOB error vs n_estimators curve
oob_errors = []
for n in range(10, 501, 10):
    rf_n = RandomForestClassifier(n_estimators=n, oob_score=True,
                                  max_features='sqrt', n_jobs=-1, random_state=42)
    rf_n.fit(X_train, y_train)
    oob_errors.append(1 - rf_n.oob_score_)
print(f"\nOOB error stabilizes around n_estimators={oob_errors.index(min(oob_errors))*10+10}")
```

---

## 11. Gradient Boosting

### Overview
| Property | Detail |
|---|---|
| **Type** | Ensemble — Classification & Regression |
| **Classification** | **Binary** and **Multi-class** (softmax extension) |
| **Core Idea** | Sequentially add weak learners correcting residuals of previous ensemble |

### Loss Functions
- **Binary:** `L = -[y·log(p) + (1-y)·log(1-p)]`
- **Multi-class:** Multinomial Deviance (softmax CE)
- **Regression:** MSE, MAE, Huber, Quantile
- Each step: fit tree to **negative gradient** of loss

### Hyperparameters
| Hyperparameter | Typical Range | Effect |
|---|---|---|
| n_estimators | 100 – 3000 | Boosting rounds |
| learning_rate | 0.01 – 0.3 | Shrinkage per step |
| max_depth | 3 – 8 | Shallow trees generalize better |
| subsample | 0.5 – 1.0 | Row subsampling per tree |

### Python Code

```python
# ── scikit-learn GradientBoostingClassifier ─────────────────────────────────
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                     random_state=42, stratify=y)

# Classic GB
gb = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    min_samples_leaf=5,
    random_state=42
)
gb.fit(X_train, y_train)
print("Gradient Boosting:")
print(classification_report(y_test, gb.predict(X_test), target_names=data.target_names))
print(f"AUC-ROC: {roc_auc_score(y_test, gb.predict_proba(X_test)[:,1]):.4f}")

# Track staged deviance for early stopping insight
train_scores = [s for s in gb.staged_decision_function(X_train)]
test_scores  = [s for s in gb.staged_decision_function(X_test)]
print(f"Best iteration: {gb.n_estimators_} (all {gb.n_estimators} used)")

# HistGradientBoosting — much faster (histogram-based, like LightGBM)
hgb = HistGradientBoostingClassifier(
    max_iter=300,
    learning_rate=0.05,
    max_depth=4,
    l2_regularization=0.1,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=42
)
hgb.fit(X_train, y_train)
print(f"\nHistGradientBoosting stopped at iteration: {hgb.n_iter_}")
print(f"AUC-ROC: {roc_auc_score(y_test, hgb.predict_proba(X_test)[:,1]):.4f}")
```

---

## 12. XGBoost / LightGBM

### Overview
| Property | Detail |
|---|---|
| **Type** | Optimized Gradient Boosting |
| **Classification** | **Binary** (`binary:logistic`) and **Multi-class** (`multi:softmax`) |
| **Core Idea** | XGBoost: regularized 2nd-order Taylor expansion. LightGBM: histogram-based leaf-wise growth with GOSS+EFB |

### Loss Functions
- **Binary:** `logloss`
- **Multi-class:** `mlogloss`
- **Regression:** `rmse`, `mae`, `huber`, `quantile`
- **Ranking:** `rank:ndcg`, `rank:pairwise`
- XGBoost: `L ≈ Σ[gᵢf(xᵢ) + ½hᵢf²(xᵢ)] + Ω(f)`

### Hyperparameters
| Hyperparameter | XGBoost | LightGBM | Effect |
|---|---|---|---|
| n_estimators | 500–3000 | 500–3000 | Boosting rounds |
| learning_rate / eta | 0.01–0.3 | 0.01–0.3 | Step size |
| max_depth | 3–10 | — | Tree depth |
| num_leaves | — | 31–255 | Leaf-wise control |
| reg_alpha (L1) | 0–10 | 0–10 | Sparsity |
| reg_lambda (L2) | 1–10 | 1–10 | Weight regularization |
| scale_pos_weight | n_neg/n_pos | — | Class imbalance |

### Python Code

```python
# ── XGBoost ──────────────────────────────────────────────────────────────────
import numpy as np
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import shap

data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                     random_state=42, stratify=y)

# Binary XGBoost
xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    scale_pos_weight=1,
    use_label_encoder=False,
    eval_metric='logloss',
    early_stopping_rounds=50,
    random_state=42
)
xgb_model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=50)

print("XGBoost Binary:")
print(classification_report(y_test, xgb_model.predict(X_test),
                            target_names=data.target_names))
print(f"AUC-ROC: {roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:,1]):.4f}")
print(f"Best iteration: {xgb_model.best_iteration}")

# SHAP explanation (critical for production audit)
explainer   = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)
print("\nTop 5 features by mean |SHAP|:")
mean_shap = np.abs(shap_values).mean(axis=0)
for idx in np.argsort(mean_shap)[::-1][:5]:
    print(f"  {data.feature_names[idx]:<35} {mean_shap[idx]:.4f}")
```

```python
# ── LightGBM ─────────────────────────────────────────────────────────────────
import lightgbm as lgb
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Multi-class with LightGBM (Iris — 3 classes)
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                     random_state=42, stratify=y)

lgb_multi = lgb.LGBMClassifier(
    objective='multiclass',
    num_class=3,
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    n_jobs=-1,
    random_state=42
)
lgb_multi.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)]
)
print("LightGBM Multi-class (Iris):")
print(classification_report(y_test, lgb_multi.predict(X_test),
                            target_names=['setosa','versicolor','virginica']))
```

---

## 13. Neural Network (MLP)

### Overview
| Property | Detail |
|---|---|
| **Type** | Supervised — Classification & Regression |
| **Classification** | **Binary** (sigmoid) and **Multi-class** (softmax) |
| **Core Idea** | Stacked learned linear transformations + non-linear activations; trained via backpropagation |

### Loss Functions
- **Binary:** `L = -[y·log(σ(ŷ)) + (1-y)·log(1-σ(ŷ))]`
- **Multi-class:** `L = -Σ yₖ·log(softmax(ŷₖ))`
- **Regression:** MSE, MAE, Huber

### Optimization — Adam Deep Dive
```
Initialize: m₀ = 0, v₀ = 0, t = 0

For each step t:
  gₜ = ∇L(θₜ)                         # Compute gradient
  mₜ = β₁·mₜ₋₁ + (1-β₁)·gₜ           # 1st moment (momentum)
  vₜ = β₂·vₜ₋₁ + (1-β₂)·gₜ²          # 2nd moment (adaptive scale)
  m̂ₜ = mₜ / (1-β₁ᵗ)                   # Bias correction
  v̂ₜ = vₜ / (1-β₂ᵗ)                   # Bias correction
  θₜ₊₁ = θₜ - α · m̂ₜ / (√v̂ₜ + ε)     # Parameter update

Default: α=0.001, β₁=0.9, β₂=0.999, ε=1e-8
```
**AdamW** (preferred): Same as Adam but weight decay is decoupled from gradient update.

### Hyperparameters
| Hyperparameter | Typical Range | Effect |
|---|---|---|
| learning_rate | 1e-4 – 1e-2 | Most impactful parameter |
| hidden_sizes | (64,), (128,64), (512,256,128) | Network width/depth |
| activation | relu, gelu, tanh | Non-linearity |
| dropout | 0.1 – 0.5 | Regularization |
| batch_size | 32 – 512 | Smaller = noisier gradients = better generalization |
| weight_decay | 1e-5 – 1e-2 | L2 via AdamW |

### Python Code

```python
# ── PyTorch MLP — Binary & Multi-class ────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np

# ── MLP Module ──────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim, dropout=0.3):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h),
                       nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def train_mlp(model, loader, criterion, optimizer, epochs=100, scheduler=None):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        if scheduler: scheduler.step()
        if (epoch+1) % 25 == 0:
            print(f"  Epoch {epoch+1:3d}  avg_loss={total_loss/len(loader):.4f}")

# ─── BINARY CLASSIFICATION ──────────────────────────────────────────────────
print("=" * 50)
print("Binary Classification (Breast Cancer)")
data = load_breast_cancer()
X, y = data.data.astype("float32"), data.target.astype("float32")
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                          random_state=42, stratify=y)
sc = StandardScaler()
X_tr = torch.tensor(sc.fit_transform(X_tr))
X_te = torch.tensor(sc.transform(X_te))
y_tr_t = torch.tensor(y_tr).unsqueeze(1)
y_te_t = torch.tensor(y_te).unsqueeze(1)

loader_bin = DataLoader(TensorDataset(X_tr, y_tr_t), batch_size=64, shuffle=True)

model_bin  = MLP(in_dim=30, hidden_dims=[128, 64, 32], out_dim=1, dropout=0.3)
optimizer  = torch.optim.AdamW(model_bin.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
criterion  = nn.BCEWithLogitsLoss()

train_mlp(model_bin, loader_bin, criterion, optimizer, epochs=100, scheduler=scheduler)

model_bin.eval()
with torch.no_grad():
    logits = model_bin(X_te).squeeze()
    probs  = torch.sigmoid(logits).numpy()
    preds  = (probs > 0.5).astype(int)
    print(classification_report(y_te, preds, target_names=data.target_names))
    print(f"AUC-ROC: {roc_auc_score(y_te, probs):.4f}")

# ─── MULTI-CLASS CLASSIFICATION ─────────────────────────────────────────────
print("=" * 50)
print("Multi-class Classification (Iris — 3 classes)")
Xi, yi = load_iris(return_X_y=True)
Xi, yi = Xi.astype("float32"), yi
X_tr2, X_te2, y_tr2, y_te2 = train_test_split(Xi, yi, test_size=0.2,
                                               random_state=42, stratify=yi)
sc2 = StandardScaler()
X_tr2 = torch.tensor(sc2.fit_transform(X_tr2))
X_te2 = torch.tensor(sc2.transform(X_te2))
y_tr2_t = torch.tensor(y_tr2)
y_te2_np = y_te2

loader_mc = DataLoader(TensorDataset(X_tr2, y_tr2_t), batch_size=32, shuffle=True)

model_mc  = MLP(in_dim=4, hidden_dims=[64, 32], out_dim=3, dropout=0.2)
opt_mc    = torch.optim.AdamW(model_mc.parameters(), lr=1e-3, weight_decay=1e-4)
crit_mc   = nn.CrossEntropyLoss()

train_mlp(model_mc, loader_mc, crit_mc, opt_mc, epochs=100)

model_mc.eval()
with torch.no_grad():
    logits_mc = model_mc(X_te2)
    preds_mc  = logits_mc.argmax(dim=1).numpy()
    print(classification_report(y_te2_np, preds_mc,
                                target_names=['setosa','versicolor','virginica']))
```

---

## 14. Convolutional Neural Network (CNN)

### Overview
| Property | Detail |
|---|---|
| **Type** | Deep Learning — Supervised |
| **Classification** | **Binary** (sigmoid head) and **Multi-class** (softmax head, 1000+ classes) |
| **Core Idea** | Shared convolutional filters learn spatial hierarchies; translation-invariant features |

### Loss Functions
- **Multi-class:** Categorical Cross-Entropy with softmax
- **Binary:** Binary Cross-Entropy with sigmoid
- **Segmentation:** Dice Loss + Cross-Entropy: `L_dice = 1 - (2|P∩G|)/(|P|+|G|)`
- **Detection:** Smooth L1 (bbox) + Cross-Entropy (class)

### Hyperparameters
| Hyperparameter | Typical Range | Effect |
|---|---|---|
| Filter size | 3×3, 5×5 | 3×3 is standard; factorized convolutions |
| n_filters | 32 – 512 per layer | Doubles with each pooling step |
| Dropout | 0.25 – 0.5 | After FC layers |
| Learning rate | 1e-4 – 1e-2 | With warmup + cosine decay |
| Batch size | 32 – 256 | |

### Python Code

```python
# ── PyTorch CNN — MNIST Multi-class (10 classes) ────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Data
transform = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,))])
train_ds = datasets.MNIST('./data', train=True,  download=True, transform=transform)
test_ds  = datasets.MNIST('./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=2)
test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=2)

# CNN Architecture
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 1→32 channels, 28×28 → 14×14
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            # Block 2: 32→64 channels, 14×14 → 7×7
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)            # CrossEntropyLoss handles softmax
        )

    def forward(self, x):
        return self.classifier(self.features(x))

model     = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=1e-2, steps_per_epoch=len(train_loader), epochs=10)

# Training loop
for epoch in range(10):
    model.train()
    running_loss = correct = total = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        running_loss += loss.item()
        correct += (logits.argmax(1) == labels).sum().item()
        total   += labels.size(0)
    print(f"Epoch {epoch+1:2d}  Loss={running_loss/len(train_loader):.4f}"
          f"  Train Acc={correct/total:.4f}")

# Evaluation
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        preds = model(imgs.to(device)).argmax(1).cpu()
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

print("\nCNN — MNIST Test Set:")
print(classification_report(all_labels, all_preds))
```

---

## 15. RNN / LSTM

### Overview
| Property | Detail |
|---|---|
| **Type** | Deep Learning — Sequential |
| **Classification** | **Binary** and **Multi-class** (classification head on hidden state) |
| **Core Idea** | Maintain hidden state across steps; LSTM gates solve vanishing gradient |

### Loss Functions
- **Classification:** Cross-Entropy on final hidden state
- **CTC (speech):** `L_CTC = -log P(y|x)` — unaligned sequences
- **Regression:** MSE over predicted time steps

### LSTM Gate Equations
```
fₜ = σ(Wf·[hₜ₋₁, xₜ] + bf)     # Forget gate
iₜ = σ(Wi·[hₜ₋₁, xₜ] + bi)     # Input gate
c̃ₜ = tanh(Wc·[hₜ₋₁, xₜ] + bc)  # Candidate cell
cₜ = fₜ⊙cₜ₋₁ + iₜ⊙c̃ₜ           # Cell state update
oₜ = σ(Wo·[hₜ₋₁, xₜ] + bo)     # Output gate
hₜ = oₜ⊙tanh(cₜ)                # Hidden state output
```

### Hyperparameters
| Hyperparameter | Typical Range | Effect |
|---|---|---|
| hidden_size | 64 – 512 | LSTM state dimensionality |
| num_layers | 1 – 4 | Stacked LSTM depth |
| dropout | 0.2 – 0.5 | Between LSTM layers |
| bidirectional | True/False | Past + future context |
| learning_rate | 1e-4 – 1e-3 | With Adam |

### Python Code

```python
# ── PyTorch LSTM — Sentiment Analysis (Binary) ──────────────────────────────
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np

# Simple character/word tokenizer for demo (binary: sci vs non-sci)
cats   = ['sci.space', 'talk.politics.misc']
data   = fetch_20newsgroups(subset='all', categories=cats,
                            remove=('headers','footers','quotes'))
labels = np.array(data.target)

# Build vocabulary from top words
from collections import Counter
import re

def tokenize(text): return re.findall(r'\b\w+\b', text.lower())

all_tokens = [w for doc in data.data for w in tokenize(doc)]
vocab = {w: i+2 for i, (w, _) in enumerate(Counter(all_tokens).most_common(5000))}
vocab['<pad>'] = 0; vocab['<unk>'] = 1

MAX_LEN = 100

def encode(doc):
    toks = [vocab.get(w, 1) for w in tokenize(doc)][:MAX_LEN]
    return toks + [0] * (MAX_LEN - len(toks))   # pad

X_enc = np.array([encode(d) for d in data.data], dtype=np.int64)
X_tr, X_te, y_tr, y_te = train_test_split(X_enc, labels, test_size=0.2,
                                           random_state=42, stratify=labels)

loader = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr, dtype=torch.float32)),
                    batch_size=64, shuffle=True)

# LSTM Classifier
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, n_layers, dropout=0.3):
        super().__init__()
        self.embedding  = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm       = nn.LSTM(embed_dim, hidden_size, n_layers, batch_first=True,
                                  bidirectional=True, dropout=dropout if n_layers>1 else 0)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, 1)  # ×2 for bidirectional

    def forward(self, x):
        emb = self.dropout(self.embedding(x))
        out, (hn, _) = self.lstm(emb)
        # Concatenate last hidden state from both directions
        h = torch.cat([hn[-2], hn[-1]], dim=1)
        return self.classifier(self.dropout(h))

model     = LSTMClassifier(vocab_size=5002, embed_dim=64, hidden_size=128,
                           n_layers=2, dropout=0.3)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

for epoch in range(10):
    model.train(); total_loss = 0
    for xb, yb in loader:
        optimizer.zero_grad()
        logits = model(xb).squeeze(1)
        loss   = criterion(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # ESSENTIAL for LSTMs
        optimizer.step()
        total_loss += loss.item()
    if (epoch+1) % 2 == 0:
        print(f"Epoch {epoch+1:2d}  Loss={total_loss/len(loader):.4f}")

model.eval()
with torch.no_grad():
    logits = model(torch.tensor(X_te)).squeeze(1)
    probs  = torch.sigmoid(logits).numpy()
    preds  = (probs > 0.5).astype(int)
    print(classification_report(y_te, preds, target_names=cats))
    print(f"AUC-ROC: {roc_auc_score(y_te, probs):.4f}")
```

---

## 16. Transformer

### Overview
| Property | Detail |
|---|---|
| **Type** | Deep Learning — Sequence/Multi-modal |
| **Classification** | **Binary** and **Multi-class** ([CLS] token or pooled representation → head) |
| **Core Idea** | Self-attention: every token attends to every other token in parallel; O(n²·d) per layer |

### Loss Functions
- **Causal LM (GPT):** `L = -Σ log P(xₜ | x₁...xₜ₋₁)`
- **Masked LM (BERT):** `L = -Σ log P(xᵢ | x\xᵢ)`
- **Fine-tuning:** Cross-Entropy on [CLS] token

### Optimization — Adam for Transformers
```
AdamW + Warmup + Cosine Decay:

Warmup schedule:
  lr(t) = α · min(t/warmup_steps, 1) · cosine_decay(t)

AdamW update:
  gₜ = ∇L + λ·θₜ                    # gradient + weight decay
  θₜ₊₁ = θₜ - α·m̂ₜ/(√v̂ₜ+ε) - α·λ·θₜ

β₁=0.9, β₂=0.98, ε=1e-9 (original Transformer paper values)
```

### Hyperparameters
| Hyperparameter | Typical Range | Effect |
|---|---|---|
| d_model | 128 – 4096 | Model dimensionality |
| num_heads | 4 – 64 | Attention heads |
| num_layers | 2 – 96 | Transformer blocks |
| dropout | 0.1 – 0.3 | Attention + FFN dropout |
| learning_rate | 1e-5 – 5e-4 | With warmup scheduler |
| warmup_steps | 1000 – 10000 | Steps before peak lr |

### Python Code

```python
# ── PyTorch Transformer for Text Classification ─────────────────────────────
import torch, math
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
from collections import Counter
import re

# ── Reuse tokenizer from LSTM section ───────────────────────────────────────
cats  = ['sci.med', 'sci.space', 'talk.politics.guns']
data  = fetch_20newsgroups(subset='all', categories=cats,
                           remove=('headers','footers','quotes'))
labels = np.array(data.target)

def tokenize(text): return re.findall(r'\b\w+\b', text.lower())
all_tokens = [w for doc in data.data for w in tokenize(doc)]
vocab = {w: i+2 for i, (w,_) in enumerate(Counter(all_tokens).most_common(8000))}
vocab['<pad>'] = 0; vocab['<unk>'] = 1
MAX_LEN = 128

def encode(doc):
    toks = [vocab.get(w,1) for w in tokenize(doc)][:MAX_LEN]
    return toks + [0]*(MAX_LEN - len(toks))

X_enc = np.array([encode(d) for d in data.data], dtype=np.int64)
X_tr, X_te, y_tr, y_te = train_test_split(X_enc, labels, test_size=0.2,
                                           random_state=42, stratify=labels)

# ── Transformer Classifier ───────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000)/d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, n_layers, n_classes,
                 dim_ff=512, dropout=0.1, max_len=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc   = PositionalEncoding(d_model, max_len, dropout)
        enc_layer      = nn.TransformerEncoderLayer(d_model, nhead, dim_ff,
                                                    dropout, batch_first=True)
        self.encoder   = nn.TransformerEncoder(enc_layer, n_layers)
        self.pool      = nn.AdaptiveAvgPool1d(1)
        self.classifier= nn.Linear(d_model, n_classes)
        # Initialize weights
        nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, x):
        pad_mask = (x == 0)                         # True = ignore padding positions
        emb = self.pos_enc(self.embedding(x) * math.sqrt(self.embedding.embedding_dim))
        enc = self.encoder(emb, src_key_padding_mask=pad_mask)
        # Mean-pool non-padding tokens
        pooled = enc.mean(dim=1)
        return self.classifier(pooled)

device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model     = TransformerClassifier(vocab_size=8002, d_model=128, nhead=4,
                                  n_layers=2, n_classes=3, dim_ff=256,
                                  dropout=0.1, max_len=128).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4,
                              betas=(0.9, 0.98), eps=1e-9, weight_decay=0.01)
loader    = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
                       batch_size=64, shuffle=True)

# Warmup scheduler
total_steps   = len(loader) * 15
warmup_steps  = total_steps // 10

def lr_lambda(step):
    if step < warmup_steps: return step / max(1, warmup_steps)
    return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

for epoch in range(15):
    model.train(); total_loss = correct = total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss   = criterion(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); scheduler.step()
        total_loss += loss.item()
        correct += (logits.argmax(1) == yb).sum().item()
        total   += yb.size(0)
    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1:2d}  Loss={total_loss/len(loader):.4f}"
              f"  Acc={correct/total:.4f}")

model.eval()
with torch.no_grad():
    preds = model(torch.tensor(X_te).to(device)).argmax(1).cpu().numpy()
print("\nTransformer (3-class text classification):")
print(classification_report(y_te, preds, target_names=cats))
```

---

## 17. Algorithm Comparison Guides

### 17.1 Binary Classification — Which Algorithm to Use?

| Algorithm | Use When | Avoid When |
|---|---|---|
| **Logistic Regression** | Need interpretable odds ratios; regulatory audit; fast baseline | Non-linear feature relationships; complex interactions exist |
| **Decision Tree** | Need printable if-then rules; clinician/analyst in the loop | High accuracy is the goal; data has high noise |
| **SVM (RBF)** | Small dataset (n<10k); high-dimensional sparse features (text) | n > 100k (too slow); need probability outputs without extra calibration |
| **Random Forest** | Medium dataset; want feature importance; moderate accuracy | Predictions needed in <1ms (large ensemble is slow) |
| **XGBoost / LightGBM** | Best accuracy on structured tabular data; class imbalance present | Very small dataset (<500 rows) — likely to overfit |
| **Naive Bayes** | Text data; need sub-millisecond inference; very small training set | Features are strongly correlated; numeric continuous features (use Gaussian NB carefully) |
| **MLP (Neural Net)** | Complex feature interactions; have GPU; large dataset | n < 5000; need interpretability |
| **KNN** | Embedding-based similarity; need prediction backed by precedents | Large n without ANN indexing; high-dimensional raw features |

**Rule of thumb priority for binary classification:**
```
Start with Logistic Regression (baseline, interpretable)
    ↓ if accuracy insufficient
Try Random Forest (robust, minimal tuning)
    ↓ if still insufficient
Try XGBoost/LightGBM (maximum accuracy, structured data)
    ↓ if features are images/text/sequences
Use CNN / LSTM / Transformer (domain-specific architectures)
```

---

### 17.2 Multi-class Classification — Which Algorithm to Use?

| Algorithm | Strategy | Best For |
|---|---|---|
| **Logistic Regression** | Multinomial (softmax) or OvR | Interpretable multi-class; text with TF-IDF |
| **Decision Tree** | Native multi-class at each node | Rule extraction; few classes |
| **SVM** | OvO (default) or OvR | High-dimensional text; few classes |
| **Random Forest** | Native multi-class voting | Tabular data; moderate class count |
| **XGBoost / LightGBM** | `objective='multi:softmax'` | Large tabular; many classes; class imbalance |
| **Naive Bayes** | Native probabilistic | Text classification; many categories |
| **MLP** | Softmax output layer | Complex non-linear multi-class |
| **CNN** | Softmax (ImageNet=1000 classes) | Image classification at any scale |
| **Transformer** | [CLS] token → linear head | Text/sequence classification |

**When classes > 100:** Use hierarchical classification or embedding + cosine similarity retrieval rather than flat multi-class.

---

### 17.3 Clustering — DBSCAN vs K-Means

| Criterion | K-Means | DBSCAN |
|---|---|---|
| Number of clusters | Must specify k upfront | Determined automatically |
| Cluster shape | Spherical only | Arbitrary shape |
| Outlier handling | Assigns all points | Labels outliers as noise (-1) |
| Speed | O(n·k·i) — fast | O(n log n) with indexing |
| Varying density | Handles well | Struggles with varying density |
| Best for | Customer segments, color quantization | Geospatial, anomaly detection |
| When to avoid | Non-spherical data, unknown k | Uniform density unknown, high-dim |

**Recommendation:** Use K-Means for well-separated, roughly equal-sized clusters with known k. Use DBSCAN/HDBSCAN when cluster shapes are unknown, when outliers must be detected, or when k is unknown.

---

### 17.4 Dimensionality Reduction — PCA vs UMAP vs t-SNE

| Criterion | PCA | UMAP | t-SNE |
|---|---|---|---|
| Speed | Fastest (closed-form) | Fast (O(n^1.14)) | Slowest (O(n² or n·log n)) |
| Preserves | Global variance | Global + local structure | Local structure only |
| Deterministic | Yes | No (random init) | No |
| New point projection | Yes (transform) | Yes (with fit) | No (must refit) |
| Interpretable axes | Partially (loadings) | No | No |
| Best for | Pre-processing before ML; factor models | Visualization + clustering pre-processing | Pure 2D/3D visualization |

---

### 17.5 Regression — Which Algorithm?

| Algorithm | Use When | Avoid When |
|---|---|---|
| **Linear Regression** | Linear relationship; need coefficients; regulatory | Non-linear target |
| **Ridge / Lasso** | Correlated features; feature selection needed | Complex non-linear relationships |
| **Decision Tree Regressor** | Interpretable rules; step-wise predictions | Smooth continuous extrapolation |
| **Random Forest Regressor** | Non-linear, robust | Need exact extrapolation beyond training range |
| **XGBoost / LightGBM** | Best accuracy on tabular regression | Very small datasets |
| **MLP Regressor** | Complex interactions; large data | Small n; need interpretability |
| **LSTM / Transformer** | Time series; sequential dependencies | Tabular regression (use tree methods) |

---

*Last updated: May 2026 | Pramod Modi — Staff ML Engineer, ServiceNow*
