# Use Case 2: Credit Card Fraud Detection
> Binary Classification with Extreme Class Imbalance — plain-English explanations before every code block

---

## Table of Contents
1. [Problem Statement](#1-problem-statement)
2. [Dataset & Data Understanding](#2-dataset--data-understanding)
3. [Feature Engineering](#3-feature-engineering)
4. [Handling Class Imbalance](#4-handling-class-imbalance)
5. [Model Selection — Why This, Not That](#5-model-selection--why-this-not-that)
6. [Solution 1: LightGBM (Real-time)](#6-solution-1-lightgbm-real-time)
7. [Solution 2: Neural Network with Focal Loss](#7-solution-2-neural-network-with-focal-loss)
8. [Solution 3: Isolation Forest + LightGBM Ensemble](#8-solution-3-isolation-forest--lightgbm-ensemble)
9. [Evaluation Strategy](#9-evaluation-strategy)
10. [Production Considerations](#10-production-considerations)

---

## 1. Problem Statement

A payment processor handles **10 million transactions per day**. Approximately **0.1% are fraudulent** — only 10,000 out of 10,000,000 are fraud.

**Two competing costs:**
- **False Negative (miss fraud):** Average loss of $150 per fraudulent transaction
- **False Positive (block legitimate transaction):** Customer frustration, support cost ~$8, churn risk

**Goal:**
- Catch 80%+ of fraud while blocking no more than 0.1% of legitimate transactions
- Inference latency < 10ms (real-time blocking, not batch)
- Every fraud flag must have a human-readable explanation (SHAP) for compliance

**The core challenge:** 0.1% fraud rate means standard accuracy is meaningless. A model predicting "always legitimate" gets 99.9% accuracy but catches zero fraud.

---

## 2. Dataset & Data Understanding

**What this code does:**
We simulate a transaction dataset where fraud and legitimate transactions have systematically different statistical patterns. Fraudsters tend to: use round amounts, transact at unusual hours, use new accounts, trigger velocity spikes (many transactions in a short window), and transact cross-country. We generate 100,000 transactions (1% fraud for the demo; real-world would be 0.1%) and compare the average value of each feature between fraud and legitimate classes.

**Key terms explained:**
- **Lognormal distribution** — a distribution where the logarithm of the variable is normally distributed. Transaction amounts follow this: most amounts are small, a few are very large. `np.random.lognormal(mean, std)` generates realistic transaction amounts.
- **Poisson distribution** — counts of events happening in a fixed time window. `np.random.poisson(lambda)` generates how many transactions a user makes per hour, where lambda is the expected rate. Fraudsters have a higher rate than normal users.
- **Exponential distribution** — models the time or distance until the next event. `np.random.exponential(scale)` generates geographic distances from home. Fraud transactions tend to be farther from the cardholder's usual location.
- **Beta distribution** — generates numbers between 0 and 1 with a specified shape. `np.random.beta(a, b)` — we use it for merchant fraud rates and email domain risk scores. `beta(3, 5)` tends toward ~0.37 (higher risk); `beta(1, 20)` tends toward ~0.05 (lower risk).

```python
import numpy as np
import pandas as pd

np.random.seed(42)
N_LEGIT = 99000
N_FRAUD = 1000    # 1% for demo; production would be 0.1% (100x more imbalanced)

def simulate_transactions(n, is_fraud=False):
    if is_fraud:
        return pd.DataFrame({
            'amount':                np.random.choice([99.99,499.99,999.99,1999.99,49.99],n)
                                     + np.random.normal(0,5,n),
            'hour_of_day':           np.random.choice([1,2,3,4,23,0], n),  # off-hours
            'merchant_category':     np.random.choice([9,12,15], n),       # high-risk MCCs
            'user_age_days':         np.random.exponential(60, n).clip(1,3650),  # new accounts
            'transactions_last_1h':  np.random.poisson(8, n),              # velocity spike
            'transactions_last_24h': np.random.poisson(25, n),
            'transactions_last_7d':  np.random.poisson(50, n),
            'amount_mean_30d':       np.random.normal(80, 30, n).clip(0),
            'amount_std_30d':        np.random.normal(20, 10, n).clip(0),
            'distance_from_home_km': np.random.exponential(800, n).clip(0,20000),
            'same_country':          np.random.binomial(1, 0.2, n),
            'card_present':          np.random.binomial(1, 0.1, n),
            'merchant_fraud_rate':   np.random.beta(3, 5, n),
            'device_seen_before':    np.random.binomial(1, 0.3, n),
            'ip_in_blacklist':       np.random.binomial(1, 0.4, n),
            'email_domain_risk':     np.random.beta(2, 3, n),
            'label': 1
        })
    else:
        return pd.DataFrame({
            'amount':                np.abs(np.random.lognormal(3.5,1.2,n)),
            'hour_of_day':           np.random.choice(range(8,22), n),
            'merchant_category':     np.random.choice(range(20), n),
            'user_age_days':         np.random.normal(730,400,n).clip(1,3650),
            'transactions_last_1h':  np.random.poisson(1, n),
            'transactions_last_24h': np.random.poisson(5, n),
            'transactions_last_7d':  np.random.poisson(20, n),
            'amount_mean_30d':       np.random.normal(75,40,n).clip(0),
            'amount_std_30d':        np.random.normal(35,20,n).clip(0),
            'distance_from_home_km': np.random.exponential(20,n).clip(0,5000),
            'same_country':          np.random.binomial(1, 0.92, n),
            'card_present':          np.random.binomial(1, 0.7, n),
            'merchant_fraud_rate':   np.random.beta(1, 20, n),
            'device_seen_before':    np.random.binomial(1, 0.9, n),
            'ip_in_blacklist':       np.random.binomial(1, 0.01, n),
            'email_domain_risk':     np.random.beta(1, 10, n),
            'label': 0
        })

df = pd.concat([simulate_transactions(N_LEGIT, False),
                simulate_transactions(N_FRAUD, True)]).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Total transactions: {len(df):,}")
print(f"Fraud rate:         {df['label'].mean():.2%}")

feat_cols = [c for c in df.columns if c != 'label']
print("\nFeature comparison — Fraud vs Legitimate (mean values):")
comp = df.groupby('label')[feat_cols].mean().T
comp.columns = ['Legitimate','Fraud']
comp['Ratio F/L'] = (comp['Fraud'] / comp['Legitimate'].clip(0.001)).round(2)
print(comp.sort_values('Ratio F/L', ascending=False).head(10).to_string())
```

---

## 3. Feature Engineering

**What this code does:**
We create new features by combining and transforming the raw transaction data. The goal is to make patterns that are obvious to a human fraud analyst — like "this transaction is 5x the user's usual spending" — explicit and numerical so the model can use them directly.

**Key terms explained:**
- **Z-score** — measures how many standard deviations a value is from the mean. Formula: `z = (x - mean) / std`. A transaction amount z-score of +4 means "this amount is 4 standard deviations above this user's usual spending." Z-scores of 3+ are highly anomalous. Fraud analysts call this "anomaly from baseline."
- **Velocity feature** — counts of events in a recent time window (1 hour, 24 hours, 7 days). "8 transactions in the last hour when the average is 1 per hour" is a 8× velocity spike — a classic fraud signal. Fraudsters make many rapid small transactions when they get a card.
- **`log1p` (log(1 + x))** — applies a logarithm transformation. Transaction amounts are heavily right-skewed (most are small, a few are enormous). Log-transforming compresses the range so that the difference between $1 and $10 is treated similarly to the difference between $100 and $1000. The `+1` ensures log(0) = 0 rather than undefined.
- **Interaction feature** — a feature created by multiplying or combining two other features. `amount × distance_from_home` captures the joint risk of "high amount AND far from home" which is more predictive than either alone.
- **Composite risk score** — a single number summing up multiple binary risk flags. If 6 out of 8 risk flags are true, the composite score is 6. Gives the model a compact summary of overall riskiness.
- **Mann-Whitney U test** — a statistical test that checks whether the distribution of a feature differs significantly between two groups (fraud vs legitimate). Unlike correlation, it doesn't assume a linear relationship. High U statistic + low p-value = this feature is highly discriminative between fraud and legitimate.

```python
import numpy as np
import pandas as pd

def engineer_fraud_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ── Amount features ──────────────────────────────────────────────────
    # Z-score: how many standard deviations from the user's 30-day average?
    # A z-score of +4 means "this amount is 4× more than usual" — anomalous
    df['amount_zscore']    = (df['amount'] - df['amount_mean_30d']) / (df['amount_std_30d'] + 1e-6)
    # Round amount flag: fraudsters often test with exact round numbers
    df['is_round_amount']  = (df['amount'] % 1.0 == 0).astype(int)
    # Log transform: compresses the huge range of transaction amounts
    df['amount_log']       = np.log1p(df['amount'])
    # Amount buckets: model can easily split on "high amount" vs "low amount"
    df['amount_bin']       = pd.cut(df['amount'], bins=[0,10,50,100,500,1000,np.inf],
                                    labels=[0,1,2,3,4,5]).astype(int)

    # ── Velocity features ────────────────────────────────────────────────
    # Ratio: recent rate vs expected rate — "are transactions accelerating?"
    # If someone made 8 transactions in 1h but their 24h average is 5/day, ratio = 8/(5/24) ≈ 38×
    df['velocity_ratio_1h']  = df['transactions_last_1h']  / (df['transactions_last_24h'] / 24 + 1e-6)
    df['velocity_ratio_24h'] = df['transactions_last_24h'] / (df['transactions_last_7d']  /  7 + 1e-6)
    # Spike flags: is the current velocity in the top 5% of all users?
    df['velocity_spike_1h']  = (df['transactions_last_1h']  > df['transactions_last_1h'].quantile(0.95)).astype(int)
    df['velocity_spike_24h'] = (df['transactions_last_24h'] > df['transactions_last_24h'].quantile(0.95)).astype(int)

    # ── Behavioural anomaly features ─────────────────────────────────────
    df['is_off_hours']      = ((df['hour_of_day'] >= 1) & (df['hour_of_day'] <= 6)).astype(int)
    df['cross_country']     = 1 - df['same_country']
    # Interaction: card-not-present (CNP) at off-hours — both risky individually, dangerous together
    df['cnp_off_hours']     = ((df['card_present'] == 0) & (df['is_off_hours'] == 1)).astype(int)
    # New account making a large purchase — common fraud pattern
    df['new_acct_high_amt'] = ((df['user_age_days'] < 30) & (df['amount'] > 200)).astype(int)

    # ── Composite risk score ─────────────────────────────────────────────
    # Sum of binary risk flags: higher total = more risk signals present simultaneously
    df['composite_risk'] = (
        df['is_off_hours'] + df['cross_country'] + df['ip_in_blacklist'] +
        (1 - df['device_seen_before']) + (1 - df['card_present']) +
        df['velocity_spike_1h'] + df['new_acct_high_amt'] +
        (df['amount_zscore'] > 3).astype(int)   # amount > 3 SDs from user baseline
    )

    # ── Interaction features ─────────────────────────────────────────────
    # These capture joint patterns neither feature alone would reveal
    df['amount_x_dist']    = df['amount_log'] * np.log1p(df['distance_from_home_km'])
    df['velocity_x_risk']  = df['velocity_ratio_1h'] * df['composite_risk']
    df['merchant_x_email'] = df['merchant_fraud_rate'] * df['email_domain_risk']

    return df

df_fe = engineer_fraud_features(df)

BASE_FEATS  = [c for c in df.columns if c not in ['label']]
NEW_FEATS   = ['amount_zscore','is_round_amount','amount_log','amount_bin',
               'velocity_ratio_1h','velocity_ratio_24h','velocity_spike_1h',
               'velocity_spike_24h','is_off_hours','cross_country','cnp_off_hours',
               'new_acct_high_amt','composite_risk','amount_x_dist',
               'velocity_x_risk','merchant_x_email']
ALL_FEATS   = BASE_FEATS + NEW_FEATS

X = df_fe[ALL_FEATS].values.astype('float32')
y = df_fe['label'].values
print(f"Features: {len(BASE_FEATS)} raw + {len(NEW_FEATS)} engineered = {len(ALL_FEATS)} total")

# Mann-Whitney U test: which features best separate fraud from legitimate?
# High U statistic = the two distributions are very different = discriminative feature
from scipy.stats import mannwhitneyu
fraud_mask = y == 1
print("\nTop discriminative features (Mann-Whitney U test):")
results = []
for i, fname in enumerate(ALL_FEATS):
    stat, pval = mannwhitneyu(X[fraud_mask, i], X[~fraud_mask, i], alternative='two-sided')
    results.append((fname, stat, pval))
results.sort(key=lambda x: x[1], reverse=True)
for name, stat, pval in results[:10]:
    print(f"  {name:<30}  U={stat:.0f}  p={pval:.2e}")
```

---

## 4. Handling Class Imbalance

**What this code does:**
The fundamental challenge is that only 1% of transactions are fraud (in our demo; real-world is 0.1%). A model trained naively will predict "legitimate" for almost everything and be "99% accurate" while catching almost no fraud. We show four strategies to fix this, with code for each.

**Key terms explained:**
- **Class weight** — during training, multiply the loss for minority-class examples by a larger number. A Security ticket incorrectly classified contributes 7× more to the loss than a Network ticket. The model is forced to pay attention to the minority class.
- **`scale_pos_weight`** — the LightGBM/XGBoost equivalent of class_weight. Set to `count(negative) / count(positive)`. If 99 legitimate for every 1 fraud, set to 99. Explicitly tells the boosting algorithm to care 99× more about catching fraud.
- **SMOTE (Synthetic Minority Oversampling Technique)** — creates new synthetic fraud examples by interpolating between existing fraud examples in feature space. If fraud example A has amount=100 and example B has amount=200, SMOTE might create a new fraud example with amount=150. Makes the training set more balanced without simply repeating examples.
- **Under-sampling** — keep all fraud examples but randomly discard most legitimate examples. Faster than SMOTE. Loses information from legitimate transactions but often works well in practice.
- **Focal Loss** — a modified cross-entropy loss that automatically down-weights easy-to-classify examples. If the model is already 99.9% confident a transaction is legitimate, the loss contribution is near zero — the model focuses its learning on the hard, ambiguous cases (which are usually fraud). Controlled by `gamma` parameter: gamma=0 = standard cross-entropy, gamma=2 = strong focus on hard examples.

```python
import numpy as np
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Train: {len(y_train):,}  Fraud: {y_train.sum():,} ({y_train.mean():.2%})")
print(f"Test:  {len(y_test):,}   Fraud: {y_test.sum():,} ({y_test.mean():.2%})")

# Strategy 1: Class weights — cheapest, no data modification
from sklearn.utils.class_weight import compute_class_weight
cw = compute_class_weight('balanced', classes=[0,1], y=y_train)
pos_weight = cw[1] / cw[0]
print(f"\nStrategy 1 — Class weights: legit={cw[0]:.2f}, fraud={cw[1]:.2f}")
print(f"  LightGBM scale_pos_weight = {pos_weight:.1f}")
print(f"  Meaning: fraud errors cost {pos_weight:.0f}× more than legit errors")

# Strategy 2: SMOTE — creates synthetic fraud examples
try:
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(sampling_strategy=0.1, random_state=42, n_jobs=-1)
    X_smote, y_smote = smote.fit_resample(X_train, y_train)
    print(f"\nStrategy 2 — SMOTE: {len(y_smote):,} samples  Fraud: {y_smote.mean():.2%}")
    SMOTE_AVAILABLE = True
except ImportError:
    print("\nSMOTE: pip install imbalanced-learn")
    X_smote, y_smote = X_train, y_train
    SMOTE_AVAILABLE = False

# Strategy 3: Under-sampling — keep all fraud, discard most legit
def undersample(X, y, ratio=0.1):
    """
    ratio = desired fraud fraction in resampled dataset.
    Keeps all fraud samples; randomly drops legitimate samples to achieve ratio.
    """
    fraud_idx = np.where(y == 1)[0]
    legit_idx = np.where(y == 0)[0]
    n_legit   = min(len(legit_idx), int(len(fraud_idx) / ratio))
    keep      = np.concatenate([fraud_idx, np.random.choice(legit_idx, n_legit, replace=False)])
    np.random.shuffle(keep)
    return X[keep], y[keep]

X_under, y_under = undersample(X_train, y_train, ratio=0.1)
print(f"\nStrategy 3 — Under-sampling: {len(y_under):,} samples  Fraud: {y_under.mean():.2%}")

# Strategy 4: Focal Loss — covered in Solution 2 (PyTorch neural network)
print("""
\nStrategy comparison:
  Class weights  → Start here. Zero cost. Implemented as one parameter.
  SMOTE          → Use when model still misses too much fraud after class weights.
                   Adds synthetic fraud examples for better boundary learning.
  Under-sampling → Useful when training time is very constrained.
                   Loses legitimate transaction signal.
  Focal Loss     → Best for neural networks. No data modification needed.
                   Automatically focuses learning on hard/uncertain examples.
""")
```

---

## 5. Model Selection — Why This, Not That

**What this section explains:**
Fraud detection has specific requirements (sub-10ms inference, SHAP explanation for every flag, continuous retraining on streaming data) that make certain algorithms unsuitable regardless of their accuracy.

| Algorithm | Verdict | Reason |
|---|---|---|
| **Logistic Regression** | ❌ | Cannot capture the complex non-linear interactions between amount, velocity, location, time of day. Fraud patterns live in the intersections of these features. |
| **Decision Tree** | ❌ | High variance — very sensitive to the small number of fraud examples. Easily overfits the specific fraud cases in training data. |
| **SVM** | ❌ | O(n²) training on 10M daily transactions means retraining takes hours. Unacceptable for daily model refreshes. |
| **Naive Bayes** | ❌ | Features (amount, velocity, location) are strongly correlated with each other. Independence assumption badly violated. Terrible recall. |
| **KNN** | ❌ | O(n) inference per transaction at 10M+ volume. Even with ANN indexing, impractical for continuous updating as new transactions arrive. |
| **LightGBM** | ✅ | Sub-millisecond inference. Native `scale_pos_weight` for imbalance. Handles missing values (device/IP data often missing). SHAP via `TreeExplainer`. Re-trains in minutes. |
| **Neural Net (Focal Loss)** | ✅ | Captures very complex feature interactions. Focal Loss handles imbalance at the loss level. GPU required. |
| **Isolation Forest + LightGBM** | ✅ | Isolation Forest catches novel fraud patterns not in training data. Combined with LightGBM for known patterns. Best for evolving fraud. |

---

## 6. Solution 1: LightGBM (Real-time)

**What this code does:**
We train LightGBM on all engineered features. LightGBM uses `scale_pos_weight` to handle the 99:1 class imbalance. We tune the classification threshold: instead of using 0.5 (predict fraud if probability > 50%), we find the threshold that maximises fraud recall while keeping the false positive rate below our business target (< 0.5% of legitimate transactions blocked).

We then use SHAP to explain every fraud prediction — required for compliance and customer appeals.

**Key terms explained:**
- **LightGBM** — a gradient boosting library that uses histogram binning and leaf-wise tree growth. Much faster than XGBoost for large datasets. "Leaf-wise" means trees grow deeper at the most information-gaining leaf first, rather than level-by-level.
- **`scale_pos_weight`** — tells LightGBM that positive (fraud) examples should count `n_legit/n_fraud` times more than negative examples. Directly counteracts the imbalance.
- **`min_child_samples`** — the minimum number of training examples in a leaf node. Setting this to 20 prevents the model from creating tiny leaves that overfit to the few fraud examples in training.
- **`average_precision` metric** — the area under the Precision-Recall curve. This is better than AUC-ROC for heavily imbalanced data because it focuses on the minority class performance. We use it for early stopping.
- **Precision-Recall curve** — plots Precision (y-axis) vs Recall (x-axis) as the classification threshold varies. Helps find the threshold that achieves the business-required recall at acceptable precision.
- **False Positive Rate (FPR)** — of all legitimate transactions, what fraction were incorrectly blocked? FPR = FP / (FP + TN). Business requires FPR < 0.005 (block fewer than 0.5% of legitimate transactions).
- **SHAP TreeExplainer** — a fast SHAP algorithm specifically for tree-based models. Uses the tree structure directly to compute exact SHAP values in O(TL²) time where T = trees, L = max leaves. Much faster than model-agnostic SHAP methods.

```python
import lightgbm as lgb
import numpy as np
import shap
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              confusion_matrix, classification_report)
from sklearn.model_selection import train_test_split as tts

fraud_ct  = y_train.sum()
legit_ct  = (y_train == 0).sum()
spw       = legit_ct / fraud_ct   # scale_pos_weight = how many times fraud errors matter more
print(f"scale_pos_weight = {spw:.1f}  (fraud errors penalised {spw:.0f}× more)")

lgb_model = lgb.LGBMClassifier(
    objective='binary',
    metric='average_precision',   # optimise AUC-PR, not AUC-ROC — better for imbalance
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=20,         # prevent overfitting the small fraud cluster
    subsample=0.8,
    subsample_freq=5,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    scale_pos_weight=spw,         # core imbalance correction
    n_jobs=-1, random_state=42, verbose=-1
)

# Keep 10% of training data as validation for early stopping
Xt, Xv, yt, yv = tts(X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)
lgb_model.fit(Xt, yt, eval_set=[(Xv, yv)],
              callbacks=[lgb.early_stopping(30), lgb.log_evaluation(100)])

y_prob = lgb_model.predict_proba(X_test)[:, 1]
print(f"\nAUC-ROC: {roc_auc_score(y_test, y_prob):.4f}")
print(f"AUC-PR:  {average_precision_score(y_test, y_prob):.4f}  ← primary metric")

# Find optimal threshold: maximum precision where recall ≥ 0.80
# We cannot simply use 0.5 — the optimal cut-off for imbalanced binary problems
# is almost never at 0.5 and must be chosen based on business requirements
from sklearn.metrics import precision_recall_curve
prec_arr, rec_arr, thresh_arr = precision_recall_curve(y_test, y_prob)
# Find highest-precision threshold that still catches at least 80% of fraud
viable = [(p,r,t) for p,r,t in zip(prec_arr, rec_arr, thresh_arr) if r >= 0.80]
if viable:
    best_p, best_r, best_t = max(viable, key=lambda x: x[0])
    print(f"\nOptimal threshold (Recall≥80%): {best_t:.4f}")
    print(f"  At this threshold: Precision={best_p:.3f}, Recall={best_r:.3f}")
else:
    best_t = 0.5

y_pred = (y_prob >= best_t).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
fpr = fp / (fp + tn)
print(f"\nAt chosen threshold:")
print(f"  Fraud caught  (TP): {tp}  |  Fraud missed (FN): {fn}")
print(f"  Legit blocked (FP): {fp}  |  Legit allowed (TN): {tn}")
print(f"  False Positive Rate: {fpr:.4%}  (blocking {fpr:.3%} of legitimate transactions)")

# SHAP explanation for every flagged transaction
explainer  = shap.TreeExplainer(lgb_model)
shap_vals  = explainer.shap_values(X_test[:200])
# For binary LightGBM, shap_values returns [neg_class_shap, pos_class_shap]
sv_fraud   = shap_vals[1] if isinstance(shap_vals, list) else shap_vals
mean_shap  = np.abs(sv_fraud).mean(axis=0)

print("\nTop 10 features by mean |SHAP| for fraud class:")
for idx in np.argsort(mean_shap)[::-1][:10]:
    print(f"  {ALL_FEATS[idx]:<35}  mean |SHAP| = {mean_shap[idx]:.5f}")

# Explain a specific flagged transaction
fraud_indices = np.where(y_test == 1)[0]
if len(fraud_indices) > 0:
    s = fraud_indices[0]
    sv_s = sv_fraud[s]
    print(f"\nExplanation for transaction #{s} (True Fraud, score={y_prob[s]:.4f}):")
    top = sorted(zip(ALL_FEATS, sv_s), key=lambda x: abs(x[1]), reverse=True)[:5]
    for feat, sv in top:
        direction = "↑ fraud" if sv > 0 else "↓ fraud"
        print(f"  {feat:<35}  SHAP={sv:+.4f}  {direction}")
```

---

## 7. Solution 2: Neural Network with Focal Loss

**What this code does:**
We build a deep neural network with 3 hidden layers. The key innovation is Focal Loss — instead of treating every transaction equally during training, Focal Loss reduces the weight of easy-to-classify examples (the 99% of obviously legitimate transactions). The model ends up spending most of its learning budget on the ambiguous, hard cases — which is exactly where fraud hides.

We also use a WeightedRandomSampler so that each mini-batch during training contains a roughly equal mix of fraud and legitimate transactions.

**Key terms explained:**
- **Focal Loss** — invented by the RetinaNet paper (Lin et al. 2017) for object detection imbalance. Formula: `FL = -alpha × (1-p)^gamma × log(p)`. The term `(1-p)^gamma` is the "focusing factor". For an easy negative (model is 99.9% confident it's legitimate, p≈0.001), this factor ≈ 0.001 — nearly zero loss contribution. For a hard case (model is uncertain, p≈0.5), factor ≈ 0.25 — still contributes significantly. This automatically shifts learning toward the difficult, uncertain examples.
  - `alpha` (default 0.75) — baseline weight for the positive (fraud) class. Higher alpha → more attention to fraud.
  - `gamma` (default 2.0) — focusing strength. gamma=0 is standard cross-entropy. gamma=2 is the recommended value for strong imbalance.
- **WeightedRandomSampler** — instead of uniform random sampling, each training example is sampled with probability proportional to its weight. Fraud examples get weight = `1/count_fraud`; legit examples get weight = `1/count_legit`. This creates approximately balanced mini-batches even from an imbalanced dataset.
- **BatchNorm (Batch Normalisation)** — normalises the activations of each layer within each mini-batch to have mean ≈ 0 and variance ≈ 1. Prevents the problem where different features (age in years vs amount in dollars) have wildly different scales propagating through the network. Allows higher learning rates.
- **Dropout** — during training, randomly sets a fraction of neuron outputs to zero on each forward pass. Forces the network to learn redundant representations. Prevents individual neurons from becoming indispensable (overfitting). Rate=0.4 means 40% of neurons are randomly silenced on each batch.
- **`BCEWithLogitsLoss`** — Binary Cross-Entropy loss that takes raw logits (before sigmoid) as input. More numerically stable than computing sigmoid first then using BCELoss.
- **`ReduceLROnPlateau`** — the learning rate scheduler that halves the learning rate when validation AUC-PR stops improving for `patience` epochs. Allows the model to make large steps early and small precise steps late in training.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Neural networks require feature scaling — unlike tree methods like LightGBM
# StandardScaler: subtract mean, divide by std → each feature has mean=0, std=1
scaler   = StandardScaler()
X_tr_s   = scaler.fit_transform(X_train).astype('float32')
X_te_s   = scaler.transform(X_test).astype('float32')

# Focal Loss: down-weight easy examples, focus on hard ones
class FocalLoss(nn.Module):
    """
    FL(p) = -alpha × (1-p)^gamma × log(p)
    
    For legitimate transactions: model quickly learns p(legit) ≈ 0.999
    → (1 - 0.999)^2 = 0.000001 → near-zero contribution to loss
    → model stops wasting learning capacity on easy negatives
    
    For ambiguous/fraud transactions: p is uncertain, closer to 0.5
    → (1 - 0.5)^2 = 0.25 → still contributes substantially to loss
    → model focuses its learning here
    """
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce     = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t     = torch.exp(-bce)                      # probability of correct class
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        weight  = alpha_t * (1 - p_t) ** self.gamma    # focusing factor
        return (weight * bce).mean()

# WeightedRandomSampler: each mini-batch has ~50% fraud, ~50% legit
# (instead of the natural 1% fraud, 99% legit)
class_counts  = np.bincount(y_train)
sample_weights= np.array([1.0/class_counts[y] for y in y_train])
sampler       = WeightedRandomSampler(
    torch.tensor(sample_weights, dtype=torch.double),
    num_samples=len(sample_weights), replacement=True
)

train_ds = TensorDataset(torch.tensor(X_tr_s), torch.tensor(y_train, dtype=torch.float32))
test_ds  = TensorDataset(torch.tensor(X_te_s), torch.tensor(y_test,  dtype=torch.float32))
train_ldr= DataLoader(train_ds, batch_size=512, sampler=sampler)
test_ldr = DataLoader(test_ds,  batch_size=1024, shuffle=False)

class FraudNet(nn.Module):
    """
    3 hidden layers: captures complex feature interactions
    BatchNorm after each layer: stabilises training on mixed-scale features
    Dropout 0.4: aggressive regularisation because fraud data is limited
    """
    def __init__(self, in_dim, hidden=(256, 128, 64), dropout=0.4):
        super().__init__()
        layers = []
        prev   = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x): return self.net(x)

model     = FraudNet(X_tr_s.shape[1]).to(device)
criterion = FocalLoss(alpha=0.75, gamma=2.0)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
# Reduce learning rate by half when AUC-PR stops improving for 5 epochs
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', patience=5, factor=0.5)

best_auc_pr = 0
best_state  = None
for epoch in range(50):
    model.train(); total_loss = 0
    for xb, yb in train_ldr:
        xb, yb = xb.to(device), yb.to(device).unsqueeze(1)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_ldr:
            probs = torch.sigmoid(model(xb.to(device))).cpu().squeeze()
            all_probs.extend(probs.numpy()); all_labels.extend(yb.numpy())

    auc_pr = average_precision_score(all_labels, all_probs)
    scheduler.step(auc_pr)
    if auc_pr > best_auc_pr:
        best_auc_pr = auc_pr
        best_state  = {k:v.clone() for k,v in model.state_dict().items()}
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1:3d}  loss={total_loss/len(train_ldr):.5f}  "
              f"AUC-PR={auc_pr:.4f}  LR={optimizer.param_groups[0]['lr']:.6f}")

model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    probs = torch.sigmoid(model(torch.tensor(X_te_s).to(device))).cpu().squeeze().numpy()
print(f"\n=== FraudNet (Focal Loss) ===")
print(f"AUC-ROC: {roc_auc_score(y_test, probs):.4f}")
print(f"AUC-PR:  {average_precision_score(y_test, probs):.4f}")
```

---

## 8. Solution 3: Isolation Forest + LightGBM Ensemble

**What this code does:**
We combine an unsupervised anomaly detector (Isolation Forest) with the supervised LightGBM. Isolation Forest doesn't use any labels — it learns what a "normal" transaction looks like and flags anomalies. Its output (an anomaly score) becomes an additional feature for LightGBM. This makes the system more robust to new fraud patterns that weren't in the training labels.

**Key terms explained:**
- **Isolation Forest** — a tree-based anomaly detection algorithm. It builds random trees that randomly split features. Anomalies (unusual transactions) are easier to isolate in fewer splits — they are "isolated" high up in the tree. Normal points require many splits to isolate. The **anomaly score** is the average path length — shorter path = more anomalous.
- **`decision_function`** — returns a raw score for each sample. Higher (less negative) = more normal. We negate it so higher = more anomalous, which is more intuitive.
- **`contamination`** — tells Isolation Forest what fraction of training data you expect to be anomalies. We set 0.01 (1%). This affects the threshold for labelling something an outlier, but the raw scores are unaffected.
- **Two-stage ensemble** — Stage 1 (unsupervised Isolation Forest) catches anything unusual even if it wasn't in the historical fraud labels. Stage 2 (supervised LightGBM) learns known fraud patterns from labels. Together they handle both known fraud patterns AND novel/evolving fraud.
- **Concept drift** — fraud patterns evolve. Fraudsters adapt their techniques when banks improve detection. A model trained solely on historical fraud may miss new attack types. Isolation Forest is unsupervised so it automatically detects anything statistically unusual, regardless of whether it matches past fraud patterns.

```python
import numpy as np
from sklearn.ensemble import IsolationForest
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score

# Stage 1: Isolation Forest — trained on ALL data (unsupervised, no labels used)
print("Training Isolation Forest (unsupervised anomaly detector)...")
iso = IsolationForest(
    n_estimators=200,
    max_samples=0.8,        # each tree sees 80% of training data
    contamination=0.01,     # expected anomaly fraction — affects threshold, not raw scores
    max_features=0.8,
    n_jobs=-1, random_state=42
)
iso.fit(X_train)

# Negate: Isolation Forest returns lower scores for anomalies
# We flip sign so: higher score = more anomalous (intuitive)
iso_train_scores = -iso.decision_function(X_train)
iso_test_scores  = -iso.decision_function(X_test)
print(f"Isolation Forest standalone AUC-ROC: {roc_auc_score(y_test, iso_test_scores):.4f}")
print(f"  (Weak alone but adds unique signal to LightGBM)")

# Stage 2: Add anomaly score as a feature to LightGBM
# The supervised model can now say "this transaction looks anomalous (IF says so)
# AND matches known fraud patterns (LightGBM knows from labels)"
X_train_aug = np.hstack([X_train, iso_train_scores.reshape(-1,1)])
X_test_aug  = np.hstack([X_test,  iso_test_scores.reshape(-1,1)])
feature_names_aug = ALL_FEATS + ['iso_anomaly_score']

lgb_ens = lgb.LGBMClassifier(
    objective='binary', n_estimators=1000, learning_rate=0.05,
    num_leaves=63, min_child_samples=20, subsample=0.8,
    colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
    scale_pos_weight=legit_ct/fraud_ct,
    n_jobs=-1, random_state=42, verbose=-1
)

from sklearn.model_selection import train_test_split as tts
Xt2, Xv2, yt2, yv2 = tts(X_train_aug, y_train, test_size=0.1, random_state=42, stratify=y_train)
lgb_ens.fit(Xt2, yt2, eval_set=[(Xv2,yv2)],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(100)])

y_prob_ens = lgb_ens.predict_proba(X_test_aug)[:, 1]
print(f"\n=== Isolation Forest + LightGBM Ensemble ===")
print(f"AUC-ROC: {roc_auc_score(y_test, y_prob_ens):.4f}")
print(f"AUC-PR:  {average_precision_score(y_test, y_prob_ens):.4f}")

# Verify that iso_anomaly_score contributes — check its SHAP rank
import shap
shap_ens = shap.TreeExplainer(lgb_ens).shap_values(X_test_aug[:200])
sv_fraud_ens  = shap_ens[1] if isinstance(shap_ens, list) else shap_ens
mean_sv       = np.abs(sv_fraud_ens).mean(axis=0)
iso_rank      = np.argsort(mean_sv)[::-1].tolist().index(len(ALL_FEATS))
print(f"\nIsolation Forest score SHAP rank: #{iso_rank+1} of {len(feature_names_aug)} features")
```

---

## 9. Evaluation Strategy

**What this code does:**
We demonstrate why standard accuracy is useless for fraud detection and show the correct evaluation metrics: AUC-PR, Recall at specific FPR targets, Lift curves, and a business impact calculation in dollars.

**Key terms explained:**
- **AUC-ROC vs AUC-PR** — AUC-ROC measures the model's ability to rank fraud above legitimate across all thresholds. It is misleadingly high for imbalanced problems (a random model gets 0.5, a model that flags everything as fraud gets ~0.5 too). AUC-PR focuses specifically on the positive (fraud) class — how precise are the flags, and how many fraud cases are caught? AUC-PR is our primary metric.
- **Recall @ FPR = 0.001** — "At the threshold where we block only 0.1% of legitimate transactions, what fraction of fraud do we catch?" This is the business-driven metric: the FPR budget is determined by how much customer friction is acceptable.
- **Lift curve** — if we score-rank all transactions and contact the top 1%, what fraction of that 1% is actually fraud? Compared to randomly selecting 1% of transactions (which would capture ~1% of fraud), lift tells us how much better the model is. Lift = 20 means "we catch 20× more fraud per investigation than random."
- **Business impact ($)** — translate model metrics into dollars. Each caught fraud saves $150. Each false positive costs $8. Net benefit = TP × $150 - FP × $8 - FN × $150.

```python
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve

def fraud_evaluation(y_true, y_prob, model_name='Model'):
    print(f"\n{'='*55}\n  {model_name}\n{'='*55}")

    # AUC-PR is the primary metric for imbalanced binary classification
    auc_roc = roc_auc_score(y_true, y_prob)
    auc_pr  = average_precision_score(y_true, y_prob)
    print(f"AUC-ROC: {auc_roc:.5f}  (can be misleadingly high for imbalance)")
    print(f"AUC-PR:  {auc_pr:.5f}  ← PRIMARY METRIC for fraud detection")

    # Recall at specific business FPR targets
    # FPR = False Positive Rate = fraction of legit transactions blocked
    fpr_arr, tpr_arr, _ = roc_curve(y_true, y_prob)
    print("\nRecall (fraud catch rate) at specific False Positive Rate budgets:")
    for target_fpr in [0.001, 0.005, 0.01]:
        idx    = np.searchsorted(fpr_arr, target_fpr)
        recall = tpr_arr[min(idx, len(tpr_arr)-1)]
        print(f"  Block {target_fpr:.1%} of legit → catch {recall:.2%} of fraud")

    # Lift: how many times better than random at top percentiles?
    df_l = pd.DataFrame({'y': y_true, 'p': y_prob}).sort_values('p', ascending=False).reset_index(drop=True)
    base = y_true.mean()  # baseline fraud rate (random selection)
    print("\nLift (how many times better than random selection):")
    for pct in [0.01, 0.05, 0.10, 0.20]:
        n_top   = max(1, int(len(df_l)*pct))
        caught  = df_l['y'][:n_top].sum()
        lift    = (caught/n_top) / base
        coverage= caught / y_true.sum()
        print(f"  Top {pct:.0%}: Lift={lift:.1f}×  Coverage={coverage:.1%}")

    # Business impact
    FRAUD_VALUE = 150   # $ saved per caught fraud
    FP_COST     = 8     # $ cost per wrongly blocked legit transaction
    # Find threshold that maximises net $benefit
    prec_arr, rec_arr, t_arr = precision_recall_curve(y_true, y_prob)
    best_benefit = -np.inf; best_t = 0.5
    n_fraud = y_true.sum(); n_legit = (y_true==0).sum()
    for p, r, t in zip(prec_arr, rec_arr, t_arr):
        if p == 0: continue
        tp_est   = r * n_fraud
        fp_est   = tp_est * (1-p) / max(p, 1e-10)
        benefit  = tp_est * FRAUD_VALUE - fp_est * FP_COST
        if benefit > best_benefit: best_benefit = benefit; best_t = t

    y_pred = (y_prob >= best_t).astype(int)
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print(f"\nBusiness Impact (optimal threshold={best_t:.4f}):")
    print(f"  Fraud caught (TP): {tp:5d}  → ${tp*FRAUD_VALUE:>10,.0f} saved")
    print(f"  Fraud missed (FN): {fn:5d}  → ${fn*FRAUD_VALUE:>10,.0f} lost")
    print(f"  Legit blocked (FP):{fp:5d}  → ${fp*FP_COST:>10,.0f} FP cost")
    print(f"  NET BENEFIT:              ${tp*FRAUD_VALUE - fn*FRAUD_VALUE - fp*FP_COST:>10,.0f}")

fraud_evaluation(y_test, y_prob, model_name='LightGBM')
```

---

## 10. Production Considerations

**What this code does:**
Shows the production architecture: a feature store for pre-computed velocity features, the decision logic (block/step-up/approve), concept drift monitoring, and weekly retraining triggers.

**Key terms explained:**
- **Feature store** — a centralised data system (typically Redis or a specialised platform) that stores pre-computed features for each user/card. Velocity features like "transactions in last 1 hour" cannot be computed from the transaction itself — they need historical context. The feature store makes these available in <1ms at inference time.
- **Step-up authentication** — instead of hard-blocking a transaction, send the cardholder an OTP (one-time password) or trigger 3DS (3D Secure) verification. Used for medium-risk transactions (scores between the soft and hard thresholds). Catches real fraud while not blocking legitimate transactions.
- **Concept drift** — fraud tactics evolve. Fraudsters change their behaviour when detection improves. A model that was excellent 6 months ago may degrade as fraud patterns shift. We detect this by monitoring the model's prediction score distribution — if the average fraud score suddenly drops, either fraud got smarter or the model degraded.

```python
import datetime
import numpy as np

print("""
Production Architecture:
═══════════════════════════════════════════════════════
Transaction arrives (JSON)
        ↓
Feature Store lookup (Redis, <1ms)
  → transactions_last_1h, amount_mean_30d, etc.
  These are pre-computed from a streaming pipeline
        ↓
Feature engineering (amount_zscore, velocity_ratio, etc.)  [<1ms]
        ↓
LightGBM inference → fraud_probability  [<0.5ms]
        ↓
Decision logic:
  prob ≥ 0.85  → BLOCK (auto-decline)
  prob ≥ 0.60  → STEP_UP (send OTP/3DS)
  prob < 0.60  → APPROVE
        ↓
Async logging → Kafka topic
  → Store {features, prediction, threshold, timestamp}
  → Confirmed fraud labels arrive later via chargebacks
        ↓
Weekly retraining pipeline:
  1. Fetch last 90 days labelled transactions
  2. Re-engineer all features
  3. Retrain LightGBM + Isolation Forest
  4. Shadow mode: run new model alongside old for 24h
  5. Compare AUC-PR: if new > old by >0.5%, promote to production
""")

class FraudMonitor:
    """Track prediction scores over time to detect model/fraud drift."""
    def __init__(self, window=10000):
        self.scores = []
        self.window = window

    def log(self, score: float):
        self.scores.append(score)
        if len(self.scores) > self.window * 2:
            self.scores = self.scores[-self.window*2:]

    def check_drift(self):
        if len(self.scores) < self.window * 2:
            return "Insufficient data"
        recent = np.mean(self.scores[-self.window:])
        hist   = np.mean(self.scores[-self.window*2:-self.window])
        shift  = abs(recent - hist) / hist
        if shift > 0.20:
            return f"⚠️  DRIFT: score shifted {shift:.1%} — retrain recommended"
        return f"✓ OK  (score shift: {shift:.1%})"

monitor = FraudMonitor(window=1000)
for score in y_prob[:2000]: monitor.log(float(score))
print(f"\nDrift check: {monitor.check_drift()}")
```

---

*Last updated: May 2026 | Pramod Modi — Staff ML Engineer, ServiceNow*
