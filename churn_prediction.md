# Use Case 3: Customer Churn Prediction
> Binary Classification on Temporal Data — plain-English explanations before every code block

---

## Table of Contents
1. [Problem Statement](#1-problem-statement)
2. [Dataset & Data Understanding](#2-dataset--data-understanding)
3. [Feature Engineering](#3-feature-engineering)
4. [Temporal Validation — Why Random Split Leaks](#4-temporal-validation--why-random-split-leaks)
5. [Model Selection — Why This, Not That](#5-model-selection--why-this-not-that)
6. [Solution 1: Logistic Regression Scorecard](#6-solution-1-logistic-regression-scorecard)
7. [Solution 2: XGBoost with SHAP Retention Actions](#7-solution-2-xgboost-with-shap-retention-actions)
8. [Solution 3: PyTorch MLP](#8-solution-3-pytorch-mlp)
9. [Evaluation Strategy](#9-evaluation-strategy)
10. [Production: Monthly Scoring Pipeline](#10-production-monthly-scoring-pipeline)

---

## 1. Problem Statement

A telecom company has **5 million customers**. Monthly churn rate is **5–8%**. Retaining a customer costs ~$30 (discount or retention call). Losing a customer costs ~$400 (in acquisition cost to replace them).

**Goal:** Score every customer monthly with a churn probability. Route highest-risk customers to retention agents with actionable reasons — which features drove their score.

**Constraints:**
- Batch scoring of 5M customers monthly (not real-time)
- Each prediction must include the top 3 reasons (for agent call scripts)
- Features must only use data available before the month being predicted (no future leakage)

---

## 2. Dataset & Data Understanding

**What this code does:**
We simulate 14 months of customer behaviour data for 20,000 customers. Each row is one customer in one observation month. Features from month M are used to predict whether the customer cancels in month M+1. We deliberately build temporal patterns into the data: usage declines over time, support calls increase, and churn probability goes up with each month.

**Key terms explained:**
- **Observation month** — the snapshot in time for which we have customer features. Each customer appears in multiple rows, one per month. Month 0 = the oldest data; month 13 = the most recent.
- **Label** — whether the customer cancelled in the NEXT month. Not in this month. This is the target we are predicting from the current month's features.
- **Sigmoid function** — converts a linear combination of features (which can range from -∞ to +∞) into a probability between 0 and 1. `p = 1/(1+e^(-logit))`. We use it to generate realistic churn probabilities from our simulated features.
- **Panel data** — data with both a cross-sectional dimension (many customers) and a time dimension (many months per customer). Each customer appears in multiple rows. This structure requires careful handling of temporal splits to avoid data leakage.

```python
import numpy as np
import pandas as pd

np.random.seed(42)
N, MONTHS = 20000, 14
rows = []

for cust_id in range(N):
    tenure = np.random.randint(1, 120)
    plan   = np.random.choice(['basic','standard','premium'], p=[0.3,0.45,0.25])

    for month in range(MONTHS - 1):
        base_usage = {'basic':2.0,'standard':5.0,'premium':9.0}[plan]
        usage_gb   = max(0, base_usage * (1 + np.random.normal(0,0.1)) * (1 - 0.02*month))
        call_min   = max(0, np.random.normal(180,60) * (1 - 0.01*month))
        support_ct = np.random.poisson(0.5 + 0.1*month)
        charge     = {'basic':29.99,'standard':59.99,'premium':89.99}[plan]
        late_pay   = int(np.random.binomial(1, 0.05 + 0.005*month))
        comp_promo = int(np.random.binomial(1, 0.1 + 0.01*month))
        contract_left = max(0, 12 - tenure + month + np.random.randint(-2,3))

        # True data-generating process for churn label
        churn_logit = (
            -2.5
            + 0.3  * support_ct                          # dissatisfaction
            - 0.5  * (usage_gb / base_usage)             # low usage = leaving
            + 0.4  * late_pay                            # payment stress
            + 0.5  * comp_promo                          # competitor attraction
            - 0.02 * tenure                              # loyalty effect
            + 0.3  * (contract_left < 2)                 # expiring contract
            + np.random.normal(0, 0.3)                   # random noise
        )
        churn_prob = 1 / (1 + np.exp(-churn_logit))     # sigmoid: convert to 0-1 probability
        churned    = int(np.random.binomial(1, churn_prob))

        rows.append({
            'customer_id':           cust_id,
            'observation_month':     month,
            'tenure_months':         tenure + month,
            'plan_type':             plan,
            'usage_gb':              round(usage_gb,2),
            'call_minutes_30d':      round(call_min,1),
            'support_calls_30d':     support_ct,
            'monthly_charge':        charge,
            'avg_bill_3m':           round(charge * (1+np.random.normal(0,0.05)),2),
            'late_payment_flag':     late_pay,
            'competitor_promo_seen': comp_promo,
            'contract_months_left':  contract_left,
            'churn_next_month':      churned
        })

df = pd.DataFrame(rows)
print(f"Dataset: {len(df):,} rows ({N:,} customers × {MONTHS-1} months)")
print(f"Overall churn rate: {df['churn_next_month'].mean():.2%}")
print(f"\nChurn rate by plan:")
print(df.groupby('plan_type')['churn_next_month'].mean().round(4))
print(f"\nChurn rate rises over time (as designed):")
print(df.groupby('observation_month')['churn_next_month'].mean().round(3))
```

---

## 3. Feature Engineering

**What this code does:**
We create features that capture **trends over time** — not just the current month's value but how that value has been changing. A customer whose usage is declining for 3 months in a row is at far higher risk than a customer with the same current usage who was stable.

**Key terms explained:**
- **Rolling average** — the average over the last N observations. For a customer with usage [10, 8, 6, 4] over 4 months, the 3-month rolling average at month 4 is (8+6+4)/3 = 6. This smooths out noise and captures the trend.
- **`transform(lambda x: x.rolling(3, min_periods=1).mean())`** — applies a rolling calculation per customer group. `.groupby('customer_id')` groups rows by customer first, then we compute the 3-month rolling mean per customer. `min_periods=1` means we compute even if fewer than 3 months of data are available (useful for new customers).
- **`diff()`** — compute month-over-month change. If usage was 9.0 last month and is 7.0 this month, diff = -2.0 (declining). Negative diff = usage trending down = churn risk signal.
- **Rolling sum** — cumulative count over a window. `support_calls_3m` = total support calls in the last 3 months. A customer making 8 support calls over 3 months is far more dissatisfied than one who made 1 call once.
- **`pd.cut` with labels** — creates categorical bins from a continuous variable. `tenure_bucket` converts tenure in months (1–120) into 5 ordinal groups (new customer, 1 year, 2 years, 3-5 years, loyal). Models often handle this cleaner than raw tenure values.
- **Engagement score** — a composite feature that combines multiple usage signals into one number. Normalising each component to [0,1] first ensures they contribute equally before combining.

```python
import pandas as pd
import numpy as np

def engineer_churn_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create trend-based features. Key insight for churn:
    The DIRECTION of change often matters more than the absolute value.
    A customer at usage_gb=3 who was at 8 three months ago is at higher
    risk than a customer at usage_gb=3 who has been stable for years.
    """
    df = df.copy().sort_values(['customer_id','observation_month'])

    plan_map = {'basic':0,'standard':1,'premium':2}
    df['plan_encoded'] = df['plan_type'].map(plan_map)

    # ── Usage trend features ──────────────────────────────────────────────
    # Per-customer mean usage (across all available months for that customer)
    cust_usage_mean  = df.groupby('customer_id')['usage_gb'].transform('mean')
    # Ratio: current usage / customer's own average
    # < 1.0 means below their own baseline; < 0.5 = strong churn signal
    df['usage_vs_mean']     = df['usage_gb'] / (cust_usage_mean + 1e-6)
    df['low_usage_flag']    = (df['usage_vs_mean'] < 0.5).astype(int)
    # Month-over-month change: negative = usage declining
    df['usage_mom_change']  = df.groupby('customer_id')['usage_gb'].diff().fillna(0)
    df['calls_mom_change']  = df.groupby('customer_id')['call_minutes_30d'].diff().fillna(0)
    # 3-month rolling average: smoothed trend (reduces single-month noise)
    df['usage_3m_avg']      = (df.groupby('customer_id')['usage_gb']
                                 .transform(lambda x: x.rolling(3, min_periods=1).mean()))
    # Trend = current value - 3-month average: positive=improving, negative=declining
    df['usage_trend']       = df['usage_gb'] - df['usage_3m_avg']

    # ── Support signals (dissatisfaction indicator) ───────────────────────
    df['support_calls_3m'] = (df.groupby('customer_id')['support_calls_30d']
                                .transform(lambda x: x.rolling(3, min_periods=1).sum()))
    df['support_spike']    = (df['support_calls_30d'] > 3).astype(int)

    # ── Financial stress signals ──────────────────────────────────────────
    df['bill_deviation']    = (df['monthly_charge'] - df['avg_bill_3m']).abs()
    df['payment_stress_3m'] = (df.groupby('customer_id')['late_payment_flag']
                                 .transform(lambda x: x.rolling(3, min_periods=1).sum()))
    df['high_bill_ratio']   = (df['monthly_charge'] > df['avg_bill_3m'] * 1.1).astype(int)

    # ── Contract and lifecycle signals ────────────────────────────────────
    df['contract_expiring']  = (df['contract_months_left'] <= 1).astype(int)
    df['contract_renewing']  = ((df['contract_months_left'] >= 1) &
                                (df['contract_months_left'] <= 3)).astype(int)
    # Bin tenure into 5 groups: new, 1yr, 2yr, 2-5yr, loyal
    df['tenure_bucket']      = pd.cut(df['tenure_months'], bins=[0,3,12,24,60,np.inf],
                                       labels=[0,1,2,3,4]).astype(int)
    df['is_new_customer']    = (df['tenure_months'] <= 3).astype(int)
    df['is_loyal_customer']  = (df['tenure_months'] >= 48).astype(int)

    # ── Competitive exposure ──────────────────────────────────────────────
    # 3-month cumulative competitor promotions seen
    df['competitor_3m_exposure'] = (df.groupby('customer_id')['competitor_promo_seen']
                                      .transform(lambda x: x.rolling(3, min_periods=1).sum()))

    # ── Composite engagement score ────────────────────────────────────────
    from sklearn.preprocessing import MinMaxScaler
    u_min, u_max = df['usage_gb'].min(), df['usage_gb'].max()
    c_min, c_max = df['call_minutes_30d'].min(), df['call_minutes_30d'].max()
    usage_norm   = (df['usage_gb'] - u_min) / (u_max - u_min + 1e-6)
    calls_norm   = (df['call_minutes_30d'] - c_min) / (c_max - c_min + 1e-6)
    # High usage + high calls + low support calls = engaged customer
    df['engagement_score'] = usage_norm + calls_norm - df['support_calls_30d'] * 0.1

    return df

df_fe = engineer_churn_features(df)

FEATURE_COLS = [
    'tenure_months','plan_encoded','usage_gb','call_minutes_30d',
    'support_calls_30d','monthly_charge','avg_bill_3m','late_payment_flag',
    'competitor_promo_seen','contract_months_left',
    # Engineered features:
    'usage_vs_mean','low_usage_flag','usage_mom_change','calls_mom_change',
    'usage_3m_avg','usage_trend','support_calls_3m','support_spike',
    'bill_deviation','payment_stress_3m','high_bill_ratio',
    'contract_expiring','contract_renewing','tenure_bucket',
    'is_new_customer','is_loyal_customer','competitor_3m_exposure','engagement_score'
]
print(f"Total features: {len(FEATURE_COLS)}  "
      f"(10 raw + {len(FEATURE_COLS)-10} engineered)")
```

---

## 4. Temporal Validation — Why Random Split Leaks

**What this code does:**
We demonstrate why splitting the data randomly into train/test is WRONG for time-series / panel data, and measure exactly how much it inflates the apparent model performance.

**Key terms explained:**
- **Data leakage** — information from the future accidentally flows into the training data. In a customer churn context: if a customer's month-12 data (future) is in training while their month-9 data (earlier) is in the test set, the model may use learned patterns from month 12 to "predict" month 9. This makes evaluation metrics look better than they actually are in production.
- **Temporal split** — train on all data from months 0–9, validate on month 10, test on months 11–12. This is the only honest evaluation: the model is always tested on data from a time period it has never seen.
- **Random k-fold** — standard cross-validation where data is randomly shuffled and split. Appropriate for i.i.d. (independent, identically distributed) data. WRONG for time-series because it allows future data to appear in training folds.
- **Optimism bias** — the tendency of random split evaluation to over-estimate real-world performance. Models that appear to achieve AUC=0.91 on a random split might only achieve AUC=0.83 in production. The difference (0.08 AUC here) is the "leakage inflation."

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

TRAIN_MONTHS = list(range(0, 10))
VAL_MONTH    = [10]
TEST_MONTHS  = [11, 12]

# Correct temporal split: train on past, test on future
train_mask = df_fe['observation_month'].isin(TRAIN_MONTHS)
val_mask   = df_fe['observation_month'].isin(VAL_MONTH)
test_mask  = df_fe['observation_month'].isin(TEST_MONTHS)

X_train = df_fe.loc[train_mask, FEATURE_COLS].values.astype('float32')
y_train = df_fe.loc[train_mask, 'churn_next_month'].values
X_val   = df_fe.loc[val_mask,   FEATURE_COLS].values.astype('float32')
y_val   = df_fe.loc[val_mask,   'churn_next_month'].values
X_test  = df_fe.loc[test_mask,  FEATURE_COLS].values.astype('float32')
y_test  = df_fe.loc[test_mask,  'churn_next_month'].values

print("Temporal split:")
print(f"  Train (months 0–9):   {X_train.shape[0]:,} rows  churn={y_train.mean():.2%}")
print(f"  Val   (month 10):     {X_val.shape[0]:,} rows  churn={y_val.mean():.2%}")
print(f"  Test  (months 11–12): {X_test.shape[0]:,} rows  churn={y_test.mean():.2%}")

# Prove the leakage: compare random split vs temporal split
gb = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)

# WRONG WAY: random split — mixes months 11,12 data into training
X_all = df_fe[FEATURE_COLS].values.astype('float32')
y_all = df_fe['churn_next_month'].values
Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
gb.fit(Xr_tr, yr_tr)
auc_random = roc_auc_score(yr_te, gb.predict_proba(Xr_te)[:,1])

# RIGHT WAY: temporal split
gb.fit(X_train, y_train)
auc_temporal = roc_auc_score(y_test, gb.predict_proba(X_test)[:,1])

print(f"\n⚠️  LEAKAGE DEMONSTRATION:")
print(f"  AUC with random split (OPTIMISTIC/WRONG):  {auc_random:.4f}")
print(f"  AUC with temporal split (HONEST/CORRECT):  {auc_temporal:.4f}")
print(f"  Leakage inflation: {(auc_random-auc_temporal)*100:.2f} AUC points")
print(f"\n  In production this model would achieve ~{auc_temporal:.2f}, not ~{auc_random:.2f}")
print(f"  The {(auc_random-auc_temporal)*100:.1f}% gap is the cost of using random split on temporal data")
```

---

## 5. Model Selection — Why This, Not That

| Algorithm | Verdict | Reason |
|---|---|---|
| **Logistic Regression** | ✅ Best baseline | Odds-ratio coefficients map directly to business actions ("each additional support call multiplies churn odds by 1.8×"). Regulatory environments require this level of interpretability. |
| **Decision Tree** | ❌ Avoid | High variance across months — the tree structure changes significantly depending on which months are in training. Unstable feature importances. |
| **Random Forest** | ⚠️ Good but not best | Biased importance toward high-cardinality features (tenure_months). SHAP values less sharp than XGBoost. |
| **XGBoost** | ✅ Best production | Top AUC. SHAP values per-customer map cleanly to retention actions. Native imbalance handling. |
| **MLP (Neural Net)** | ⚠️ Competitive | Can learn complex interactions. But harder to explain to retention agents. SHAP requires more computation. |
| **LSTM** | ❌ Overkill here | LSTMs process sequential data per-customer. Requires per-customer time series format, not cross-sectional monthly snapshots. Would be appropriate if we had per-click behavioural streams. |
| **Naive Bayes** | ❌ | Features are strongly correlated (usage, calls, tenure all move together). Independence assumption badly violated. |
| **SVM** | ❌ | Cannot scale to 5M customers for monthly batch scoring. |

---

## 6. Solution 1: Logistic Regression Scorecard

**What this code does:**
We train Logistic Regression and use its coefficients to build a business scorecard: a table showing which factors increase or decrease churn risk and by how much (as odds ratios). This is the model a compliance team or C-suite would ask for because every score is fully explainable by a formula.

**Key terms explained:**
- **Odds ratio** — `exp(coefficient)`. If the coefficient for `support_calls_30d` is 0.58, the odds ratio is `exp(0.58) ≈ 1.8`. This means: every additional support call in the last month multiplies the odds of churning by 1.8 — an 80% increase in churn odds. Business people understand odds ratios intuitively ("support calls increase churn risk by 80%").
- **Odds vs probability** — odds = p/(1-p). If churn probability is 20%, odds = 0.2/0.8 = 0.25. If odds ratio is 1.8, new odds = 0.25 × 1.8 = 0.45. New probability = 0.45/1.45 = 31%. Logistic regression models the log of the odds (log-odds = logit) as a linear function.
- **StandardScaler** — subtracts the mean and divides by the standard deviation for each feature. Required for Logistic Regression because it is sensitive to feature scales — a feature measured in dollars (monthly_charge ≈ 60) and one measured in hours (contract_months_left ≈ 8) need to be on comparable scales for the L2 regularisation to work correctly.
- **Risk tiers** — business decision: segment customers by predicted churn probability into actionable buckets. Critical (>75%): retention agent + best available offer. High (55–75%): retention agent call. Medium (35–55%): automated discount email. Low (<35%): no action. This prioritises limited retention budget toward highest-impact customers.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pandas as pd

# LR requires feature scaling — the regularisation penalty treats all features equally,
# so they must be on the same numerical scale
scaler      = StandardScaler()
X_tr_s      = scaler.fit_transform(X_train)
X_val_s     = scaler.transform(X_val)
X_te_s      = scaler.transform(X_test)

churn_rate  = y_train.mean()
cw          = compute_class_weight('balanced', classes=[0,1], y=y_train)
print(f"Churn rate: {churn_rate:.2%}  →  class weights: legit={cw[0]:.2f}, churn={cw[1]:.2f}")

lr = LogisticRegression(
    C=0.5,                 # moderate regularisation: prevents any single feature dominating
    solver='saga',         # efficient for large datasets with L2
    penalty='l2',
    max_iter=1000,
    class_weight={0:cw[0], 1:cw[1]},
    n_jobs=-1, random_state=42
)
lr.fit(X_tr_s, y_train)
y_prob_lr = lr.predict_proba(X_te_s)[:,1]

print(f"\n=== Solution 1: Logistic Regression Scorecard ===")
print(f"AUC-ROC: {roc_auc_score(y_test, y_prob_lr):.4f}")
print(f"AUC-PR:  {average_precision_score(y_test, y_prob_lr):.4f}")

# Build the odds-ratio scorecard — interpretable by business stakeholders
coef_df = pd.DataFrame({
    'feature':     FEATURE_COLS,
    'coefficient': lr.coef_[0],
    'odds_ratio':  np.exp(lr.coef_[0])  # exp(coef) = how many times churn odds multiply
}).sort_values('coefficient', key=abs, ascending=False)

print("\nChurn Risk Scorecard (sorted by strength of effect):")
print(f"{'Feature':<30} {'Odds Ratio':>10}  Interpretation")
print("-"*70)
for _, row in coef_df.head(12).iterrows():
    direction = "increases" if row['coefficient'] > 0 else "decreases"
    change    = abs(row['odds_ratio'] - 1) * 100
    print(f"{row['feature']:<30} {row['odds_ratio']:>10.3f}x  {direction} churn odds by {change:.0f}%")

# Segment into risk tiers for the retention team
test_df = df_fe[df_fe['observation_month'].isin(TEST_MONTHS)].copy()
test_df['churn_prob'] = y_prob_lr

def assign_tier(p):
    if p >= 0.75: return 'CRITICAL'
    if p >= 0.55: return 'HIGH'
    if p >= 0.35: return 'MEDIUM'
    return 'LOW'

test_df['risk_tier'] = test_df['churn_prob'].apply(assign_tier)
print("\nRisk tier summary:")
print(test_df.groupby('risk_tier', observed=True).agg(
    count=('churn_next_month','size'),
    actual_churn_rate=('churn_next_month','mean'),
    avg_model_score=('churn_prob','mean')
).round(4).sort_values('avg_model_score', ascending=False))
```

---

## 7. Solution 2: XGBoost with SHAP Retention Actions

**What this code does:**
We train XGBoost for maximum accuracy, then use SHAP to translate the top 3 model-driving features for each high-risk customer into specific retention actions. The output is a CRM-ready brief: "This customer is at high churn risk because of [declining usage + recent support calls + contract expiring]. Recommended action: [data plan upgrade + service credit + early renewal offer]."

**Key terms explained:**
- **`eval_metric='aucpr'`** — tells XGBoost to monitor AUC-PR (not accuracy or log-loss) when deciding whether to stop early. AUC-PR is the right metric for imbalanced binary classification.
- **`min_child_weight`** — minimum sum of instance weights in a leaf node. Setting to 10 prevents the model from creating tiny leaves that memorise individual churn examples from months with unusual patterns.
- **SHAP value sign** — positive SHAP = this feature's value is pushing the prediction TOWARD churn. Negative SHAP = pushing AWAY from churn. Example: `usage_trend = -2.5` with SHAP = +0.3 means "declining usage trend is contributing +0.3 to this customer's churn score."
- **Retention action mapping** — a lookup dictionary that translates each feature name into a human-readable action for the retention agent. This bridges the gap between "the model says SHAP(`support_calls_30d`) = +0.4" and "the agent should say: 'I see you've needed to call us a few times recently — let me make sure everything is resolved.'"

```python
import xgboost as xgb
import shap
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

churn_rate   = y_train.mean()
scale_pos_w  = (1 - churn_rate) / churn_rate

xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='aucpr',           # optimise AUC-PR — better for imbalanced data
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.05,
    reg_lambda=1.0,
    min_child_weight=10,           # prevent overfitting tiny churn clusters in specific months
    scale_pos_weight=scale_pos_w,  # up-weight churn examples
    early_stopping_rounds=50,
    random_state=42, n_jobs=-1
)
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)

y_prob_xgb = xgb_model.predict_proba(X_test)[:,1]
print(f"AUC-ROC: {roc_auc_score(y_test, y_prob_xgb):.4f}")
print(f"AUC-PR:  {average_precision_score(y_test, y_prob_xgb):.4f}")
print(f"Best iteration: {xgb_model.best_iteration}")

# SHAP: compute for all test customers
explainer   = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)   # shape: (n_test, n_features)

mean_shap = np.abs(shap_values).mean(axis=0)
print("\nTop 10 features by mean |SHAP|:")
for idx in np.argsort(mean_shap)[::-1][:10]:
    print(f"  {FEATURE_COLS[idx]:<30}  mean|SHAP|={mean_shap[idx]:.5f}")

# Map features to retention actions
# This is the bridge between model output and agent action
RETENTION_ACTIONS = {
    'usage_trend':            "Offer bonus data or additional minutes package",
    'low_usage_flag':         "Proactive usage review call — ensure customer knows all features",
    'usage_mom_change':       "Check for any service quality issues in the area",
    'usage_vs_mean':          "Engage with personalised usage tips and tutorial",
    'support_calls_30d':      "Escalate to senior support — ensure all issues fully resolved",
    'support_calls_3m':       "Schedule customer satisfaction review — offer service credit",
    'support_spike':          "Assign dedicated support contact for next 30 days",
    'contract_expiring':      "Initiate proactive renewal conversation with loyalty discount",
    'contract_months_left':   "Share new plan options and renewal incentives",
    'late_payment_flag':      "Offer flexible payment schedule or bill smoothing plan",
    'payment_stress_3m':      "Discuss financial hardship support options",
    'competitor_promo_seen':  "Match competitor pricing or highlight unique value",
    'competitor_3m_exposure': "Send competitive comparison and exclusive loyalty offer",
    'engagement_score':       "Re-engagement campaign with personalised feature showcase",
    'tenure_months':          "Acknowledge loyalty milestone with reward or upgrade",
    'is_loyal_customer':      "VIP treatment — dedicated account manager",
}

def generate_retention_brief(idx, X_test, shap_values, probs, n_actions=3):
    """
    For a single customer, generate an agent script using SHAP-driven reasons.
    Each of the top SHAP features maps to a specific retention action.
    """
    prob   = probs[idx]
    sv     = shap_values[idx]
    # Sort features by SHAP value magnitude (most influential first)
    # Only take features with POSITIVE SHAP (driving churn UP, not down)
    top3   = sorted([(FEATURE_COLS[i], sv[i], X_test[idx,i])
                     for i in range(len(FEATURE_COLS)) if sv[i] > 0],
                    key=lambda x: -x[1])[:n_actions]
    tier   = "CRITICAL" if prob>0.75 else "HIGH" if prob>0.55 else "MEDIUM"

    brief  = f"\n{'─'*58}\n"
    brief += f"  CUSTOMER RETENTION BRIEF  |  {tier}  |  Score: {prob:.1%}\n"
    brief += f"{'─'*58}\n"
    brief += "  Detected Risk Factors & Recommended Actions:\n\n"
    for rank, (feat, sv_val, feat_val) in enumerate(top3, 1):
        action = RETENTION_ACTIONS.get(feat, "Discuss general service satisfaction")
        brief += f"  {rank}. {feat} = {feat_val:.2f}  (SHAP contribution: +{sv_val:.3f})\n"
        brief += f"     → ACTION: {action}\n\n"
    return brief

# Show for top 3 highest-risk customers
for idx in np.argsort(y_prob_xgb)[::-1][:3]:
    print(generate_retention_brief(idx, X_test, shap_values, y_prob_xgb))
```

---

## 8. Solution 3: PyTorch MLP

**What this code does:**
We build a 3-layer neural network (Multilayer Perceptron) as our third solution. It uses GELU activations, BatchNorm, Dropout, and the OneCycleLR learning rate schedule. The key advantage over LR is the ability to learn complex non-linear interactions between features (e.g., the interaction between contract_expiring AND high competitor exposure is more dangerous than either alone).

**Key terms explained:**
- **GELU (Gaussian Error Linear Unit)** — a smoother alternative to ReLU activation. `GELU(x) = x × Φ(x)` where Φ is the cumulative normal distribution. Instead of hard-setting negative inputs to zero (ReLU), GELU smoothly weighs them near zero. Works slightly better than ReLU for tabular data.
- **Kaiming initialisation** — a weight initialisation strategy designed for ReLU-family activations. Sets initial weights from a normal distribution with variance scaled to prevent activations from exploding or vanishing in deep networks.
- **`BCEWithLogitsLoss(pos_weight=...)`** — binary cross-entropy that takes raw logits (before sigmoid) as input and up-weights positive-class (churn) errors. `pos_weight = count(non-churn) / count(churn)`. Numerically more stable than applying sigmoid first.
- **OneCycleLR** — a learning rate schedule that starts low, ramps up to a maximum rate (annealing phase), then decreases. Invented by Leslie Smith. Often converges faster and to better solutions than fixed learning rates. The idea: high learning rates explore the loss landscape broadly; low rates fine-tune.
- **`WeightedRandomSampler`** — same concept as in fraud detection. Creates mini-batches where churn and non-churn examples appear in equal proportions, despite the natural ~7% churn rate in the dataset.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sc       = StandardScaler()
X_tr_s   = sc.fit_transform(X_train).astype('float32')
X_val_s  = sc.transform(X_val).astype('float32')
X_te_s   = sc.transform(X_test).astype('float32')

# Weighted sampler: ensure ~50% churn in each mini-batch
churn_rate    = y_train.mean()
sample_wts    = np.where(y_train==1, 1/churn_rate, 1/(1-churn_rate))
sampler       = torch.utils.data.WeightedRandomSampler(
    torch.tensor(sample_wts), len(sample_wts), replacement=True)

train_ldr = DataLoader(
    TensorDataset(torch.tensor(X_tr_s), torch.tensor(y_train, dtype=torch.float32)),
    batch_size=512, sampler=sampler)
val_ldr   = DataLoader(
    TensorDataset(torch.tensor(X_val_s), torch.tensor(y_val, dtype=torch.float32)),
    batch_size=1024, shuffle=False)
test_ldr  = DataLoader(
    TensorDataset(torch.tensor(X_te_s), torch.tensor(y_test, dtype=torch.float32)),
    batch_size=1024, shuffle=False)

class ChurnMLP(nn.Module):
    """
    Architecture:
      - 3 hidden layers: enough depth to learn feature interactions
      - BatchNorm: normalises activations, allows higher learning rates
      - GELU: smoother than ReLU, works well for tabular data
      - Dropout 0.3: prevents overfitting to specific months in training
    """
    def __init__(self, in_dim, hidden=(256,128,64), dropout=0.3):
        super().__init__()
        layers = []
        prev   = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        # Kaiming init: correct variance scaling for GELU activations
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x): return self.net(x)

model     = ChurnMLP(len(FEATURE_COLS)).to(device)
# pos_weight counteracts imbalance: non-churn count / churn count
pos_w     = torch.tensor([(1-churn_rate)/churn_rate]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
# OneCycleLR: ramp up to max_lr then anneal down — often finds better minima
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=5e-3, steps_per_epoch=len(train_ldr), epochs=30)

def evaluate(loader):
    model.eval(); probs, labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            p = torch.sigmoid(model(xb.to(device))).cpu().squeeze()
            probs.extend(p.numpy()); labels.extend(yb.numpy())
    return np.array(probs), np.array(labels)

best_auc_pr = 0; best_state = None
for epoch in range(30):
    model.train(); total_loss = 0
    for xb, yb in train_ldr:
        xb, yb = xb.to(device), yb.to(device).unsqueeze(1)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); scheduler.step()
        total_loss += loss.item()

    vp, vl = evaluate(val_ldr)
    auc_pr = average_precision_score(vl, vp)
    if auc_pr > best_auc_pr:
        best_auc_pr = auc_pr
        best_state  = {k:v.clone() for k,v in model.state_dict().items()}
    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1:3d}  loss={total_loss/len(train_ldr):.5f}  "
              f"val_AUC-PR={auc_pr:.4f}  LR={scheduler.get_last_lr()[0]:.6f}")

model.load_state_dict(best_state)
tp, tl = evaluate(test_ldr)
print(f"\n=== PyTorch ChurnMLP ===")
print(f"AUC-ROC: {roc_auc_score(tl, tp):.4f}")
print(f"AUC-PR:  {average_precision_score(tl, tp):.4f}")
```

---

## 9. Evaluation Strategy

**What this code does:**
We evaluate churn models using metrics that directly reflect business value: lift curves (how efficiently do we target retention spend?) and ROI analysis (how many dollars does the model save vs random targeting?).

**Key terms explained:**
- **Lift** — if we target the top 10% of customers by predicted churn score, what fraction of those are actual churners, compared to randomly selecting 10%? Lift = (precision in top 10%) / (baseline churn rate). Lift of 4 means: "by targeting our model's top 10%, we catch 4× more churners than random."
- **Coverage** — of all customers who will actually churn, what fraction do we capture by targeting the top X%? Coverage = TP / total_actual_churners.
- **Calibration** — does the model's predicted probability match the actual observed churn rate? If the model predicts 30% churn probability for 1000 customers, roughly 300 of them should actually churn. Poor calibration means the scores are not trustworthy as probabilities (even if ranking is good).
- **ROI (Return on Investment)** — how much money does the model save per retention dollar spent? If we spend $30 to retain each contacted customer, and we contact the top 10%, our ROI = (churners caught × $400 value) - (correctly retained × $30 cost) - (non-churners contacted × $30 wasted cost).

```python
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from sklearn.calibration import calibration_curve

def churn_evaluation(y_true, y_prob, model_name='Model', top_pct=0.10):
    print(f"\n{'='*60}\n  {model_name}\n{'='*60}")
    print(f"AUC-ROC: {roc_auc_score(y_true, y_prob):.4f}")
    print(f"AUC-PR:  {average_precision_score(y_true, y_prob):.4f}")

    # Lift curve: sort by predicted probability, measure precision and coverage in top %
    df_lift = pd.DataFrame({'y':y_true,'p':y_prob}).sort_values('p',ascending=False).reset_index(drop=True)
    total_churn  = y_true.sum()
    baseline     = y_true.mean()
    print(f"\nBaseline churn rate: {baseline:.2%}")
    print(f"\nLift Curve:")
    print(f"  {'Top %':>8}  {'Precision':>10}  {'Coverage':>10}  {'Lift':>8}")
    for pct in [0.01, 0.02, 0.05, 0.10, 0.20]:
        n_top     = max(1, int(len(df_lift)*pct))
        caught    = df_lift['y'][:n_top].sum()
        precision = caught / n_top
        coverage  = caught / total_churn
        lift      = precision / baseline
        print(f"  {pct:>8.0%}  {precision:>10.2%}  {coverage:>10.2%}  {lift:>8.1f}×")

    # ROI analysis
    LOST_VALUE    = 400   # $ value of retaining one churner
    RETENTION_COST= 30    # $ cost to contact and make an offer
    n_contacted   = int(len(df_lift) * top_pct)
    true_churners = df_lift['y'][:n_contacted].sum()
    false_pos     = n_contacted - true_churners
    saved         = true_churners * (LOST_VALUE - RETENTION_COST)
    wasted        = false_pos * RETENTION_COST
    net_roi       = saved - wasted

    print(f"\nROI Analysis (targeting top {top_pct:.0%}):")
    print(f"  Customers contacted: {n_contacted:,}")
    print(f"  True churners saved: {true_churners:,}  → ${saved:,.0f}")
    print(f"  False positives:     {false_pos:,}  → ${wasted:,.0f} wasted")
    print(f"  NET ROI:                         ${net_roi:,.0f}")
    print(f"  ROI per retention $:             ${net_roi/(n_contacted*RETENTION_COST):.2f}")

    # Calibration check: do predicted probabilities match actual rates?
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=8)
    print(f"\nCalibration check:")
    print(f"  {'Predicted':>10}  {'Actual':>10}  {'Diff':>8}")
    for mp, fp in zip(mean_pred, frac_pos):
        diff = fp - mp
        flag = "⚠️" if abs(diff) > 0.05 else "✓"
        print(f"  {mp:>10.3f}  {fp:>10.3f}  {diff:>+8.3f} {flag}")

churn_evaluation(y_test, y_prob_xgb, model_name='XGBoost')
churn_evaluation(y_test, y_prob_lr,  model_name='Logistic Regression')
```

---

## 10. Production: Monthly Scoring Pipeline

**What this code does:**
Shows the complete production flow: take a fresh monthly customer snapshot, run feature engineering, score all customers, assign risk tiers, attach SHAP-based reasons, and output a CRM-ready file.

**Key terms explained:**
- **Batch inference** — score all 5 million customers at once (not one at a time as requests come in). Runs as a scheduled job at the start of each month.
- **CRM integration** — the output (customer_id + risk_tier + top_3_reasons) is written to the CRM system (e.g., Salesforce, ServiceNow) so retention agents see the churn score and action suggestions when they open a customer record.
- **Pipeline reproducibility** — saving the scaler alongside the model ensures that the exact same feature transformation is applied at scoring time as was applied during training. A mismatch here would silently produce incorrect predictions.

```python
import datetime
import numpy as np
import pandas as pd
import joblib

def monthly_scoring_pipeline(current_month_df: pd.DataFrame,
                              model, scaler, explainer,
                              feature_cols: list, top_pct=0.10):
    """
    End-to-end monthly batch scoring for all customers.
    Input: raw customer data for the observation month
    Output: scored DataFrame with risk tiers and SHAP reasons
    """
    # Step 1: Feature engineering
    df_scored = engineer_churn_features(current_month_df)
    X = df_scored[feature_cols].values.astype('float32')

    # Step 2: Score all customers
    X_scaled = scaler.transform(X)    # use the SAME scaler fitted on training data
    probs    = model.predict_proba(X_scaled)[:,1]
    df_scored['churn_probability'] = probs

    # Step 3: Risk tiers
    def tier(p):
        if p >= 0.75: return 'CRITICAL'
        if p >= 0.55: return 'HIGH'
        if p >= 0.35: return 'MEDIUM'
        return 'LOW'
    df_scored['risk_tier']  = df_scored['churn_probability'].apply(tier)
    df_scored['score_date'] = datetime.date.today().isoformat()

    # Step 4: SHAP reasons for CRITICAL and HIGH tier customers only
    # (Computing SHAP for all 5M customers would be too slow)
    high_risk_mask = df_scored['risk_tier'].isin(['CRITICAL','HIGH'])
    if high_risk_mask.sum() > 0:
        sv     = explainer.shap_values(X_scaled[high_risk_mask])
        reasons = []
        for row_sv in sv:
            # Top 3 positive SHAP features (driving churn up)
            top3 = sorted(
                [(feature_cols[i], row_sv[i]) for i in range(len(feature_cols)) if row_sv[i] > 0],
                key=lambda x: -x[1])[:3]
            reasons.append(' | '.join([f"{f}(+{v:.3f})" for f,v in top3]))
        df_scored.loc[high_risk_mask, 'top_reasons'] = reasons
    else:
        df_scored['top_reasons'] = ''

    # Summary
    tier_counts = df_scored['risk_tier'].value_counts()
    print(f"\nMonthly Scoring Summary  {datetime.date.today()}")
    print(f"  Total customers scored: {len(df_scored):,}")
    for t, c in tier_counts.items():
        print(f"  {t:<10}: {c:>8,} ({c/len(df_scored):.1%})")

    return df_scored[['customer_id','churn_probability','risk_tier','score_date','top_reasons']]

# Run on last month's data as a demo
last_month_data = df[df['observation_month'] == 12].copy()
scored = monthly_scoring_pipeline(
    last_month_data, lr, scaler,
    shap.TreeExplainer(xgb_model),
    FEATURE_COLS
)
print("\nSample CRM output (top 5 high-risk customers):")
print(scored[scored['risk_tier']=='CRITICAL'].head(5)[
    ['customer_id','churn_probability','risk_tier','top_reasons']
].to_string(index=False))
```

---

*Last updated: May 2026 | Pramod Modi — Staff ML Engineer, ServiceNow*
