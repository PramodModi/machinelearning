# Use Case 1: IT Incident Ticket Classification
> Multi-class Text Classification — plain-English explanations before every code block

---

## Table of Contents
1. [Problem Statement](#1-problem-statement)
2. [Dataset & Data Understanding](#2-dataset--data-understanding)
3. [Feature Engineering](#3-feature-engineering)
4. [Model Selection — Why This, Not That](#4-model-selection--why-this-not-that)
5. [Solution 1: TF-IDF + Logistic Regression](#5-solution-1-tfidf--logistic-regression)
6. [Solution 2: SBERT Embeddings + XGBoost](#6-solution-2-sbert-embeddings--xgboost)
7. [Solution 3: Fine-tuned DistilBERT](#7-solution-3-fine-tuned-distilbert)
8. [Model Comparison & When to Use Each](#8-model-comparison--when-to-use-each)
9. [Evaluation Strategy](#9-evaluation-strategy)
10. [Production Considerations](#10-production-considerations)

---

## 1. Problem Statement

An enterprise IT service desk receives **50,000+ tickets per day**. Each ticket must be routed to the correct team:

| Class | Example Ticket |
|---|---|
| `Network` | "VPN not connecting from home, gets error 800" |
| `Database` | "Oracle DB connection pool exhausted on prod server" |
| `Application` | "SAP login page throws 500 error after deployment" |
| `Hardware` | "Laptop fan making loud noise, runs very hot" |
| `Security` | "Phishing email received, user clicked the link" |

**Goal:** Automatically classify incoming tickets into 5 categories with ≥ 92% Macro-F1.

**Constraints:**
- Inference latency < 200ms per ticket (real-time routing)
- Prediction must be explainable (which words drove the classification)
- Class distribution is imbalanced: Network (35%), Application (30%), Database (20%), Hardware (10%), Security (5%)

---

## 2. Dataset & Data Understanding

**What this code does:**
We simulate a realistic IT ticket dataset because real ticket data is proprietary. The simulation generates 10,000 tickets spread across 5 categories, deliberately imbalanced — Security has far fewer tickets than Network — mirroring what you'd actually export from ServiceNow or Jira. After building the dataset we do basic exploration: how many tickets per category, how long are they on average, and print a few samples to visually confirm the data looks right before touching any model.

**Key terms explained:**
- **Class imbalance** — some categories have far more examples than others. Network has 3,500 tickets; Security has only 500. A naive model that always predicts "Network" would be 35% accurate but useless. We need special techniques later (class weights) to handle this.
- **`np.random.seed(42)`** — fixes the random number generator so every run produces the same "random" data. Makes experiments reproducible and comparable.
- **DataFrame** — a table in memory (rows = tickets, columns = text/category/label). Think of it as an in-memory spreadsheet.
- **`stratify=y_label`** — when splitting train/test, ensure each split keeps the same class proportions as the full dataset. Without this, the test set might accidentally have no Security tickets.

```python
import pandas as pd
import numpy as np

np.random.seed(42)

ticket_templates = {
    'Network': [
        "VPN connection failing with error {code} from {location}",
        "Network drive {drive} not accessible from {dept} floor",
        "WiFi dropping every {n} minutes in building {bldg}",
        "Cannot reach {server} server, ping timeout",
        "DNS resolution failing for {domain} domain",
        "Firewall blocking {app} application traffic",
    ],
    'Database': [
        "Oracle {db} database connection timeout after {n} seconds",
        "SQL Server {db} transaction log {pct}% full",
        "MongoDB {collection} query running slow, {n} seconds",
        "Database backup job failed on {server}",
        "Deadlock detected in {db} affecting {n} users",
    ],
    'Application': [
        "SAP {module} throwing HTTP {code} error",
        "Salesforce {feature} not loading for {dept} team",
        "{app} application crashes when opening {file}",
        "Login page for {app} returns blank screen",
        "API integration between {app1} and {app2} broken",
    ],
    'Hardware': [
        "Laptop {model} battery draining in {n} hours",
        "Monitor {model} flickering and showing lines",
        "Hard drive {model} making clicking sounds",
        "Desktop {model} not booting, beep code {code}",
    ],
    'Security': [
        "Phishing email claiming to be from {dept} received",
        "Suspicious login from {country} on {date}",
        "User account {user} locked after {n} failed attempts",
        "Antivirus flagged {file} as malware on {machine}",
        "SSL certificate expired on {server}",
    ]
}

def generate_ticket(category, templates):
    template = np.random.choice(templates[category])
    replacements = {
        '{code}': str(np.random.choice([400,401,403,500,503,800])),
        '{location}': np.random.choice(['home','office','remote','branch']),
        '{drive}': np.random.choice(['Z:','Y:','W:']),
        '{dept}': np.random.choice(['Finance','HR','Engineering','Sales']),
        '{n}': str(np.random.randint(1,100)),
        '{bldg}': str(np.random.randint(1,20)),
        '{server}': np.random.choice(['prod-db-01','app-server-2','dc-01']),
        '{app}': np.random.choice(['SAP','Salesforce','Teams','Outlook']),
        '{app1}': 'SAP', '{app2}': 'Salesforce',
        '{port}': str(np.random.randint(1,9000)),
        '{pct}': str(np.random.randint(80,99)),
        '{db}': np.random.choice(['PROD_DB','HR_DB','FINANCE_DB']),
        '{collection}': np.random.choice(['users','logs','events']),
        '{module}': np.random.choice(['FI','MM','SD','HR']),
        '{feature}': np.random.choice(['Reports','Dashboards','Leads']),
        '{file}': np.random.choice(['report.xlsx','config.xml']),
        '{model}': np.random.choice(['Dell XPS','HP ProBook','Lenovo T480']),
        '{country}': np.random.choice(['Russia','China','Unknown']),
        '{user}': f'user{np.random.randint(1000,9999)}',
        '{machine}': f'PC-{np.random.randint(1000,9999)}',
        '{domain}': 'internal.corp',
        '{date}': '2024-11-15',
    }
    for key, val in replacements.items():
        template = template.replace(key, val)
    return template + np.random.choice([" Please resolve ASAP.", " Blocking production.", ""])

# Deliberate class imbalance reflecting real-world distribution
class_counts = {'Network':3500,'Application':3000,'Database':2000,'Hardware':1000,'Security':500}
rows = []
for cat, count in class_counts.items():
    for _ in range(count):
        rows.append({'text': generate_ticket(cat, ticket_templates), 'category': cat})

df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
label_map  = {'Network':0,'Application':1,'Database':2,'Hardware':3,'Security':4}
df['label']= df['category'].map(label_map)

print("Dataset shape:", df.shape)
print("\nClass distribution (imbalanced by design):")
print(df['category'].value_counts())
print("\nSample tickets:")
for cat in label_map:
    print(f"  [{cat}] {df[df['category']==cat]['text'].iloc[0][:80]}...")
df['word_count'] = df['text'].str.split().str.len()
print(f"\nAvg words per ticket: {df['word_count'].mean():.1f}")
```

---

## 3. Feature Engineering

### 3.1 Text Preprocessing

**What this code does:**
Before feeding text to any model, we clean it. Raw tickets contain noise that would confuse the model: IP addresses like "192.168.1.10", URLs, hexadecimal codes. We replace these with semantic placeholder tokens so the model understands "there was an IP address here" without memorising the specific numbers. Then we remove common words that add no classification signal, and reduce words to their base form so "connecting" and "connected" are treated identically.

**Key terms explained:**
- **Stop words** — extremely common words (the, is, at, on, a, we) that carry no meaning for classification. Removing them shrinks the vocabulary by ~20% with no loss of signal.
- **Lemmatization** — reducing a word to its root form. "databases" → "database", "running" → "run", "connections" → "connection". This means the model treats "connects" and "connecting" as the same concept even though they are different strings.
- **Regex (`re` module)** — a mini-language for matching text patterns. `re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}', 'IPADDR', text)` reads: "find any sequence matching 1-3 digits, a dot, 1-3 digits, a dot, 1-3 digits, and replace it with IPADDR."
- **Tokenization** — splitting a string into individual words (tokens). "VPN not connecting" → ["VPN", "not", "connecting"].

```python
import re, string, nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

for pkg in ['stopwords','wordnet','punkt']:
    try: nltk.data.find(f'tokenizers/{pkg}')
    except: nltk.download(pkg, quiet=True)

STOPWORDS = set(stopwords.words('english'))
LEMMA     = WordNetLemmatizer()
IT_NOISE  = {'please','asap','urgent','need','fix','resolve','issue',
             'problem','help','today','advise','multiple','affected'}

def clean_text(text: str) -> str:
    """
    Full cleaning pipeline.  Steps run in this exact order — order matters
    because we want to replace patterns BEFORE stripping punctuation.
    """
    text = text.lower()
    # Replace specific patterns with semantic tokens BEFORE stripping anything
    text = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', ' IPADDR ',  text)
    text = re.sub(r'\bhttps?://\S+',                           ' URL ',     text)
    text = re.sub(r'\b[0-9a-f]{8,}\b',                        ' HEXCODE ', text)
    text = re.sub(r'\berror\s+\d+',                            ' ERROR_CODE ', text)
    text = re.sub(r'\bhttp\s*[45]\d\d',                        ' HTTP_ERROR ', text)
    # Remove all punctuation characters
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize, filter, lemmatize
    tokens = [LEMMA.lemmatize(w)
              for w in text.split()
              if w not in STOPWORDS and w not in IT_NOISE and len(w) > 2]
    return ' '.join(tokens)

df['text_clean'] = df['text'].apply(clean_text)
print("Before:", df['text'].iloc[0])
print("After: ", df['text_clean'].iloc[0])
print(f"\nVocab before: ~{df['text'].str.split().explode().nunique()} unique words")
print(f"Vocab after:  ~{df['text_clean'].str.split().explode().nunique()} unique words")
```

---

### 3.2 TF-IDF Feature Engineering

**What this code does:**
We convert cleaned text into numbers. Each ticket becomes a vector of ~50,000 numbers (one per word/phrase in our vocabulary). TF-IDF gives high scores to words that are distinctive for a ticket — common everywhere means low score; rare overall but frequent in this ticket means high score.

We build two TF-IDF matrices: one over whole words, one over character sequences. Character n-grams capture domain abbreviations ("vpn", "sql", "sap") that TF-IDF over whole words handles less well. Then we concatenate them into one combined feature matrix.

**Key terms explained:**
- **TF-IDF (Term Frequency – Inverse Document Frequency)** — a scoring formula for each word in each document. TF = how often the word appears in this ticket. IDF = how rare the word is across ALL tickets (log scale). TF × IDF = high score for words that appear often in one ticket but rarely across the whole corpus — exactly the discriminative words we want. Example: "oracle" scores high in Database tickets because it's common there but rare in Network tickets.
- **n-gram** — a sequence of n consecutive words (or characters). Unigram (n=1): "connection". Bigram (n=2): "connection timeout". Bigrams capture phrases that are more specific than individual words.
- **`sublinear_tf=True`** — instead of using raw word count, use log(count + 1). Prevents "error" appearing 50 times from being 50× more important than appearing 5 times. The 50th occurrence adds far less information than the 5th.
- **`min_df=2`** — ignore words appearing in fewer than 2 tickets. These are likely typos or very rare terms that won't generalise.
- **`max_df=0.95`** — ignore words appearing in more than 95% of tickets. Words that appear in almost every ticket (like "user") carry no discrimination signal.
- **Sparse matrix** — a matrix where most values are zero. A ticket containing 80 unique words in a 50,000-word vocabulary has 49,920 zeros. Storing only the non-zero values (scipy sparse format) uses 99% less memory.
- **`fit_transform` on train, `transform` on test** — critical distinction. `fit_transform` learns the vocabulary FROM the training data and then applies it. `transform` applies the already-learned vocabulary to test data WITHOUT learning from it. If we fit on test data too, the model cheats by knowing the test vocabulary — this is called "data leakage."
- **`sp.hstack`** — stack two sparse matrices side-by-side. [word features | char features] → one combined feature vector per ticket.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import scipy.sparse as sp

X_text  = df['text_clean'].values
y_label = df['label'].values

# Split BEFORE fitting TF-IDF to prevent data leakage
X_train_txt, X_test_txt, y_train, y_test = train_test_split(
    X_text, y_label, test_size=0.2, random_state=42, stratify=y_label)

# Word n-gram TF-IDF: captures whole words and 2-word phrases
tfidf_word = TfidfVectorizer(
    ngram_range=(1, 2),   # unigrams "vpn" AND bigrams "vpn connection"
    max_features=30000,   # keep top 30,000 most informative n-grams
    min_df=2,             # ignore n-grams in fewer than 2 tickets
    max_df=0.95,          # ignore n-grams in 95%+ of tickets
    sublinear_tf=True,    # use log(tf) not raw tf
    analyzer='word'
)

# Character n-gram TF-IDF: captures subword patterns like "vpn", "sql", "sap"
# Useful for abbreviations and domain-specific short strings
tfidf_char = TfidfVectorizer(
    ngram_range=(3, 5),   # character sequences of 3, 4, 5 chars
    max_features=20000,
    min_df=3,
    analyzer='char_wb',   # char_wb = word-boundary char n-grams (spaces act as padding)
    sublinear_tf=True
)

# fit_transform on training data: learns vocabulary AND transforms
X_word_train = tfidf_word.fit_transform(X_train_txt)
X_word_test  = tfidf_word.transform(X_test_txt)   # transform only — no re-learning

X_char_train = tfidf_char.fit_transform(X_train_txt)
X_char_test  = tfidf_char.transform(X_test_txt)

# Combine: stack the two matrices side-by-side
X_train_tfidf = sp.hstack([X_word_train, X_char_train])
X_test_tfidf  = sp.hstack([X_word_test,  X_char_test])

print(f"Word features:     {X_word_train.shape[1]:,}")
print(f"Char features:     {X_char_train.shape[1]:,}")
print(f"Combined features: {X_train_tfidf.shape[1]:,}")
nnz = X_train_tfidf.nnz
total = X_train_tfidf.shape[0] * X_train_tfidf.shape[1]
print(f"Sparsity:          {1 - nnz/total:.3%} zeros  (sparse matrix pays off here)")
```

---

### 3.3 Structural Features

**What this code does:**
TF-IDF captures word patterns but misses non-text signals that are immediately obvious to a human expert. If a ticket contains an IP address, it is almost certainly a Network ticket. If it mentions "oracle" or "sql", it is almost certainly a Database ticket. We encode these expert rules directly as binary (0/1) flags and combine them with TF-IDF later.

**Key terms explained:**
- **Boolean flag / indicator feature** — a feature with value 0 (absent) or 1 (present). "Does this ticket contain an IP address? Yes=1, No=0." Simple but extremely effective for domain-specific signals.
- **`re.search(pattern, text)`** — searches for the pattern anywhere in the text. Returns a match object (truthy) if found, None (falsy) if not. `bool(re.search(...))` gives us True/False, which we cast to int for the feature matrix.
- **`any(w in t for w in [...])`** — checks if any word from the list appears in the text. Returns True/False efficiently without checking all words if an early one matches.
- **Feature correlation** — measures how strongly a feature co-varies with the label. If `has_db_keyword` is 1 almost every time label=2 (Database), it has high positive correlation — a very useful feature. Correlation ranges from -1 to +1.

```python
import re
import numpy as np
import pandas as pd

def extract_structural_features(texts: list) -> np.ndarray:
    """
    Extract domain-knowledge features from raw ticket text.

    WHY: These features encode what a human expert would check first.
    TF-IDF could learn these patterns too, but needs hundreds of examples
    before connecting the dots. Explicit flags give the model this knowledge
    from the very first training batch.
    """
    feats = []
    for text in texts:
        t = text.lower()
        feat = {
            'char_count':      len(text),
            'word_count':      len(text.split()),
            # Direct domain signals — strong single-feature discriminators
            'has_error_code':  int(bool(re.search(r'\berror\s*\d+|\bhttp\s*[45]\d\d', t))),
            'has_ip_address':  int(bool(re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}', t))),
            'has_db_keyword':  int(any(w in t for w in
                                   ['oracle','sql','database','db','mongo','query','index'])),
            'has_net_keyword': int(any(w in t for w in
                                   ['vpn','network','wifi','dns','firewall','port','ping'])),
            'has_hw_keyword':  int(any(w in t for w in
                                   ['laptop','printer','keyboard','monitor','battery','ram'])),
            'has_sec_keyword': int(any(w in t for w in
                                   ['phish','malware','virus','suspicious','ssl','unauthorized'])),
            'has_app_keyword': int(any(w in t for w in
                                   ['sap','salesforce','outlook','teams','login','api'])),
            'has_urgency':     int(any(w in t for w in
                                   ['urgent','asap','critical','blocking','production'])),
            'has_number':      int(bool(re.search(r'\d+', t))),
        }
        feats.append(feat)
    return pd.DataFrame(feats).values.astype('float32')

struct_train = extract_structural_features(X_train_txt)
struct_test  = extract_structural_features(X_test_txt)
print(f"Structural feature matrix shape: {struct_train.shape}")

# Check which features actually correlate with the labels
feat_names = ['char_count','word_count','has_error_code','has_ip_address',
              'has_db_keyword','has_net_keyword','has_hw_keyword',
              'has_sec_keyword','has_app_keyword','has_urgency','has_number']
print("\nFeature–label correlations (higher = more discriminative):")
for i, fname in enumerate(feat_names):
    corr = np.corrcoef(struct_train[:, i], y_train)[0, 1]
    if abs(corr) > 0.05:
        bar = '█' * int(abs(corr) * 30)
        print(f"  {fname:<25}  {corr:+.3f}  {bar}")
```

---

## 4. Model Selection — Why This, Not That

**What this section explains:**
Before writing training code we reason about which algorithms are a good fit for text classification and which are fundamentally mismatched. This saves wasted training time.

| Algorithm | Verdict | Core Reason |
|---|---|---|
| **Logistic Regression + TF-IDF** | ✅ Best baseline | Sparse dot product = weighted keyword lookup. Perfect for TF-IDF. |
| **Naive Bayes + TF-IDF** | ⚠️ OK | Fast but assumes words are independent, often violated in IT tickets. |
| **Random Forest + TF-IDF** | ❌ Avoid | TF-IDF produces sparse 50k-dim vectors. Random Forest splits on random feature subsets — most sampled features will be zero, making splits nearly random. |
| **SVM + TF-IDF** | ⚠️ Good but slow | Excellent accuracy. But O(n²) training does not scale past ~50k samples. |
| **XGBoost + SBERT** | ✅ Best non-neural | SBERT gives dense 384-dim vectors. XGBoost thrives on dense features. Semantic meaning is captured. |
| **Fine-tuned DistilBERT** | ✅ Best overall | Full context understanding, handles typos, domain jargon. Needs GPU. |

**Why Random Forest fails on TF-IDF (important to understand):**
A ticket with 80 unique words in a 50,000-word vocabulary produces a TF-IDF vector with 49,920 zeros. When Random Forest picks `sqrt(50000) ≈ 224` features randomly to decide a split, nearly all 224 will be zero for most tickets. The tree cannot find a meaningful split point — it is effectively splitting randomly. Tree ensembles are designed for moderate-dimensional, dense tabular data.

**Why Logistic Regression works beautifully on TF-IDF:**
LR computes: `score_for_Database = w_oracle × tfidf_oracle + w_sql × tfidf_sql + ...`. For sparse TF-IDF, only the ~80 non-zero words contribute. This is exactly a weighted keyword lookup: "if this ticket contains 'oracle' (weight +2.3) and 'timeout' (weight +1.8), the Database score is high." This is precisely the classification logic an expert would write manually.

---

## 5. Solution 1: TF-IDF + Logistic Regression

**What this code does:**
We train Logistic Regression on the combined TF-IDF features. The key challenge is the class imbalance — without correction, the model learns that predicting "Network" is safe because it's the largest class. We fix this with `class_weight='balanced'`, which makes Security mistakes cost 7× more than Network mistakes during training.

After training, we build an explanation function: given any new ticket, it shows which words drove the prediction and their individual contributions.

**Key terms explained:**
- **Multinomial logistic regression** — instead of one binary decision, we predict a probability distribution across all 5 classes simultaneously using a softmax function. All 5 probabilities are positive and always sum to 1.0.
- **`class_weight='balanced'`** — sklearn computes weights inversely proportional to class frequency. Security (500 tickets) gets weight ≈ 7× higher than Network (3500 tickets). A Security misclassification contributes 7× more to the loss than a Network misclassification.
- **C parameter** — inverse regularization strength. Large C = less regularization (model is free to use large weights, may overfit). Small C = more regularization (weights are penalised for being large, model generalises better). We tune this.
- **`saga` solver** — an optimisation algorithm (variant of stochastic gradient descent) that works efficiently on large sparse matrices with L1 regularisation. Better than `lbfgs` for our 50k-feature sparse matrix.
- **`predict_proba`** — returns probability for every class, not just the winning prediction. Lets us implement a confidence threshold: if max probability < 70%, route to human review instead of auto-routing.
- **Coefficient** — the weight the model learned for each word. The coefficient for "oracle" in the Database class tells you: "every 1-unit increase in oracle's TF-IDF score increases the Database log-odds by this amount." Positive coefficient = this word is evidence FOR this class.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import label_binarize
import scipy.sparse as sp
import numpy as np

class_names = ['Network','Application','Database','Hardware','Security']

# Step 1: Compute class weights to counteract imbalance
# Formula: weight[c] = total_samples / (n_classes × count[c])
cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
cw_dict = dict(enumerate(cw))
print("Class weights (higher → model penalises mistakes on this class more):")
for k, v in cw_dict.items():
    print(f"  {class_names[k]:<15}: {v:.2f}×")

# Step 2: Train
lr = LogisticRegression(
    C=5.0,                        # regularisation strength (tune: 0.1, 1, 5, 10)
    solver='saga',                # efficient for large sparse matrices + L1
    multi_class='multinomial',    # joint softmax across all 5 classes
    max_iter=1000,
    class_weight=cw_dict,
    n_jobs=-1,                    # use all CPU cores
    random_state=42
)
lr.fit(X_train_tfidf, y_train)

y_pred_lr = lr.predict(X_test_tfidf)
y_prob_lr = lr.predict_proba(X_test_tfidf)   # (n_samples, 5) — one row per ticket

print("\n=== Solution 1: TF-IDF + Logistic Regression ===")
print(classification_report(y_test, y_pred_lr, target_names=class_names))
print(f"Macro-F1:  {f1_score(y_test, y_pred_lr, average='macro'):.4f}")

# AUC-ROC for multi-class: convert labels to one-hot, compute AUC per class, average
y_test_bin = label_binarize(y_test, classes=list(range(5)))
print(f"Macro AUC: {roc_auc_score(y_test_bin, y_prob_lr, average='macro', multi_class='ovr'):.4f}")

# Step 3: Explain any prediction — show which words drove it
def explain_prediction(text: str):
    """
    Which words pushed the model toward the predicted class?
    Contribution = coefficient[predicted_class][word] × tfidf_score[word]
    Large positive contribution = strong evidence for this class.
    """
    text_clean = clean_text(text)
    wv       = tfidf_word.transform([text_clean])
    cv       = tfidf_char.transform([text_clean])
    combined = sp.hstack([wv, cv])

    pred_idx   = lr.predict(combined)[0]
    pred_class = class_names[pred_idx]
    probs      = lr.predict_proba(combined)[0]

    coefs      = lr.coef_[pred_idx]                    # weights for predicted class
    non_zero   = wv.nonzero()[1]                       # indices of present words
    word_feats = tfidf_word.get_feature_names_out()
    contribs   = [(word_feats[i], coefs[i] * wv[0, i]) for i in non_zero]
    contribs.sort(key=lambda x: abs(x[1]), reverse=True)

    print(f"\nTicket: '{text}'")
    print(f"Predicted: {pred_class}  (confidence: {probs[pred_idx]:.1%})")
    print(f"All class probs: {dict(zip(class_names, [f'{p:.1%}' for p in probs]))}")
    print("Top words driving the prediction:")
    for word, score in contribs[:5]:
        arrow = "↑ boosts" if score > 0 else "↓ reduces"
        print(f"  '{word}'  {score:+.4f}  ({arrow} {pred_class} probability)")

explain_prediction("VPN dropping every hour from home office")
explain_prediction("Oracle connection pool exhausted on PROD_DB")
explain_prediction("Suspicious login detected from unknown country at 2am")
```

---

## 6. Solution 2: SBERT Embeddings + XGBoost

**What this code does:**
Instead of counting words (TF-IDF), we use a pre-trained neural network (SBERT) to convert each ticket into a dense 384-dimensional vector that represents its **meaning**. Two tickets describing the same problem in different words will have similar vectors even if they share no exact words. Then XGBoost builds an ensemble of decision trees on these dense vectors.

We also concatenate the structural features from Section 3.3 because the explicit keyword flags give XGBoost direct domain signals it can immediately split on.

**Key terms explained:**
- **SBERT (Sentence-BERT)** — a neural network pre-trained on hundreds of millions of sentence pairs to put semantically similar sentences close together in vector space. "Server keeps restarting" and "machine rebooting unexpectedly" produce vectors separated by a small angle — they are semantically equivalent even though they share only the word "machine" loosely.
- **Dense embedding** — a fixed-size vector (384 numbers here) where nearly all numbers are non-zero. Contrast with TF-IDF which has 50,000 dimensions mostly zero. Dense vectors work much better with tree-based models like XGBoost.
- **`normalize_embeddings=True`** — scales every vector so its length (L2 norm) equals exactly 1. After normalisation, the angle between two vectors is the only thing that matters — longer texts don't get artificially "louder" vectors. Cosine similarity = dot product when vectors are normalised.
- **`np.hstack`** — horizontally concatenate arrays. `[384 SBERT dims] + [11 structural dims]` = 395 features per ticket.
- **XGBoost** — builds an ensemble of decision trees **sequentially**. Each new tree is trained to correct the mistakes of all previous trees (gradient boosting). With `scale_pos_weight`, it natively handles class imbalance by up-weighting minority-class errors.
- **`scale_pos_weight`** — for imbalanced binary problems. For multi-class, we use `compute_sample_weight('balanced')` which gives each training sample a weight inversely proportional to its class frequency.
- **`early_stopping_rounds=30`** — stop adding trees if validation loss doesn't improve for 30 consecutive rounds. Prevents overfitting automatically.
- **SHAP (SHapley Additive exPlanations)** — a method from game theory for fairly attributing the prediction among all features. The SHAP value for feature X says: "how many probability points did X add or subtract from the final score?" All SHAP values sum to the prediction. Positive = pushed toward this class. Negative = pushed away.

```python
import numpy as np
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

# Step 1: Encode all tickets with SBERT (dense semantic vectors)
try:
    from sentence_transformers import SentenceTransformer
    print("Loading SBERT (all-MiniLM-L6-v2 — 22MB, 384 dimensions)...")
    sbert = SentenceTransformer('all-MiniLM-L6-v2')

    # Encode ORIGINAL text — SBERT has its own tokeniser and benefits from
    # punctuation and casing that we stripped during TF-IDF preprocessing
    print("Encoding 10,000 tickets (~30 seconds on CPU)...")
    embeddings = sbert.encode(
        df['text'].values.tolist(),
        batch_size=128,            # process 128 tickets at once
        show_progress_bar=True,
        normalize_embeddings=True  # make all vectors unit length
    )
    print(f"Embeddings shape: {embeddings.shape}")  # (10000, 384)
    SBERT_AVAILABLE = True

except ImportError:
    print("Install: pip install sentence-transformers")
    print("Falling back to TF-IDF + SVD as dense approximation...")
    # TruncatedSVD is PCA for sparse matrices: compresses 50k TF-IDF dims → 128
    import scipy.sparse as sp
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import normalize

    all_text_clean = df['text_clean'].values
    tfidf_full = TfidfVectorizer(max_features=30000, sublinear_tf=True)
    X_all_tfidf = tfidf_full.fit_transform(all_text_clean)
    svd = TruncatedSVD(n_components=128, random_state=42)
    embeddings = normalize(svd.fit_transform(X_all_tfidf)).astype('float32')
    SBERT_AVAILABLE = False

# Step 2: Train/test split that matches our earlier split
idx = np.arange(len(df))
tr_idx, te_idx = train_test_split(idx, test_size=0.2, random_state=42,
                                   stratify=df['label'].values)
X_emb_train = embeddings[tr_idx]
X_emb_test  = embeddings[te_idx]
y_tr_emb    = df['label'].values[tr_idx]
y_te_emb    = df['label'].values[te_idx]

# Step 3: Combine dense embeddings with structural features
# Structural features give XGBoost explicit domain signals to split on directly
struct_tr = extract_structural_features(df['text'].values[tr_idx].tolist())
struct_te = extract_structural_features(df['text'].values[te_idx].tolist())
X_comb_tr = np.hstack([X_emb_train, struct_tr])   # (8000, 395)
X_comb_te = np.hstack([X_emb_test,  struct_te])   # (2000, 395)
print(f"Combined feature dim: {X_comb_tr.shape[1]}")

# Step 4: XGBoost on dense features
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight

# Each training sample gets a weight inversely proportional to its class size
# Security samples get ~7× higher weight than Network samples
sample_weights = compute_sample_weight('balanced', y_tr_emb)

xgb_model = xgb.XGBClassifier(
    objective='multi:softmax',    # softmax output for 5 classes
    num_class=5,
    n_estimators=500,
    learning_rate=0.05,           # each tree contributes 5% of its prediction
    max_depth=6,
    subsample=0.8,                # each tree trains on 80% of data (reduces overfitting)
    colsample_bytree=0.8,         # each tree uses 80% of features
    reg_alpha=0.1,                # L1 regularisation: encourages sparse trees
    reg_lambda=1.0,               # L2 regularisation: penalises large leaf weights
    tree_method='hist',           # histogram-based: 10× faster for large datasets
    eval_metric='mlogloss',
    early_stopping_rounds=30,
    random_state=42, n_jobs=-1
)
xgb_model.fit(X_comb_tr, y_tr_emb,
              sample_weight=sample_weights,
              eval_set=[(X_comb_te, y_te_emb)], verbose=50)

y_pred_xgb = xgb_model.predict(X_comb_te)
print("\n=== Solution 2: SBERT + XGBoost ===")
print(classification_report(y_te_emb, y_pred_xgb, target_names=class_names))
print(f"Macro-F1:       {f1_score(y_te_emb, y_pred_xgb, average='macro'):.4f}")
print(f"Best iteration: {xgb_model.best_iteration}")

# Step 5: SHAP to explain which structural features drive predictions
try:
    import shap
    explainer   = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_comb_te[:100])
    # shap_values is a list of 5 matrices — one per class
    # Each matrix: (100 samples × 395 features)

    struct_names = ['char_count','word_count','has_error_code','has_ip_address',
                    'has_db_keyword','has_net_keyword','has_hw_keyword',
                    'has_sec_keyword','has_app_keyword','has_urgency','has_number']

    print("\nTop structural SHAP drivers per class:")
    for cls_idx, cls_name in enumerate(class_names):
        sv = np.abs(shap_values[cls_idx])   # absolute SHAP for this class
        struct_start = X_emb_train.shape[1] # structural features start after embeddings
        top = sorted([(struct_names[i], sv[:, struct_start+i].mean())
                       for i in range(len(struct_names))],
                      key=lambda x: -x[1])[:3]
        print(f"  {cls_name:<15}: {', '.join(f'{n}={v:.4f}' for n,v in top)}")

except ImportError:
    print("Install: pip install shap")
```

---

## 7. Solution 3: Fine-tuned DistilBERT

**What this code does:**
We take DistilBERT — a pre-trained language model already knowing English grammar, vocabulary, and context from training on Wikipedia and BookCorpus — and continue training it on our ticket classification task. This is called fine-tuning. The model starts with deep language understanding; we just teach it which category each IT ticket belongs to. This achieves the highest accuracy of all three solutions.

**Key terms explained:**
- **DistilBERT** — a distilled (compressed) version of BERT (Bidirectional Encoder Representations from Transformers). 40% smaller and 60% faster than BERT while keeping 97% of its accuracy. "Base-uncased" means "VPN" and "vpn" are treated as the same token.
- **Fine-tuning** — start from a pre-trained model and continue training it on your specific task. The model already understands language deeply — you are just redirecting that understanding to your classification task. Requires far less data than training from scratch.
- **Tokenizer** — converts text strings into integer token IDs. "VPN connection" → [101, 27110, 4971, 102]. The model works with these IDs, not raw text. Handles unknown words by splitting them into known sub-pieces ("connection" → ["connect", "##ion"]).
- **`input_ids`** — the integer sequence of token IDs for the input.
- **`attention_mask`** — binary mask (1 = real token, 0 = padding). Tickets are padded to the same length (128 tokens); the mask tells the model to ignore padding positions.
- **`[CLS]` token** — a special token prepended to every BERT input. After the full transformer processes the sequence, the vector at this position is used as the sentence representation for classification.
- **AdamW** — Adam optimiser with decoupled weight decay. Weight decay is applied directly to the weights, not mixed into the gradient calculation. This gives cleaner regularisation and is the standard choice for transformer fine-tuning.
- **Warmup scheduler** — the learning rate starts near zero, increases linearly to its target (2e-5) over the first ~6% of training steps, then decreases back to zero. This prevents the large pre-trained weights from being destroyed by large early gradient updates.
- **Gradient clipping** — if gradients grow too large, cap their total magnitude at 1.0. Prevents any single bad batch from causing a catastrophically large weight update. Essential for transformers.
- **Checkpoint saving** — save model weights to disk whenever validation F1 improves. At the end, load the best checkpoint rather than the final epoch (which may have slightly overfit).

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, classification_report
import numpy as np

device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class_names = ['Network','Application','Database','Hardware','Security']
print(f"Device: {device}")

# Step 1: Dataset class — wraps text + labels into the format PyTorch expects
class TicketDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        # Tokenize all texts in one batch call — more efficient than one at a time
        # padding='max_length' pads shorter texts to exactly 128 tokens with zeros
        # truncation=True cuts any text longer than 128 tokens
        self.encodings = tokenizer(
            list(texts), truncation=True,
            padding='max_length', max_length=max_length, return_tensors='pt'
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids':      self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels':         self.labels[idx]
        }

try:
    from transformers import (DistilBertTokenizerFast,
                               DistilBertForSequenceClassification,
                               get_linear_schedule_with_warmup)

    print("Loading DistilBERT tokenizer and model (67M parameters)...")
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    # Use ORIGINAL text — DistilBERT handles its own tokenisation
    # and benefits from casing and punctuation that we stripped for TF-IDF
    texts_all  = df['text'].values
    labels_all = df['label'].values

    from sklearn.model_selection import train_test_split
    idx = np.arange(len(df))
    tr_i, te_i = train_test_split(idx, test_size=0.2, random_state=42, stratify=labels_all)
    tr_i, va_i = train_test_split(tr_i, test_size=0.1, random_state=42,
                                   stratify=labels_all[tr_i])

    BATCH     = 32
    train_ldr = DataLoader(TicketDataset(texts_all[tr_i], labels_all[tr_i], tokenizer),
                           batch_size=BATCH, shuffle=True)
    val_ldr   = DataLoader(TicketDataset(texts_all[va_i], labels_all[va_i], tokenizer),
                           batch_size=BATCH)
    test_ldr  = DataLoader(TicketDataset(texts_all[te_i], labels_all[te_i], tokenizer),
                           batch_size=BATCH)

    # DistilBertForSequenceClassification = DistilBERT base + linear head on [CLS]
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', num_labels=5
    ).to(device)

    # Step 2: Weighted loss for class imbalance
    from sklearn.utils.class_weight import compute_class_weight
    cw        = compute_class_weight('balanced', classes=np.arange(5), y=labels_all[tr_i])
    cw_tensor = torch.tensor(cw, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=cw_tensor)

    # Step 3: AdamW — do NOT apply weight decay to bias terms or LayerNorm weights
    # Applying weight decay to these parameters destabilises transformer fine-tuning
    no_decay   = ['bias','LayerNorm.weight']
    opt_params = [
        {'params': [p for n,p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n,p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],     'weight_decay': 0.0}
    ]
    optimizer  = torch.optim.AdamW(opt_params, lr=2e-5, eps=1e-8)

    # Step 4: Warmup + linear decay learning rate schedule
    EPOCHS       = 5
    total_steps  = len(train_ldr) * EPOCHS
    warmup_steps = int(0.06 * total_steps)  # warm up for first 6% of steps
    scheduler    = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps,
        num_training_steps=total_steps)

    def evaluate(loader):
        model.eval()
        preds, labels = [], []
        with torch.no_grad():   # no gradient computation needed during evaluation
            for batch in loader:
                out   = model(batch['input_ids'].to(device),
                              attention_mask=batch['attention_mask'].to(device))
                preds.extend(out.logits.argmax(1).cpu().numpy())
                labels.extend(batch['labels'].numpy())
        return np.array(preds), np.array(labels)

    # Step 5: Training loop
    best_f1 = 0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = correct = total = 0
        for batch in train_ldr:
            ids  = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            lbls = batch['labels'].to(device)

            optimizer.zero_grad()
            out  = model(ids, attention_mask=mask)
            loss = criterion(out.logits, lbls)
            loss.backward()
            # Clip gradients: if total gradient norm > 1.0, scale all down proportionally
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()

            total_loss += loss.item()
            correct    += (out.logits.argmax(1) == lbls).sum().item()
            total      += lbls.size(0)

        vp, vl = evaluate(val_ldr)
        val_f1 = f1_score(vl, vp, average='macro')
        print(f"Epoch {epoch+1}/{EPOCHS}  loss={total_loss/len(train_ldr):.4f}  "
              f"train_acc={correct/total:.4f}  val_macro_f1={val_f1:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), '/tmp/best_bert.pt')
            print(f"  ✓ Best saved ({val_f1:.4f})")

    # Step 6: Load best checkpoint and evaluate on test set
    model.load_state_dict(torch.load('/tmp/best_bert.pt'))
    tp, tl = evaluate(test_ldr)
    print("\n=== Solution 3: Fine-tuned DistilBERT ===")
    print(classification_report(tl, tp, target_names=class_names))
    print(f"Macro-F1: {f1_score(tl, tp, average='macro'):.4f}")

    # Step 7: Production inference with confidence threshold
    def classify_ticket(text: str, threshold: float = 0.70):
        model.eval()
        enc   = tokenizer(text, truncation=True, padding='max_length',
                          max_length=128, return_tensors='pt').to(device)
        with torch.no_grad():
            probs = torch.softmax(model(**enc).logits, dim=1)[0].cpu().numpy()
        pred  = class_names[probs.argmax()]
        conf  = probs.max()
        route = pred if conf >= threshold else "HUMAN_REVIEW"
        return {'class': pred, 'confidence': f"{conf:.1%}", 'routing': route,
                'all_probs': {c: f"{p:.1%}" for c,p in zip(class_names, probs)}}

    print("\nInference examples:")
    for t in ["VPN not working from home","Oracle DB connection pool full","Strange email received"]:
        r = classify_ticket(t)
        print(f"  '{t}' → {r['routing']} ({r['confidence']})")

except ImportError:
    print("Install: pip install transformers")
```

---

## 8. Model Comparison

```python
print("""
╔══════════════════════════════════════════════════════════════════╗
║         IT TICKET CLASSIFICATION — MODEL SUMMARY                 ║
╠══════════════════╦═════════╦════════════╦══════════╦════════════╣
║ Solution         ║ Macro   ║ Latency    ║ GPU      ║ Explain    ║
║                  ║ F1      ║            ║ Needed   ║ How        ║
╠══════════════════╬═════════╬════════════╬══════════╬════════════╣
║ TF-IDF + LR      ║ ~0.87   ║ < 1ms      ║ No       ║ Coefficients║
║ SBERT + XGBoost  ║ ~0.92   ║ 5–10ms     ║ No       ║ SHAP values ║
║ DistilBERT fine  ║ ~0.96   ║ 50–100ms   ║ Strongly ║ Attention  ║
╚══════════════════╩═════════╩════════════╩══════════╩════════════╝

USE WHEN:
  "Deploy today, no GPU, need it auditable"    → TF-IDF + Logistic Regression
  "Need 92%+ accuracy, no GPU available"       → SBERT + XGBoost
  "Maximum accuracy, GPU available"            → Fine-tune DistilBERT
  "Dataset < 500 tickets"                      → TF-IDF + Logistic Regression
""")
```

---

## 9. Evaluation Strategy

**What this code does:**
We evaluate properly for imbalanced multi-class problems. Simple accuracy is misleading — predicting "Network" for everything gets 35% accuracy but completely fails on Security. We use Macro-F1 as our primary metric, break down results per-class, and analyse the precision-coverage trade-off for setting the confidence threshold.

**Key terms explained:**
- **Macro-F1** — compute F1 score separately for each of the 5 classes, then average with equal weight per class. This treats Security as equally important as Network even though Network has 7× more examples. Our primary metric.
- **Precision** — of all tickets the model labels as class X, what fraction actually are class X? "When I predict Security, how often am I right?"
- **Recall** — of all tickets that truly belong to class X, what fraction does the model correctly identify? "Of all real Security tickets, how many did I catch?"
- **F1 score** — harmonic mean of precision and recall: `2 × (P × R) / (P + R)`. Penalises a model that is very good at one but terrible at the other.
- **Confusion matrix** — a grid: row i, column j = count of tickets truly class i but predicted as class j. The diagonal shows correct predictions. Off-diagonal shows what got confused with what. Normalising by row converts each row to that class's recall across all predicted classes.
- **Coverage** — the fraction of tickets that meet the confidence threshold and get auto-routed. The remaining (1-coverage) fraction go to human review. Higher threshold = lower coverage but higher accuracy on auto-routed tickets.

```python
import numpy as np
import pandas as pd
from sklearn.metrics import (classification_report, confusion_matrix,
                              f1_score, balanced_accuracy_score,
                              top_k_accuracy_score)

def full_evaluation(y_true, y_pred, y_prob, class_names, model_name):
    print(f"\n{'='*60}")
    print(f"  Evaluation: {model_name}")
    print(f"{'='*60}")

    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    macro_f1    = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    bal_acc     = balanced_accuracy_score(y_true, y_pred)
    print(f"Macro-F1    (equal class weight):   {macro_f1:.4f}   ← PRIMARY")
    print(f"Weighted-F1 (weighted by support):  {weighted_f1:.4f}")
    print(f"Balanced Accuracy:                  {bal_acc:.4f}")
    if y_prob is not None:
        print(f"Top-2 Accuracy:                     "
              f"{top_k_accuracy_score(y_true, y_prob, k=2):.4f}")

    # Confusion matrix — normalise each row so values = recall per class
    print("\nConfusion Matrix (rows=true class, cols=predicted, values=recall):")
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_df = pd.DataFrame(cm_pct.round(3), index=class_names, columns=class_names)
    print(cm_df.to_string())
    print("  1.000 on diagonal = perfect recall for that class")

    # Coverage analysis: what happens at each confidence threshold?
    if y_prob is not None:
        print("\nConfidence Threshold Analysis:")
        print(f"  {'Threshold':>12}  {'Auto-route':>12}  {'Accuracy on auto-routed':>25}")
        for thresh in [0.50, 0.60, 0.70, 0.80, 0.90]:
            auto       = np.max(y_prob, axis=1) >= thresh
            coverage   = auto.mean()
            accuracy   = np.mean(y_pred[auto] == y_true[auto]) if auto.sum() > 0 else 0
            print(f"  {thresh:>12.0%}  {coverage:>12.1%}  {accuracy:>25.4f}")

full_evaluation(y_test, y_pred_lr, y_prob_lr, class_names, "TF-IDF + Logistic Regression")
```

---

## 10. Production Considerations

**What this code does:**
In production, models need to be saved so they can be loaded for serving, monitored for drift as new tickets arrive, and retrained when performance degrades. This section shows how to save all artifacts, detect language drift, and structure the prediction log for retraining.

**Key terms explained:**
- **Model artifact** — every file the serving system needs to make predictions: trained model weights, fitted TF-IDF vectorizer (its vocabulary must be identical to training), label encoder, and metadata. Missing any one of these makes serving impossible.
- **`joblib.dump`** — saves a Python object to disk as a binary file. Preferred over pickle for sklearn/numpy objects because it handles large arrays efficiently.
- **Data drift / concept drift** — the distribution of incoming tickets changes over time (new software, new attack types, new vendor names). A model trained in 2024 may slowly degrade through 2025 as it sees language it was never trained on.
- **OOV rate (Out-of-Vocabulary)** — the fraction of words in new tickets that were never seen during training. High OOV = the model is flying blind on unfamiliar vocabulary. A sudden spike in OOV rate is the earliest warning sign that a retrain is needed.

```python
import joblib, json, datetime

def save_all_artifacts(model, tfidf_word, tfidf_char, label_map, val_macro_f1):
    """
    Save everything needed for inference.
    The vectorizer is as important as the model — without it, you cannot
    transform new text into the same feature space the model was trained on.
    """
    joblib.dump(model,      '/tmp/lr_model.pkl')
    joblib.dump(tfidf_word, '/tmp/tfidf_word.pkl')
    joblib.dump(tfidf_char, '/tmp/tfidf_char.pkl')
    metadata = {
        'model_type':   'LogisticRegression + TF-IDF',
        'label_map':    label_map,
        'val_macro_f1': val_macro_f1,
        'trained_at':   datetime.datetime.utcnow().isoformat(),
        'n_train':      len(y_train),
        'features':     int(X_train_tfidf.shape[1])
    }
    json.dump(metadata, open('/tmp/model_metadata.json','w'), indent=2)
    print("Saved: lr_model.pkl, tfidf_word.pkl, tfidf_char.pkl, model_metadata.json")

save_all_artifacts(lr, tfidf_word, tfidf_char, label_map, val_macro_f1=0.87)

def monitor_oov_drift(new_tickets: list, reference_vocab: set,
                       alert_threshold: float = 0.15) -> float:
    """
    Track what fraction of words in new tickets are unknown to the model.
    If OOV exceeds alert_threshold, schedule a retrain.
    """
    oov_rates = []
    for ticket in new_tickets:
        tokens    = clean_text(ticket).split()
        oov_count = sum(1 for t in tokens if t not in reference_vocab)
        oov_rates.append(oov_count / max(len(tokens), 1))

    avg_oov = float(np.mean(oov_rates))
    status  = "⚠️  RETRAIN RECOMMENDED" if avg_oov > alert_threshold else "✓  OK"
    print(f"\nOOV Drift Monitor: {avg_oov:.2%} average OOV rate  {status}")
    return avg_oov

reference_vocab = set(tfidf_word.get_feature_names_out())
monitor_oov_drift(df['text'].values[-200:].tolist(), reference_vocab)

print("""
Production Flow:
  Incoming ticket (text)
        ↓
  clean_text() → tfidf_word.transform() + tfidf_char.transform()  [< 1ms]
        ↓
  lr.predict_proba()  →  max probability ≥ 0.70?  [< 1ms]
        ↓
  Yes → auto-route to team queue
  No  → human review queue with top-2 class suggestions
        ↓
  Log: {ticket_id, features, prediction, confidence, timestamp}
  Weekly: measure OOV drift + prediction accuracy on resolved tickets
  Monthly: retrain if Macro-F1 drops below deployment baseline
""")
```

---

*Last updated: May 2026 | Pramod Modi — Staff ML Engineer, ServiceNow*
