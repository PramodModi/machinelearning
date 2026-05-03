# Use Case 5: Video / Audio / Content Recommendation System
> Collaborative Filtering · Content-Based Filtering · Two-Tower Neural Networks · Matrix Factorization
> plain-English explanations before every code block

---

## Table of Contents
1. [Problem Statement](#1-problem-statement)
2. [How Recommendation Systems Work — The Big Picture](#2-how-recommendation-systems-work--the-big-picture)
3. [Dataset & Data Understanding](#3-dataset--data-understanding)
4. [Feature Engineering](#4-feature-engineering)
5. [Model Selection — Why This, Not That](#5-model-selection--why-this-not-that)
6. [Solution 1: Collaborative Filtering — Matrix Factorization (ALS)](#6-solution-1-collaborative-filtering--matrix-factorization-als)
7. [Solution 2: Content-Based Filtering — Item Embeddings + Cosine Search](#7-solution-2-content-based-filtering--item-embeddings--cosine-search)
8. [Solution 3: Two-Tower Neural Network (YouTube / Spotify Style)](#8-solution-3-two-tower-neural-network-youtube--spotify-style)
9. [Solution 4: Hybrid System — Combining All Signals](#9-solution-4-hybrid-system--combining-all-signals)
10. [Evaluation Strategy](#10-evaluation-strategy)
11. [The Cold Start Problem & Solutions](#11-the-cold-start-problem--solutions)
12. [Production Architecture](#12-production-architecture)

---

## 1. Problem Statement

A streaming platform (like YouTube, Netflix, or Spotify) has:
- **50 million users**
- **10 million items** (videos, songs, podcasts, movies)
- **5 billion interaction events** per day (plays, skips, likes, watch-time)

When a user opens the app, the platform must decide — in **< 100ms** — which 20 items to show on their home screen from 10 million possibilities.

**Platform comparison:**

| Platform | Input Signal | Item Type | Key Metric |
|---|---|---|---|
| **YouTube** | Watch history, search, clicks | Videos | Watch time (not clicks) |
| **Netflix** | Watch history, ratings, time of day | Movies/Shows | Retention, completion rate |
| **Spotify** | Listen history, skips, saves, playlist adds | Songs/Podcasts | Skip rate, listening duration |
| **TikTok** | Swipes, watch-through rate, shares | Short videos | Watch-through %, re-plays |

**The fundamental challenge — three hard problems at once:**

1. **Scale:** 50M users × 10M items = 500 trillion possible (user, item) pairs. You cannot score all of them.
2. **Sparsity:** Each user has interacted with maybe 200 items out of 10 million. That is 0.002% coverage — 99.998% of the matrix is unknown.
3. **Cold start:** A new user has zero history. A new item has zero interactions. What do you recommend?

**Goal:** Given user U, return a ranked list of 20 items they are most likely to engage with, in < 100ms, refreshed in near-real-time as they interact.

---

## 2. How Recommendation Systems Work — The Big Picture

Modern recommendation systems at YouTube/Netflix/Spotify scale use a **two-stage funnel**:

```
Stage 1 — RETRIEVAL (Candidate Generation)     [< 10ms]
════════════════════════════════════════════════
  10 million items → top ~500 candidates

  Methods:
  • Collaborative Filtering: "users like you watched X"
  • Content-Based:           "you watched X; here are similar items"
  • Two-Tower Neural Net:    learned dense embeddings for user + item

Stage 2 — RANKING                               [< 50ms]
════════════════════════════════════════════════
  500 candidates → top 20 to show

  Methods:
  • LambdaRank / LambdaMART (Learning to Rank)
  • Neural ranker with many more features
  • Business rules (diversity, freshness, boost new content)

Stage 3 — RE-RANKING (optional)                 [< 10ms]
════════════════════════════════════════════════
  20 → final ordered list

  Methods:
  • Diversity injection (don't show 20 action movies)
  • Freshness boost (surface recently uploaded content)
  • Policy constraints (avoid showing same creator twice in a row)
```

**In this document we cover Stages 1 and 2 in full.** Stage 3 is business-logic specific.

---

## 3. Dataset & Data Understanding

**What this code does:**
We simulate a realistic content streaming dataset: users, items (videos/songs), and interaction events. Each interaction has a type (play, skip, like, share) and engagement depth (how much of the item was consumed). We deliberately simulate several real-world patterns: users cluster into taste groups, some items are globally popular, and newer items have fewer interactions (recency bias). We then explore the data to understand its sparsity and distribution before building any models.

**Key terms explained:**
- **Interaction matrix** — a giant table where rows are users, columns are items, and cells contain how much user U engaged with item I (or 0 if no interaction). At 50M users × 10M items, this matrix has 500 trillion cells. Storing it fully would need 4 petabytes of memory — instead we store only the non-zero entries as a **sparse matrix**.
- **Implicit feedback** — the user never explicitly rated an item. Instead, we infer preference from behaviour: a user who watched 90% of a video probably liked it; one who skipped after 5 seconds probably did not. YouTube, Netflix, and Spotify all primarily use implicit feedback (not star ratings).
- **Explicit feedback** — direct rating: 1–5 stars, thumbs up/down. Less common in practice because most users never rate anything. When available it is very high signal.
- **Sparsity** — the fraction of (user, item) pairs with no interaction. 99.998% sparse means most cells in the matrix are empty. Models must generalise from the tiny fraction of observed interactions to predict preferences across the entire matrix.
- **Long-tail distribution** — a small number of items (popular movies, hit songs) get most of the interactions. The vast majority of items (indie films, obscure songs) get almost none. This makes recommendation hard: there is limited signal to learn from for long-tail items.
- **Watch time / engagement depth** — how far through an item the user got. 5% watch = probably disliked. 90% watch = probably liked. This is more informative than just "did they click it."

```python
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

np.random.seed(42)

# ── Scale parameters (reduced for local execution) ──────────────────────────
N_USERS = 10_000    # in production: 50 million
N_ITEMS = 5_000     # in production: 10 million
N_EVENTS= 500_000   # interaction events (play, skip, like, etc.)

# ── Simulate items (videos/songs/podcasts) ──────────────────────────────────
GENRES  = ['Action','Comedy','Drama','Thriller','Horror','Documentary',
           'Romance','SciFi','Music','Podcast']
CREATORS= [f'creator_{i:04d}' for i in range(500)]

items_df = pd.DataFrame({
    'item_id':    range(N_ITEMS),
    'title':      [f'Item_{i:05d}' for i in range(N_ITEMS)],
    'genre':      np.random.choice(GENRES, N_ITEMS),
    'creator_id': np.random.choice(CREATORS, N_ITEMS),
    'duration_s': np.random.lognormal(np.log(300), 0.8, N_ITEMS).clip(30, 7200).astype(int),
    # Item popularity: power-law (a few items are very popular, most are niche)
    # Zipf distribution: item 1 is 2× as popular as item 2, etc.
    'popularity':  (1.0 / np.arange(1, N_ITEMS + 1) ** 0.8),
    'release_day': np.random.randint(0, 365, N_ITEMS),  # day since platform launch
    'avg_rating':  np.random.beta(7, 3, N_ITEMS) * 4 + 1,  # 1–5 stars, skewed positive
})
items_df['popularity'] /= items_df['popularity'].sum()  # normalise to probability

# ── Simulate users with latent taste profiles ────────────────────────────────
# Users cluster into taste groups (like genres they prefer)
N_TASTE_GROUPS = 8
user_taste_group = np.random.randint(0, N_TASTE_GROUPS, N_USERS)

# Each taste group has a preference distribution over genres
taste_genre_prefs = np.random.dirichlet(np.ones(len(GENRES)) * 0.5, N_TASTE_GROUPS)

users_df = pd.DataFrame({
    'user_id':       range(N_USERS),
    'taste_group':   user_taste_group,
    'account_age_d': np.random.exponential(400, N_USERS).clip(1, 1500).astype(int),
    'device':        np.random.choice(['mobile','desktop','tv','tablet'], N_USERS,
                                       p=[0.5, 0.2, 0.2, 0.1]),
    'country':       np.random.choice(['US','IN','BR','DE','GB','JP','MX'], N_USERS,
                                       p=[0.3,0.2,0.1,0.08,0.08,0.07,0.07]),
})

# ── Simulate interaction events ──────────────────────────────────────────────
# Users from the same taste group are more likely to interact with items
# from genres that taste group prefers — this creates the collaborative signal
event_types = ['play','skip','like','share','save','dislike']
event_weights = [0.55, 0.25, 0.10, 0.04, 0.04, 0.02]   # play is most common

events = []
for _ in range(N_EVENTS):
    # Sample user uniformly
    user_id    = np.random.randint(0, N_USERS)
    taste_grp  = users_df.loc[user_id, 'taste_group']

    # Item sampling: blend taste-group genre preference + global popularity
    # 60% taste-driven, 40% popularity-driven (reflects real platforms)
    if np.random.random() < 0.6:
        preferred_genre = np.random.choice(GENRES, p=taste_genre_prefs[taste_grp])
        genre_items = items_df[items_df['genre'] == preferred_genre]['item_id'].values
        item_id     = np.random.choice(genre_items)
    else:
        item_id = np.random.choice(N_ITEMS, p=items_df['popularity'].values)

    event_type = np.random.choice(event_types, p=event_weights)
    duration   = items_df.loc[item_id, 'duration_s']

    # Engagement depth: how much of the item did they consume?
    # Likes and saves = high engagement; skips = low; plays = variable
    if event_type == 'like':
        pct_watched = np.random.beta(8, 2)     # mostly high (0.8–1.0)
    elif event_type == 'skip':
        pct_watched = np.random.beta(1, 8)     # mostly low (0.05–0.3)
    elif event_type == 'play':
        pct_watched = np.random.beta(3, 2)     # moderate (0.5–0.8)
    else:
        pct_watched = np.random.beta(5, 2)

    events.append({
        'user_id':       user_id,
        'item_id':       item_id,
        'event_type':    event_type,
        'pct_consumed':  round(pct_watched, 3),
        'watch_seconds': int(pct_watched * duration),
        'timestamp':     np.random.randint(0, 86400 * 30),  # last 30 days in seconds
    })

events_df = pd.DataFrame(events)
print(f"Users:  {N_USERS:,}")
print(f"Items:  {N_ITEMS:,}")
print(f"Events: {len(events_df):,}")
print(f"\nEvent type distribution:")
print(events_df['event_type'].value_counts(normalize=True).round(3))
print(f"\nMedian pct_consumed by event type:")
print(events_df.groupby('event_type')['pct_consumed'].median().round(3))

# Count unique interactions per user (how many items has each user touched?)
interactions_per_user = events_df.groupby('user_id')['item_id'].nunique()
interactions_per_item = events_df.groupby('item_id')['user_id'].nunique()

print(f"\nInteraction matrix sparsity analysis:")
total_pairs   = N_USERS * N_ITEMS
observed_pairs= events_df.groupby(['user_id','item_id']).ngroups
sparsity      = 1 - observed_pairs / total_pairs
print(f"  Total possible (user, item) pairs: {total_pairs:,}")
print(f"  Observed pairs:                    {observed_pairs:,}")
print(f"  Sparsity:                          {sparsity:.5%}")
print(f"  Avg items per user: {interactions_per_user.mean():.1f}")
print(f"  Avg users per item: {interactions_per_item.mean():.1f}")
print(f"  Median items per user: {interactions_per_user.median():.0f}")
print(f"\nLong-tail: top 1% of items get {interactions_per_item.nlargest(int(N_ITEMS*0.01)).sum() / len(events_df):.1%} of all interactions")
```

---

## 4. Feature Engineering

**What this code does:**
We engineer features at three levels: (1) user-level features capturing their overall taste profile, (2) item-level features capturing content characteristics, and (3) interaction-level signals converting raw events into meaningful scores. The most important step is converting raw events (plays, skips, likes) into a single **implicit rating score** that encodes how much each user liked each item.

**Key terms explained:**
- **Implicit rating score** — a single number we compute from multiple weak signals (play, skip, like, watch time) to represent how much a user liked an item. It is "implicit" because the user never stated it directly — we infer it from their behaviour. Formula: `score = watch_pct × base_weight × event_multiplier`. A skip after 5% is near-zero; a like after watching 95% is near 1.0.
- **User-item interaction matrix** — a sparse matrix where entry (u, i) = implicit rating score of user u for item i. Row = one user's preferences across all items. Column = all users' reactions to one item. Most entries are 0 (unseen).
- **CSR (Compressed Sparse Row) matrix** — a memory-efficient way to store the sparse interaction matrix. Instead of storing all N_USERS × N_ITEMS values, CSR stores only the non-zero values along with their row and column indices. 99.998% sparse means 99.998% of values are zero — CSR saves ~99.998% of memory.
- **Recency weighting** — more recent interactions should count more. A user who loved a genre 3 years ago might have evolved. `weight = exp(-lambda × days_ago)` reduces the score of old interactions exponentially.
- **Genre affinity** — the fraction of a user's high-engagement interactions that belong to each genre. Captures the user's content preferences. "This user watches 40% Action and 35% Comedy" is a powerful signal for generating candidates.
- **Item popularity rank** — log-scaled rank of item by total interaction count. Helps the model distinguish mega-hits (rank 1–100) from niche content (rank 100,000+). Log-scaling prevents the top item from being 100,000× more influential than item 100,000.
- **`pivot_table`** — aggregates a DataFrame into a 2D matrix. `pivot_table(values='score', index='user_id', columns='item_id')` creates the user-item matrix from the long-form events table.

```python
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

# ── Step 1: Convert raw events → implicit rating scores ─────────────────────
# Different event types carry different signal strength
# A "like" is stronger than a "play"; a "share" is strongest; a "skip" is negative
EVENT_WEIGHTS = {
    'play':    1.0,
    'like':    3.0,
    'save':    2.5,
    'share':   4.0,
    'skip':    0.1,   # weak but still a signal (seen the item, chose not to engage)
    'dislike': -1.0,  # explicit negative signal (rare but valuable)
}

# Recency decay: interactions from 30 days ago count half as much as today's
DECAY_LAMBDA = np.log(2) / (30 * 86400)  # half-life = 30 days in seconds

MAX_TIMESTAMP = events_df['timestamp'].max()

def compute_implicit_score(row) -> float:
    """
    Combine event type, watch percentage, and recency into a single score.

    Score formula:
      base        = watch_percentage × event_weight
      recency     = exp(-lambda × seconds_ago)   [recent interactions score higher]
      final_score = base × recency

    Examples:
      Play + 95% watched + today          → 1.0 × 0.95 × 1.00 = 0.95
      Like + 90% watched + today          → 3.0 × 0.90 × 1.00 = 2.70
      Skip + 5%  watched + 20 days ago    → 0.1 × 0.05 × 0.63 = 0.003
      Dislike + 30% watched + 5 days ago  → -1.0 × 0.30 × 0.89 = -0.27
    """
    base      = row['pct_consumed'] * EVENT_WEIGHTS.get(row['event_type'], 1.0)
    seconds_ago = MAX_TIMESTAMP - row['timestamp']
    recency     = np.exp(-DECAY_LAMBDA * seconds_ago)
    return base * recency

events_df['implicit_score'] = events_df.apply(compute_implicit_score, axis=1)

# Aggregate: a user may have interacted with the same item multiple times
# Sum scores across all interactions between the same (user, item) pair
interactions = (events_df
    .groupby(['user_id', 'item_id'])['implicit_score']
    .sum()
    .reset_index()
    .rename(columns={'implicit_score': 'score'})
)
# Clip: cap maximum score to prevent one super-engaged interaction dominating
interactions['score'] = interactions['score'].clip(lower=-2.0, upper=5.0)

print(f"Unique (user, item) pairs: {len(interactions):,}")
print(f"Score distribution:")
print(interactions['score'].describe().round(3))

# ── Step 2: Build sparse user-item matrix ────────────────────────────────────
# CSR format: only stores non-zero entries → huge memory saving
# Row index = user_id, Column index = item_id, Value = implicit score
user_item_matrix = csr_matrix(
    (interactions['score'].values,
     (interactions['user_id'].values, interactions['item_id'].values)),
    shape=(N_USERS, N_ITEMS),
    dtype=np.float32
)
print(f"\nUser-item matrix: {user_item_matrix.shape}")
print(f"Non-zero entries: {user_item_matrix.nnz:,}")
print(f"Memory (CSR):     ~{user_item_matrix.data.nbytes / 1e6:.1f} MB")
dense_mem = N_USERS * N_ITEMS * 4 / 1e9
print(f"Memory (dense):   ~{dense_mem:.1f} GB  (CSR saves {dense_mem*1e3/user_item_matrix.data.nbytes*1e-3:.0f}× memory)")

# ── Step 3: User-level features ──────────────────────────────────────────────
# Compute each user's genre affinity from their high-engagement interactions
print("\nComputing user genre affinities...")
high_eng     = events_df[events_df['pct_consumed'] > 0.6].copy()
high_eng     = high_eng.merge(items_df[['item_id','genre']], on='item_id')
genre_counts = (high_eng.groupby(['user_id','genre']).size()
                         .unstack(fill_value=0))
# Normalise each user's genre counts to sum to 1 (affinity proportions)
genre_affinity = genre_counts.div(genre_counts.sum(axis=1).clip(1), axis=0)
genre_affinity.columns = [f'genre_affinity_{g.lower()}' for g in genre_affinity.columns]

user_features = users_df.set_index('user_id').copy()
user_features = user_features.join(genre_affinity, how='left').fillna(0)
user_features['n_interactions']   = interactions.groupby('user_id')['score'].count()
user_features['avg_score']        = interactions.groupby('user_id')['score'].mean()
user_features['n_unique_genres']  = (genre_counts > 0).sum(axis=1)
user_features = user_features.fillna(0)
print(f"User feature matrix: {user_features.shape}")

# ── Step 4: Item-level features ──────────────────────────────────────────────
item_stats = (interactions.groupby('item_id')['score']
              .agg(['count','mean','sum'])
              .rename(columns={'count':'n_interactions','mean':'avg_score','sum':'total_score'}))
items_features = items_df.set_index('item_id').join(item_stats, how='left').fillna(0)

# Log-scaled popularity rank: reduces impact of extreme popularity differences
items_features['popularity_rank'] = items_features['n_interactions'].rank(ascending=False)
items_features['log_popularity']  = np.log1p(items_features['n_interactions'])
items_features['duration_bin']    = pd.cut(items_df['duration_s'],
                                            bins=[0,60,300,900,3600,np.inf],
                                            labels=[0,1,2,3,4]).astype(int)
print(f"Item feature matrix: {items_features.shape}")
```

---

## 5. Model Selection — Why This, Not That

**What this section explains:**
Recommendation systems have several very different algorithmic approaches. The right choice depends on whether you have enough interaction history, whether you need explanations, and what your latency budget is.

| Approach | How It Works | Use When | Avoid When |
|---|---|---|---|
| **Popularity baseline** | Recommend the most-played items globally | Zero-history users; sanity check baseline | Personalisation needed |
| **Collaborative Filtering (CF)** | "Users like you also liked X" | Rich interaction history; good for mainstream taste | Cold start (new users/items); niche taste |
| **Matrix Factorisation (ALS/SVD)** | Decompose the interaction matrix into user and item vectors | Best accuracy for pure CF; explainable vectors | Very new users; purely content-based use cases |
| **Content-Based Filtering** | "You liked X; here are similar items by content" | New items with zero interactions; explainable | Serendipity (only recommends more of the same) |
| **Two-Tower Neural Net** | Learned embeddings for users and items jointly | Maximum accuracy; handles both CF and content signals; scalable | Small datasets (< 1M interactions); no GPU |
| **Session-based (RNN/Transformer)** | "What do you want right now based on last 10 actions?" | Short-session apps (TikTok, news); highly contextual | Long-term preference modelling |
| **Knowledge Graph** | Use entity relationships (actor → movie → genre) | Explainability required; complex domain knowledge | Latency-sensitive retrieval |

**Why Collaborative Filtering as the core:**
CF is the single most powerful technique for mature platforms. The intuition is simple: if user A and user B both loved movies X, Y, and Z, then when A watches W, there's a good chance B will like W too. No content understanding needed. This works even for highly subjective content (humour, aesthetic) where content features are inadequate.

**Why Matrix Factorization specifically:**
Instead of finding exact neighbour users (which is O(n) at query time), matrix factorization compresses every user and item into a small dense vector (e.g., 128 numbers). Then user-item similarity = dot product of their vectors. At inference: one dot product per item = very fast. The vectors are learned to reconstruct the observed interaction matrix as faithfully as possible.

**Why Two-Tower Neural Networks at YouTube/Spotify scale:**
Two-tower models learn user and item representations using both interaction history AND content features simultaneously. The output is two 128-dim vectors that can be compared with dot product — same fast inference as matrix factorization, but much richer input features. This is the architecture used by YouTube, Facebook, and Spotify in production.

---

## 6. Solution 1: Collaborative Filtering — Matrix Factorization (ALS)

**What this code does:**
We factorise the sparse user-item interaction matrix into two smaller matrices: a user embedding matrix (N_USERS × K) and an item embedding matrix (N_ITEMS × K). The product of these two matrices approximately reconstructs the full interaction matrix. The K-dimensional vectors are learned latent factors — they capture hidden patterns like "this user likes fast-paced content" without us ever defining what "fast-paced" means.

**Key terms explained:**
- **Matrix Factorization** — decomposing a large matrix M into the product of two smaller matrices: M ≈ U × Vᵀ. U has shape (N_USERS × K) and V has shape (N_ITEMS × K). K is the latent factor dimension (e.g., 64 or 128). Each row of U is a user's "taste vector"; each row of V is an item's "characteristic vector."
- **Latent factors** — the K numbers representing a user or item are called latent factors because we never explicitly define what they mean. They are automatically discovered from the interaction data. One dimension might correlate with "action vs drama", another with "short vs long content", but we never label them.
- **ALS (Alternating Least Squares)** — the training algorithm for matrix factorization. It alternates between two steps: (1) fix user vectors, solve for the best item vectors. (2) fix item vectors, solve for the best user vectors. Each step is a least-squares problem with a closed-form solution. More stable than gradient descent for sparse data.
- **Implicit ALS (iALS)** — a variant of ALS designed specifically for implicit feedback (play counts, watch time) rather than explicit ratings. Treats unobserved entries as "probably not preferred" with low confidence, rather than treating them as "definitely 0."
- **Regularization λ** — a penalty term that prevents user and item vectors from becoming very large. Large vectors lead to overfitting: the model memorises the training interactions perfectly but generalises poorly to new interactions. Lambda = 0.01 to 0.1 is typical.
- **`user_factors` / `item_factors`** — the learned matrices. After training, `user_factors[user_id]` is a 64-dim vector for that user. `item_factors[item_id]` is a 64-dim vector for that item. Their dot product is the predicted rating.
- **Candidate generation** — the retrieval step: given a user vector, find the top-K item vectors with highest dot product. We use ANN (approximate nearest neighbour) search for this — the same HNSW index as in Use Case 4.

```python
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# We implement ALS (Alternating Least Squares) from scratch for clarity.
# In production you would use: pip install implicit  (optimised C++ + CUDA ALS)

class ALSRecommender:
    """
    Matrix Factorisation via Alternating Least Squares.

    The interaction matrix R ≈ U × Vᵀ where:
      U: (n_users × n_factors)  — user taste vectors
      V: (n_items × n_factors)  — item characteristic vectors

    Training alternates between:
      Fix U, solve for best V  (closed-form per item)
      Fix V, solve for best U  (closed-form per user)

    Each solve is a regularised least squares: (AᵀA + λI)⁻¹Aᵀb
    """

    def __init__(self, n_factors: int = 64, n_iterations: int = 15,
                 regularization: float = 0.01, alpha: float = 40.0):
        """
        n_factors:      dimension of latent factor vectors (K)
        n_iterations:   number of ALS iterations (alternating steps)
        regularization: lambda — prevents vectors from growing too large
        alpha:          confidence scaling for implicit feedback
                        Higher alpha → stronger signal from positive interactions
        """
        self.n_factors     = n_factors
        self.n_iterations  = n_iterations
        self.regularization= regularization
        self.alpha         = alpha

    def fit(self, R: csr_matrix):
        """
        R: sparse user-item interaction matrix (n_users × n_items)
        Values are implicit scores (higher = more confident the user likes it)
        """
        n_users, n_items = R.shape

        # Initialise U and V with small random values
        # Using sqrt(n_factors) scaling for numerical stability
        scale = 0.01
        self.user_factors = np.random.normal(0, scale, (n_users, self.n_factors)).astype(np.float32)
        self.item_factors = np.random.normal(0, scale, (n_items, self.n_factors)).astype(np.float32)

        # Confidence matrix: C = 1 + alpha × R
        # Unobserved entries (R=0): confidence = 1 (low, but not ignored completely)
        # Observed entries (R=2.5): confidence = 1 + 40×2.5 = 101 (much stronger signal)
        # This is the core insight of implicit ALS — treat 0s as weak negatives, not missing

        print(f"Training ALS ({n_users} users × {n_items} items, {self.n_factors} factors)...")

        for iteration in range(self.n_iterations):
            # ── Step 1: Fix item_factors V, update user_factors U ──────────
            # For each user u, solve: (VᵀCᵤV + λI) uᵤ = Vᵀ Cᵤ pᵤ
            # where Cᵤ = diagonal matrix of confidences for user u
            #       pᵤ = preference vector (1 where interacted, 0 elsewhere)
            VTV = self.item_factors.T @ self.item_factors           # (K×K)
            VTV_reg = VTV + self.regularization * np.eye(self.n_factors)

            for u in range(n_users):
                # Get items this user interacted with
                u_items = R[u].indices              # column indices of non-zero entries
                u_scores= R[u].data                 # interaction scores

                if len(u_items) == 0:
                    continue  # no data for this user; keep random initialisation

                # Confidence values for this user's interactions
                conf   = 1 + self.alpha * u_scores   # (n_interacted,)
                V_u    = self.item_factors[u_items]   # item vectors user interacted with (n_interacted × K)

                # A = VᵀCᵤV + λI  (K×K matrix to invert)
                # We only add the delta from confidence > 1 for efficiency
                A  = VTV_reg + V_u.T @ (np.diag(conf - 1) @ V_u)
                # b = Vᵀ Cᵤ pᵤ  (p_u = 1 for all interacted items)
                b  = V_u.T @ conf                    # (K,)
                # Solve the linear system A @ u_u = b
                self.user_factors[u] = np.linalg.solve(A, b)

            # ── Step 2: Fix user_factors U, update item_factors V ──────────
            # Symmetric to Step 1: transpose the problem
            UᵀU     = self.user_factors.T @ self.user_factors
            UᵀU_reg = UᵀU + self.regularization * np.eye(self.n_factors)
            R_csc   = R.T.tocsr()  # transpose to CSR for efficient column access

            for i in range(n_items):
                i_users = R_csc[i].indices
                i_scores= R_csc[i].data

                if len(i_users) == 0:
                    continue

                conf   = 1 + self.alpha * i_scores
                U_i    = self.user_factors[i_users]
                A  = UᵀU_reg + U_i.T @ (np.diag(conf - 1) @ U_i)
                b  = U_i.T @ conf
                self.item_factors[i] = np.linalg.solve(A, b)

            if (iteration + 1) % 3 == 0 or iteration == 0:
                # Quick training loss estimate (RMSE on observed entries)
                # Full evaluation uses Recall@K (see Section 10)
                sample_u = np.random.choice(n_users, min(500, n_users), replace=False)
                preds, trues = [], []
                for u in sample_u:
                    items = R[u].indices[:5]
                    for i in items:
                        pred = float(self.user_factors[u] @ self.item_factors[i])
                        true = float(R[u, i])
                        preds.append(pred); trues.append(true)
                rmse = np.sqrt(np.mean((np.array(preds) - np.array(trues))**2))
                print(f"  Iteration {iteration+1:2d}  RMSE (observed sample): {rmse:.4f}")

        return self

    def recommend(self, user_id: int, n: int = 20,
                  exclude_seen: bool = True) -> list:
        """
        Generate top-n recommendations for a user.

        Method: compute dot product of user vector with ALL item vectors.
        Top-n highest dot products = most predicted-to-be-liked items.

        In production: replace the inner product loop with HNSW ANN search
        on the item_factors matrix for sub-millisecond retrieval.
        """
        user_vec = self.user_factors[user_id]          # (K,)
        # Predicted score for all items: dot product user vector × each item vector
        scores   = self.item_factors @ user_vec        # (n_items,)

        if exclude_seen:
            # Zero out items the user has already interacted with
            seen_items = user_item_matrix[user_id].indices
            scores[seen_items] = -np.inf

        top_n = np.argsort(scores)[::-1][:n]
        return [(int(item_id), float(scores[item_id])) for item_id in top_n]

    def similar_items(self, item_id: int, n: int = 10) -> list:
        """
        Items most similar to item_id in the latent factor space.
        Uses cosine similarity between item vectors.
        Useful for "more like this" and "because you watched X" features.
        """
        item_vec  = self.item_factors[item_id]
        # Cosine similarity = dot product of unit-normalised vectors
        norms     = np.linalg.norm(self.item_factors, axis=1, keepdims=True) + 1e-8
        normed    = self.item_factors / norms
        item_norm = item_vec / (np.linalg.norm(item_vec) + 1e-8)
        cos_sims  = normed @ item_norm
        cos_sims[item_id] = -1   # exclude the item itself
        top_n = np.argsort(cos_sims)[::-1][:n]
        return [(int(i), float(cos_sims[i])) for i in top_n]

# ── Train ALS ────────────────────────────────────────────────────────────────
als = ALSRecommender(n_factors=64, n_iterations=15,
                     regularization=0.01, alpha=40.0)
als.fit(user_item_matrix)

# ── Generate recommendations for sample users ────────────────────────────────
print("\n── ALS Recommendations ──")
for user_id in [0, 100, 500]:
    recs = als.recommend(user_id, n=5)
    taste = users_df.loc[user_id, 'taste_group']
    print(f"\nUser {user_id} (taste group {taste}):")
    for item_id, score in recs:
        genre = items_df.loc[item_id, 'genre']
        title = items_df.loc[item_id, 'title']
        print(f"  Item {item_id:5d}  [{genre:<12}]  {title}  score={score:.4f}")

# ── Similar items example ────────────────────────────────────────────────────
print("\n── Similar Items (ALS cosine similarity) ──")
ref_item = 42
ref_genre = items_df.loc[ref_item, 'genre']
print(f"Reference: Item {ref_item}  [{ref_genre}]")
for item_id, sim in als.similar_items(ref_item, n=5):
    genre = items_df.loc[item_id, 'genre']
    print(f"  Similar: Item {item_id:5d}  [{genre:<12}]  cosine={sim:.4f}")

# ── User factor interpretation ────────────────────────────────────────────────
print("\n── Latent Factor Analysis ──")
print(f"User factor matrix: {als.user_factors.shape}")
print(f"Item factor matrix: {als.item_factors.shape}")
print(f"Factor dimensions captured (variance explained by top 5 factors):")
from numpy.linalg import svd
_, sv, _ = svd(als.item_factors, full_matrices=False)
var_explained = (sv**2 / (sv**2).sum()).cumsum()
for k in [1, 5, 10, 20, 64]:
    k = min(k, len(var_explained))
    print(f"  Top {k:3d} factors: {var_explained[k-1]:.1%} variance explained")
```

---

## 7. Solution 2: Content-Based Filtering — Item Embeddings + Cosine Search

**What this code does:**
Content-based filtering recommends items similar to what a user has previously liked, based entirely on item features (genre, creator, duration, tags) — no other users' behaviour is used. This is crucial for cold-start items (newly uploaded content with zero interactions) and for users with very niche tastes not well-represented by other users.

We build a content embedding for each item by combining structured features (genre, duration) with a text embedding of its title and description. Then, given a user's like history, we retrieve the items whose content vectors are most similar to the items they loved.

**Key terms explained:**
- **Content-based filtering** — recommend items by comparing their content features to what the user has liked before. No other users needed. Works entirely from item metadata. "You liked action movies with fast editing → here are more action movies with fast editing."
- **Item content vector** — a fixed-size numerical representation of an item's content. Combines: one-hot encoded genre, normalised duration, creator embedding, etc. Cosine similarity between two item vectors measures content similarity.
- **One-hot encoding** — converting a categorical variable (genre = "Action") into a binary vector. With 10 genres: "Action" → [1,0,0,0,0,0,0,0,0,0]. "Comedy" → [0,1,0,0,0,0,0,0,0,0]. Every item's genre becomes a 10-dimensional binary vector.
- **User profile vector** — the weighted average of the content vectors of items a user has liked. If a user loved 5 Action items and 2 Comedy items (with higher scores for Action), their profile vector leans toward the Action genre region of the vector space.
- **Cosine similarity** — measures the angle between two vectors. Two items with similar genre + duration + tags produce vectors pointing in similar directions → cosine similarity close to 1.0. Unrelated items point in different directions → cosine similarity close to 0.
- **FAISS IndexFlatIP** — FAISS index for exact inner product (dot product) search. After L2-normalising all vectors, inner product equals cosine similarity. Given a query vector, it finds the top-K database vectors with highest dot product. "IP" = inner product.

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity

# ── Step 1: Build content vectors for each item ──────────────────────────────
print("Building item content vectors...")

# Encode categorical features
# One-hot encoding: genre "Action" → a binary vector of length n_genres
enc_genre = OneHotEncoder(sparse_output=False)
genre_ohe  = enc_genre.fit_transform(items_df[['genre']])   # (N_ITEMS, 10)

# Normalise numeric features: bring all to similar scale [0,1]
# Without scaling, duration_s (range 30–7200) dominates avg_rating (range 1–5)
scaler     = StandardScaler()
numeric_feats = items_df[['duration_s','avg_rating','release_day','log_popularity']].copy()
numeric_feats['log_popularity'] = np.log1p(items_features.reindex(items_df.index)['n_interactions'].fillna(0))
numeric_scaled = scaler.fit_transform(numeric_feats)   # (N_ITEMS, 4)

# Creator embedding: each creator gets a random but consistent 16-dim vector
# In production: train creator embeddings from interaction data
creator_list = items_df['creator_id'].unique()
creator_emb  = {c: np.random.normal(0, 0.1, 16) for c in creator_list}
creator_vecs = np.array([creator_emb[c] for c in items_df['creator_id']])  # (N_ITEMS, 16)

# Concatenate all features into one content vector per item
# Weights: genre is most important, then numeric, then creator
item_content_vectors = np.hstack([
    genre_ohe   * 2.0,   # genres weighted 2× — most important signal
    numeric_scaled,
    creator_vecs
]).astype(np.float32)

# L2-normalise: each vector has unit length → cosine similarity = dot product
norms = np.linalg.norm(item_content_vectors, axis=1, keepdims=True) + 1e-8
item_content_norm = item_content_vectors / norms

print(f"Item content vectors: {item_content_vectors.shape}  "
      f"({genre_ohe.shape[1]} genre + {numeric_scaled.shape[1]} numeric + {creator_vecs.shape[1]} creator)")

# ── Step 2: FAISS index for fast content similarity search ──────────────────
try:
    import faiss
    DIM = item_content_norm.shape[1]
    # IndexFlatIP: exact inner product search (= cosine similarity on normalised vectors)
    content_index = faiss.IndexFlatIP(DIM)
    content_index.add(item_content_norm)
    print(f"FAISS content index: {content_index.ntotal} items indexed")

    def find_similar_content(query_vec: np.ndarray, k: int = 20,
                              exclude_ids: list = None) -> list:
        """Find k items most similar in content space to query_vec."""
        q = query_vec.reshape(1,-1).astype(np.float32)
        q = q / (np.linalg.norm(q) + 1e-8)  # normalise query too
        D, I = content_index.search(q, k + (len(exclude_ids or [])))
        results = [(int(i), float(d)) for i, d in zip(I[0], D[0])
                   if i >= 0 and (exclude_ids is None or i not in exclude_ids)]
        return results[:k]
    FAISS_AVAILABLE = True

except ImportError:
    print("FAISS not installed. Using brute-force cosine search.")
    def find_similar_content(query_vec: np.ndarray, k: int = 20,
                              exclude_ids: list = None) -> list:
        sims  = cosine_similarity(query_vec.reshape(1,-1), item_content_norm)[0]
        if exclude_ids:
            sims[exclude_ids] = -1
        top_k = np.argsort(sims)[::-1][:k]
        return [(int(i), float(sims[i])) for i in top_k]
    FAISS_AVAILABLE = False

# ── Step 3: Build user profile as weighted average of liked item vectors ─────
def build_user_content_profile(user_id: int, top_n_liked: int = 20) -> np.ndarray:
    """
    A user's content profile = weighted average of content vectors of items they liked.
    Weights = their implicit scores (higher score = liked more = contributes more to profile).

    This captures: "this user's taste leans toward Action + long-form + creator_042"
    without ever seeing what other users like.
    """
    # Get this user's interactions, sorted by score (most liked first)
    user_rows = interactions[interactions['user_id'] == user_id].nlargest(top_n_liked, 'score')

    if len(user_rows) == 0:
        # No history: return zero vector (will need cold-start handling)
        return np.zeros(item_content_norm.shape[1], dtype=np.float32)

    # Weighted average: items with higher scores contribute more to the profile
    scores    = user_rows['score'].values.reshape(-1, 1)    # (n_liked, 1)
    item_vecs = item_content_norm[user_rows['item_id'].values]  # (n_liked, D)
    profile   = (item_vecs * scores).sum(axis=0) / (scores.sum() + 1e-8)
    return profile.astype(np.float32)

# ── Step 4: Content-based recommendations ───────────────────────────────────
def content_based_recommend(user_id: int, n: int = 20) -> list:
    """
    1. Build user profile from their like history
    2. Find items with content most similar to that profile
    3. Exclude items they've already seen
    """
    user_profile = build_user_content_profile(user_id)
    seen_items   = interactions[interactions['user_id'] == user_id]['item_id'].tolist()
    recs = find_similar_content(user_profile, k=n + len(seen_items),
                                 exclude_ids=seen_items)
    return recs[:n]

print("\n── Content-Based Recommendations ──")
for user_id in [0, 100, 500]:
    recs         = content_based_recommend(user_id, n=5)
    profile_vec  = build_user_content_profile(user_id)
    # Find the dominant genre in the user's profile
    genre_scores = (profile_vec[:genre_ohe.shape[1]] * 2.0 / 2.0)  # undo the 2× weight
    dom_genre    = enc_genre.categories_[0][np.argmax(genre_scores)]
    print(f"\nUser {user_id} (dominant genre in profile: {dom_genre}):")
    for item_id, score in recs:
        genre = items_df.loc[item_id, 'genre']
        print(f"  Item {item_id:5d}  [{genre:<12}]  content_sim={score:.4f}")

# ── Step 5: "More like this" for a specific item ─────────────────────────────
def more_like_this(item_id: int, n: int = 5) -> list:
    """Find items most similar in content to a given item. Used in item detail pages."""
    query_vec = item_content_norm[item_id]
    return find_similar_content(query_vec, k=n+1, exclude_ids=[item_id])[:n]

ref   = 100
print(f"\n── More Like This: Item {ref} [{items_df.loc[ref,'genre']}] ──")
for sim_id, score in more_like_this(ref, n=5):
    print(f"  Item {sim_id:5d}  [{items_df.loc[sim_id,'genre']:<12}]  sim={score:.4f}")
```

---

## 8. Solution 3: Two-Tower Neural Network (YouTube / Spotify Style)

**What this code does:**
We build a Two-Tower neural network — the architecture used in production by YouTube, Spotify, and Facebook. The idea: train one neural network (the "user tower") to encode user features + history into a 128-dim vector, and a separate neural network (the "item tower") to encode item features into a 128-dim vector. Both towers are trained together so that vectors of users and items they interacted with end up close in the 128-dim shared space.

After training, all item vectors are pre-computed and indexed with HNSW. At inference, we only run the user tower (one forward pass) and do an ANN lookup — same speed as matrix factorization but with much richer features.

**Key terms explained:**
- **Two-Tower architecture** — two separate encoder networks sharing the same output space. Tower 1 encodes users (input: user features + interaction history). Tower 2 encodes items (input: item features + content). Both output vectors of the same dimension (128). The model is trained so that positive (user, item) pairs produce similar vectors.
- **Shared embedding space** — users and items are mapped into the same 128-dimensional space. A user who loves Action movies will have a vector close to Action movies' vectors. Unrelated items will be far away.
- **In-batch negatives** — a training trick. In each mini-batch of 512 (user, item) positive pairs, every item that belongs to a different user in the batch is treated as a negative example for the current user. This gives 511 negatives per positive without any extra data, making training efficient.
- **Contrastive loss (InfoNCE)** — the training loss. For each positive (user, item) pair, the loss pushes that pair's dot product high while pushing all other (user, item) pairs in the batch low. `L = -log(exp(u·i) / Σ exp(u·j))`. This is the same loss used in CLIP (image-text matching) and word2vec.
- **Temperature scaling (τ)** — a parameter that controls how "sharp" the score distribution is. Low temperature (τ=0.05): the model must be very confident — high scores for positives and very low for negatives. High temperature: allows more uncertainty. Typical: τ=0.05 to 0.1.
- **`@torch.no_grad()`** — during inference, we don't need to compute gradients (which are only needed during backpropagation/training). Disabling gradient computation saves ~50% of memory and speeds up inference 2×.
- **Pre-computing item embeddings** — after training, we run all N_ITEMS through the item tower once and store the resulting vectors. At query time, we only run the user tower (one forward pass) then do an ANN lookup against the pre-computed item vectors. This is what makes the system scale to 10M items.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ── Step 1: Prepare training data (positive user-item pairs) ─────────────────
# Each training example = one (user_id, item_id) pair where the user interacted
# positively (score > 0.5) with the item
positive_pairs = interactions[interactions['score'] > 0.5][['user_id','item_id']].values
print(f"Positive training pairs: {len(positive_pairs):,}")

# ── Step 2: Feature preparation ──────────────────────────────────────────────
# User features: genre affinities + account age + device encoding
device_map  = {'mobile':0,'desktop':1,'tv':2,'tablet':3}
country_map = {'US':0,'IN':1,'BR':2,'DE':3,'GB':4,'JP':5,'MX':6}

user_feat_cols = ([f'genre_affinity_{g.lower()}' for g in GENRES] +
                  ['n_interactions','avg_score','n_unique_genres'])
user_feat_matrix = user_features[user_feat_cols].fillna(0).values.astype(np.float32)
user_device_enc  = users_df['device'].map(device_map).fillna(0).values.astype(np.int64)
user_account_age = np.log1p(users_df['account_age_d'].values).astype(np.float32)

# Item features: genre OHE + numeric + popularity
item_feat_matrix = np.hstack([
    genre_ohe,
    numeric_scaled,
]).astype(np.float32)

N_USER_FEATS = user_feat_matrix.shape[1] + 1  # +1 for log(account_age)
N_ITEM_FEATS = item_feat_matrix.shape[1]

# ── Step 3: Dataset class ─────────────────────────────────────────────────────
class InteractionDataset(torch.utils.data.Dataset):
    """Each sample is a (user_id, item_id) positive pair."""
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        u, i = self.pairs[idx]
        # User features: concatenate continuous features + log(account_age)
        u_feat = np.append(user_feat_matrix[u], user_account_age[u])
        i_feat = item_feat_matrix[i]
        return (torch.tensor(u_feat,   dtype=torch.float32),
                torch.tensor(i_feat,   dtype=torch.float32),
                torch.tensor(u,        dtype=torch.long),
                torch.tensor(i,        dtype=torch.long))

# ── Step 4: Two-Tower Model ──────────────────────────────────────────────────
class UserTower(nn.Module):
    """
    Encodes a user's features + interaction history into a 128-dim vector.
    Architecture: MLP with BatchNorm + ReLU + Dropout.
    Output is L2-normalised so cosine similarity = dot product.
    """
    def __init__(self, input_dim: int, hidden_dims=(256, 128),
                 output_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        layers = []
        prev   = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        emb = self.net(x)
        # L2 normalise: makes dot product equivalent to cosine similarity
        return F.normalize(emb, p=2, dim=1)

class ItemTower(nn.Module):
    """
    Encodes an item's content features into a 128-dim vector.
    Symmetric to UserTower — same output dimension for comparability.
    """
    def __init__(self, input_dim: int, hidden_dims=(128, 128),
                 output_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        layers = []
        prev   = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        emb = self.net(x)
        return F.normalize(emb, p=2, dim=1)

class TwoTowerModel(nn.Module):
    def __init__(self, n_user_feats, n_item_feats, output_dim=128):
        super().__init__()
        self.user_tower = UserTower(n_user_feats, hidden_dims=(256,128),
                                    output_dim=output_dim)
        self.item_tower = ItemTower(n_item_feats, hidden_dims=(128,128),
                                    output_dim=output_dim)
        # Temperature: learnable parameter that controls contrast sharpness
        # Initialised to log(1/0.07) ≈ 2.66, equivalent to τ=0.07
        self.log_temperature = nn.Parameter(torch.tensor(2.66))

    def forward(self, user_feats, item_feats):
        u_emb = self.user_tower(user_feats)   # (batch, 128)
        i_emb = self.item_tower(item_feats)   # (batch, 128)
        return u_emb, i_emb

    @property
    def temperature(self):
        # Clamp temperature to [0.01, 0.5] to prevent instability
        return torch.clamp(torch.exp(self.log_temperature), 0.01, 0.5)

def infonce_loss(user_embs: torch.Tensor, item_embs: torch.Tensor,
                 temperature: torch.Tensor) -> torch.Tensor:
    """
    InfoNCE contrastive loss (same as SimCLR, CLIP).

    For each (user_i, item_i) pair in the batch:
      Positive: user_i should be close to item_i
      Negatives: user_i should be far from item_j for all j ≠ i

    Loss = -log( exp(u_i · i_i / τ) / Σ_j exp(u_i · i_j / τ) )

    This treats all OTHER items in the batch as negatives.
    With batch size 512: each positive gets 511 in-batch negatives for free.
    """
    batch_size = user_embs.shape[0]
    # Compute all pairwise dot products: (batch × batch) matrix
    logits = torch.matmul(user_embs, item_embs.T) / temperature
    # Positive pairs are on the diagonal (user_i matched with item_i)
    labels = torch.arange(batch_size, device=user_embs.device)
    # Cross-entropy: row i should predict column i as the matching item
    loss = F.cross_entropy(logits, labels)
    return loss

# ── Step 5: Training ──────────────────────────────────────────────────────────
np.random.shuffle(positive_pairs)
split = int(0.9 * len(positive_pairs))
train_pairs, val_pairs = positive_pairs[:split], positive_pairs[split:]

BATCH   = 512
train_loader = DataLoader(InteractionDataset(train_pairs), batch_size=BATCH,
                          shuffle=True, num_workers=0, drop_last=True)
val_loader   = DataLoader(InteractionDataset(val_pairs),   batch_size=BATCH,
                          shuffle=False, num_workers=0, drop_last=True)

model     = TwoTowerModel(N_USER_FEATS + 1, N_ITEM_FEATS, output_dim=128).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Training pairs:   {len(train_pairs):,}")

best_val_loss = np.inf
EPOCHS = 20

for epoch in range(EPOCHS):
    model.train(); train_loss = 0
    for u_feat, i_feat, u_ids, i_ids in train_loader:
        u_feat = u_feat.to(device); i_feat = i_feat.to(device)
        optimizer.zero_grad()
        u_emb, i_emb = model(u_feat, i_feat)
        loss = infonce_loss(u_emb, i_emb, model.temperature)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()

    # Validation
    model.eval(); val_loss = 0
    with torch.no_grad():
        for u_feat, i_feat, _, _ in val_loader:
            u_emb, i_emb = model(u_feat.to(device), i_feat.to(device))
            val_loss += infonce_loss(u_emb, i_emb, model.temperature).item()

    scheduler.step()
    avg_train = train_loss / len(train_loader)
    avg_val   = val_loss   / len(val_loader)

    if avg_val < best_val_loss:
        best_val_loss = avg_val
        torch.save(model.state_dict(), '/tmp/two_tower_best.pt')

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1:3d}  train_loss={avg_train:.4f}  "
              f"val_loss={avg_val:.4f}  τ={model.temperature.item():.4f}  "
              f"LR={scheduler.get_last_lr()[0]:.6f}")

model.load_state_dict(torch.load('/tmp/two_tower_best.pt'))
print(f"\nBest val loss: {best_val_loss:.4f}")

# ── Step 6: Pre-compute all item embeddings (OFFLINE step) ──────────────────
# This runs once after training and the vectors are stored in the ANN index.
# At query time, only the user tower runs (one forward pass).
print("\nPre-computing item embeddings for all items...")
model.eval()
all_item_embeddings = []
item_feat_tensor    = torch.tensor(item_feat_matrix, dtype=torch.float32)
CHUNK = 512

with torch.no_grad():
    for start in range(0, N_ITEMS, CHUNK):
        chunk = item_feat_tensor[start:start+CHUNK].to(device)
        embs  = model.item_tower(chunk).cpu().numpy()
        all_item_embeddings.append(embs)

all_item_embeddings = np.vstack(all_item_embeddings)   # (N_ITEMS, 128)
print(f"Item embedding matrix: {all_item_embeddings.shape}")

# ── Step 7: Inference — generate recommendations for a user ─────────────────
def two_tower_recommend(user_id: int, n: int = 20) -> list:
    """
    Fast inference:
    1. Encode the user with the user tower  [~2ms]
    2. ANN search against pre-computed item embeddings  [~5ms]
    3. Return top-n items  [~1ms]
    """
    model.eval()
    u_feat = np.append(user_feat_matrix[user_id], user_account_age[user_id])
    u_tensor = torch.tensor(u_feat, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        u_emb = model.user_tower(u_tensor).cpu().numpy()[0]   # (128,)

    # Dot product against all item embeddings (cosine sim since both are L2-normalised)
    scores     = all_item_embeddings @ u_emb   # (N_ITEMS,)
    seen_items = interactions[interactions['user_id']==user_id]['item_id'].values
    scores[seen_items] = -np.inf

    top_n = np.argsort(scores)[::-1][:n]
    return [(int(i), float(scores[i])) for i in top_n]

print("\n── Two-Tower Recommendations ──")
for user_id in [0, 100, 500]:
    recs = two_tower_recommend(user_id, n=5)
    print(f"\nUser {user_id}:")
    for item_id, score in recs:
        genre = items_df.loc[item_id, 'genre']
        print(f"  Item {item_id:5d}  [{genre:<12}]  score={score:.4f}")
```

---

## 9. Solution 4: Hybrid System — Combining All Signals

**What this code does:**
In production, no single recommender is best for all users and all situations. We combine signals from all three approaches: ALS (collaborative), content-based, and two-tower. The combination handles edge cases better than any single model: collaborative handles mainstream taste, content-based handles cold-start items, two-tower handles complex cross-feature patterns.

We also add a **re-ranking layer** that applies business rules: diversity (not too many items of the same genre), freshness (boost recently published items), and a novelty penalty (reduce score of items too similar to what the user has already seen).

**Key terms explained:**
- **Score normalisation** — before combining scores from different models, we need them on the same scale. ALS scores might range 0–5; two-tower cosine scores range -1 to 1; content scores range 0–1. Min-max normalisation converts each model's scores to [0, 1] so they can be fairly combined.
- **Weighted ensemble** — multiply each model's normalised score by a weight and sum them. `final = w1×als_score + w2×content_score + w3×two_tower_score`. Weights reflect how much we trust each model (often tuned with A/B testing).
- **Diversity re-ranking** — modify the final ranking to avoid showing 20 Action movies in a row. After sorting by score, iterate through results: if the current item's genre has already appeared K times, skip it (push it down) and continue. This ensures genre diversity without completely ignoring the model's ranking.
- **Freshness boost** — multiply item scores by a small factor that is higher for recently published items. `boost = 1 + β × exp(-days_since_publish / half_life)`. β=0.1 means a brand-new item gets a 10% score bump, decaying to 0 as it ages. This helps surface new content and keeps the platform feeling fresh.
- **Novelty penalty** — reduce the score of items that are very similar (in content space) to items the user has already seen recently. Prevents the "filter bubble" where users only ever see the same type of content. Encourages serendipitous discovery.
- **A/B testing** — the standard method to evaluate recommender system changes in production. Split users randomly into two groups: Group A (control, old model) and Group B (treatment, new model). Measure engagement metrics after 2 weeks. If treatment is significantly better, deploy to everyone.

```python
import numpy as np
import pandas as pd

class HybridRecommender:
    """
    Combines ALS collaborative filtering, content-based filtering,
    and two-tower neural network into one unified recommendation pipeline.

    Also applies business rules:
      • Freshness boost   — promote recently published items
      • Diversity filter  — limit items per genre
      • Novelty penalty   — penalise items too similar to already-seen content
    """

    def __init__(self, als_model, content_norm, item_embeddings,
                 items_df,  interactions_df,
                 weights=(0.40, 0.25, 0.35),
                 freshness_beta=0.10, freshness_halflife_days=14,
                 diversity_max_per_genre=4, novelty_penalty=0.15):
        """
        weights:               (w_als, w_content, w_twotower) — sum to 1
        freshness_beta:        max fractional boost for brand-new items (0.10 = 10%)
        freshness_halflife:    days until freshness boost halves
        diversity_max_per_genre: max items of the same genre in final list
        novelty_penalty:       fraction to reduce score of items similar to seen history
        """
        self.als         = als_model
        self.cnt_norm    = content_norm       # item content vectors (normalised)
        self.tt_embs     = item_embeddings    # two-tower item embeddings (normalised)
        self.items_df    = items_df
        self.inter_df    = interactions_df
        self.w_als, self.w_cnt, self.w_tt = weights
        self.fb          = freshness_beta
        self.fhl         = freshness_halflife_days
        self.div_max     = diversity_max_per_genre
        self.nov_pen     = novelty_penalty

    def _normalise(self, scores: np.ndarray) -> np.ndarray:
        """Min-max normalise scores to [0,1] range."""
        lo, hi = scores.min(), scores.max()
        if hi == lo: return np.zeros_like(scores)
        return (scores - lo) / (hi - lo)

    def _freshness_boost(self, item_ids: np.ndarray) -> np.ndarray:
        """
        Return a per-item freshness multiplier.
        New items: multiplier = 1 + beta. Old items: multiplier → 1.0.
        Uses exponential decay: boost = beta × exp(-day / halflife)
        """
        days  = self.items_df.loc[item_ids, 'release_day'].values
        boost = self.fb * np.exp(-days / (self.fhl * 365 / 365))
        return 1.0 + boost

    def _novelty_penalty(self, item_ids: np.ndarray, user_id: int,
                         n_seen: int = 10) -> np.ndarray:
        """
        Penalise items that are very similar (in content space) to what
        the user has already seen recently. Encourages exploration.
        """
        seen = (self.inter_df[self.inter_df['user_id'] == user_id]
                .nlargest(n_seen, 'score')['item_id'].values)
        if len(seen) == 0:
            return np.ones(len(item_ids))

        # Average content vector of recently-seen items
        seen_profile  = self.cnt_norm[seen].mean(axis=0)
        # Cosine similarity between candidates and seen-item profile
        cand_vecs     = self.cnt_norm[item_ids]
        similarities  = cand_vecs @ seen_profile  # (n_candidates,)
        # Items very similar to seen history get penalised
        # High similarity → large penalty; low similarity → no penalty
        penalty = 1.0 - self.nov_pen * np.clip(similarities, 0, 1)
        return penalty

    def _diversity_filter(self, ranked_items: list, n: int) -> list:
        """
        From a ranked list, pick top-n ensuring no genre appears more than
        diversity_max times. Preserves ranking order otherwise.
        """
        genre_counts = {}
        final        = []
        for item_id, score in ranked_items:
            genre = self.items_df.loc[item_id, 'genre']
            if genre_counts.get(genre, 0) < self.div_max:
                final.append((item_id, score))
                genre_counts[genre] = genre_counts.get(genre, 0) + 1
            if len(final) == n:
                break
        return final

    def recommend(self, user_id: int, n: int = 20,
                  n_candidates: int = 200) -> list:
        """
        Full hybrid recommendation pipeline for one user.

        Stage 1 (Retrieval): each model contributes n_candidates candidates
        Stage 2 (Scoring):  blend normalised scores from all models
        Stage 3 (Re-rank):  apply freshness boost + novelty penalty + diversity
        """
        seen_items = set(self.inter_df[self.inter_df['user_id'] == user_id]['item_id'])

        # ── Stage 1: Collect candidates from each model ───────────────────
        als_recs  = self.als.recommend(user_id, n=n_candidates, exclude_seen=True)
        cnt_recs  = content_based_recommend(user_id, n=n_candidates)
        tt_recs   = two_tower_recommend(user_id, n=n_candidates)

        # Union of all candidates
        all_ids = list(set(
            [i for i,_ in als_recs] +
            [i for i,_ in cnt_recs] +
            [i for i,_ in tt_recs]
        ))
        all_ids = [i for i in all_ids if i not in seen_items]
        if not all_ids:
            return []
        all_ids = np.array(all_ids)

        # ── Stage 2: Compute and normalise scores from each model ──────────
        als_map  = {i: s for i, s in als_recs}
        cnt_map  = {i: s for i, s in cnt_recs}
        tt_map   = {i: s for i, s in tt_recs}

        als_scores = np.array([als_map.get(i,  0.0) for i in all_ids])
        cnt_scores = np.array([cnt_map.get(i,  0.0) for i in all_ids])
        tt_scores  = np.array([tt_map.get(i, -1.0) for i in all_ids])

        als_norm = self._normalise(als_scores)
        cnt_norm = self._normalise(cnt_scores)
        tt_norm  = self._normalise(tt_scores)

        # Weighted blend
        blended = self.w_als * als_norm + self.w_cnt * cnt_norm + self.w_tt * tt_norm

        # ── Stage 3: Apply business rules ─────────────────────────────────
        # 3a: Freshness boost — new items get a small score bump
        blended *= self._freshness_boost(all_ids)
        # 3b: Novelty penalty — reduce scores of items too similar to history
        blended *= self._novelty_penalty(all_ids, user_id)

        # Sort by final score
        order  = np.argsort(blended)[::-1]
        ranked = [(int(all_ids[i]), float(blended[i])) for i in order]

        # 3c: Diversity filter — cap items per genre
        final = self._diversity_filter(ranked, n)
        return final

# Instantiate the hybrid recommender
hybrid = HybridRecommender(
    als_model      = als,
    content_norm   = item_content_norm,
    item_embeddings= all_item_embeddings,
    items_df       = items_df,
    interactions_df= interactions,
    weights        = (0.40, 0.25, 0.35),   # ALS 40%, Content 25%, TwoTower 35%
    freshness_beta      = 0.10,
    freshness_halflife_days= 14,
    diversity_max_per_genre= 4,
    novelty_penalty        = 0.15
)

print("── Hybrid Recommendations ──")
for user_id in [0, 100, 500]:
    recs = hybrid.recommend(user_id, n=10)
    genres_seen = [items_df.loc[i,'genre'] for i,_ in recs]
    from collections import Counter
    genre_dist = Counter(genres_seen)
    print(f"\nUser {user_id}  ({len(recs)} recs, genre spread: {dict(genre_dist)}):")
    for item_id, score in recs[:5]:
        genre = items_df.loc[item_id, 'genre']
        print(f"  Item {item_id:5d}  [{genre:<12}]  hybrid_score={score:.4f}")
```

---

## 10. Evaluation Strategy

**What this code does:**
Standard classification metrics (accuracy, F1) do not apply to recommendation. We use retrieval-style metrics that measure: how highly are items the user actually engaged with ranked in our predictions? We also compute the business metric directly — predicted watch time — since that is what YouTube and Netflix actually optimise.

**Key terms explained:**
- **Recall@K** — of all items a user actually engaged with in the test period, what fraction appear in our top-K recommendations? Recall@20 = 0.35 means: for 35% of test interactions, the item appeared in our top-20. Primary metric for candidate retrieval.
- **Precision@K** — of our top-K recommendations, what fraction did the user actually engage with? Precision@10 = 0.15 means: 1.5 out of 10 recommended items are genuinely interesting to the user. Matters when screen space is limited.
- **NDCG@K (Normalised Discounted Cumulative Gain)** — rewards putting highly-engaged items at the top of the ranking. An item the user watched 90% of showing at rank 1 is much better than at rank 10. "Discounted" because rank 2 is half as good as rank 1, rank 3 is a third as good, etc.
- **MRR (Mean Reciprocal Rank)** — average of 1/rank where rank = position of the first item the user actually clicked/played. MRR=0.4 means the first relevant item appears at rank 2.5 on average. Good for "did we get at least one thing right?"
- **Hit Rate@K** — for at least one of our top-K recommendations, did the user engage with it? Binary. Hit Rate@10 = 0.65 means: for 65% of users, at least one of our top-10 recommendations was something they genuinely wanted to watch.
- **Coverage** — what fraction of all items does the model ever recommend to anyone? Low coverage = the model only ever recommends the same popular items = discovery is poor.
- **Train-test temporal split** — interactions from the last N days are test; earlier interactions are train. This is the correct split: the model learns from the past and is evaluated on future behaviour.

```python
import numpy as np
import pandas as pd
from collections import defaultdict

# ── Temporal train/test split ────────────────────────────────────────────────
# Use last 20% of timestamps as test set — never split randomly for recommenders
split_time = events_df['timestamp'].quantile(0.80)
train_events = events_df[events_df['timestamp'] <= split_time]
test_events  = events_df[events_df['timestamp'] >  split_time]

# Ground truth: items each user engaged with positively in the test period
test_ground_truth = (test_events[test_events['pct_consumed'] > 0.5]
                     .groupby('user_id')['item_id']
                     .apply(set).to_dict())

print(f"Train events: {len(train_events):,}  ({train_events['timestamp'].min():.0f}–{split_time:.0f})")
print(f"Test events:  {len(test_events):,}   ({split_time:.0f}–{test_events['timestamp'].max():.0f})")
print(f"Users in test ground truth: {len(test_ground_truth):,}")

def evaluate_recommender(recommend_fn, model_name: str,
                         test_gt: dict, k_values=[5,10,20],
                         n_eval_users: int = 200) -> dict:
    """
    Evaluate a recommendation function against ground truth.

    recommend_fn: callable(user_id, n) → list of (item_id, score) tuples
    test_gt:      dict mapping user_id → set of item_ids they engaged with in test
    """
    metrics = defaultdict(list)

    # Sample users who have test ground truth AND enough history to recommend for
    eval_users = [u for u in list(test_gt.keys())[:n_eval_users]
                  if len(test_gt[u]) > 0]

    for user_id in eval_users:
        relevant = test_gt[user_id]

        try:
            recs = recommend_fn(user_id, n=max(k_values))
        except Exception:
            continue

        rec_ids = [item_id for item_id, _ in recs]

        for k in k_values:
            top_k = set(rec_ids[:k])
            hits  = len(top_k & relevant)

            # Recall@K: fraction of relevant items that appear in top-K
            metrics[f'recall@{k}'].append(hits / max(len(relevant), 1))
            # Precision@K: fraction of top-K that are relevant
            metrics[f'precision@{k}'].append(hits / k)
            # Hit Rate@K: did ANY of top-K match? (binary)
            metrics[f'hit_rate@{k}'].append(float(hits > 0))

        # NDCG@10: accounts for rank position of relevant items
        # An item at rank 1 contributes log(2)/log(2)=1.0; at rank 5: log(2)/log(6)=0.39
        gains = [1.0 if rec_ids[r] in relevant else 0.0
                 for r in range(min(10, len(rec_ids)))]
        dcg   = sum(g / np.log2(r + 2) for r, g in enumerate(gains))
        ideal = sum(1.0 / np.log2(r + 2) for r in range(min(len(relevant), 10)))
        metrics['ndcg@10'].append(dcg / ideal if ideal > 0 else 0.0)

        # MRR: 1 / rank of first relevant item
        mrr = 0.0
        for rank, item_id in enumerate(rec_ids, 1):
            if item_id in relevant:
                mrr = 1.0 / rank
                break
        metrics['mrr'].append(mrr)

    results = {k: float(np.mean(v)) for k, v in metrics.items() if v}
    results['n_users_evaluated'] = len(eval_users)

    print(f"\n{'='*55}")
    print(f"  {model_name}")
    print(f"{'='*55}")
    for metric, value in sorted(results.items()):
        if metric == 'n_users_evaluated': continue
        bar = '█' * int(value * 40)
        print(f"  {metric:<15} {value:.4f}  {bar}")
    print(f"  Evaluated on {results['n_users_evaluated']} users")
    return results

# Evaluate each model
# Note: ALS and two-tower were trained on ALL data; in practice retrain on train_events only
als_results = evaluate_recommender(
    lambda u, n: als.recommend(u, n=n), "ALS Collaborative Filtering",
    test_ground_truth, k_values=[5,10,20])

content_results = evaluate_recommender(
    lambda u, n: content_based_recommend(u, n=n), "Content-Based Filtering",
    test_ground_truth, k_values=[5,10,20])

hybrid_results = evaluate_recommender(
    lambda u, n: hybrid.recommend(u, n=n), "Hybrid (ALS + Content + TwoTower)",
    test_ground_truth, k_values=[5,10,20])

# Popularity baseline — recommend the most globally popular items
global_popular = (interactions.groupby('item_id')['score'].sum()
                  .sort_values(ascending=False).index.tolist())
pop_results = evaluate_recommender(
    lambda u, n: [(i, 1.0) for i in global_popular[:n]], "Popularity Baseline",
    test_ground_truth, k_values=[5,10,20])

print("""
Target benchmarks for a production system:
  hit_rate@10    ≥ 0.40  (40% of users find at least 1 relevant item in top 10)
  recall@20      ≥ 0.25  (25% of test interactions appear in top 20 recommendations)
  ndcg@10        ≥ 0.20
  mrr            ≥ 0.15
""")

# Coverage analysis: what fraction of items does each model ever recommend?
all_recommended = set()
for user_id in list(test_ground_truth.keys())[:500]:
    recs = hybrid.recommend(user_id, n=20)
    all_recommended.update(i for i,_ in recs)
coverage = len(all_recommended) / N_ITEMS
print(f"Catalog coverage (fraction of items ever recommended): {coverage:.2%}")
print(f"  Low coverage = popularity bias = users only see the same popular items")
print(f"  Target: > 20% coverage to serve diverse taste groups")
```

---

## 11. The Cold Start Problem & Solutions

**What this code does:**
Cold start is the single hardest problem in recommendation. New users have no history; new items have no interactions. A pure collaborative filter is useless in both cases. We show three concrete solutions: (1) onboarding survey for new users, (2) content-based fallback for new items, (3) exploration strategy that learns about new users quickly.

**Key terms explained:**
- **User cold start** — a brand new user has zero interaction history. The collaborative filter has nothing to work from. The model would give this user the same recommendations as every other user with no history.
- **Item cold start** — a newly published video/song/article has zero plays, zero likes, zero data. The collaborative filter cannot recommend it to anyone (no users have played it to create a signal). But new content is exactly what the platform wants to promote.
- **Onboarding survey** — ask new users a few questions ("Which genres do you like? Pick 3 creators you follow") and map their answers directly to seed the content-based profile. One survey response = a starting point for collaborative filtering.
- **Popularity-based warm-up** — for new users: show globally popular and critically acclaimed content first. Use their reactions (plays, skips, likes) to quickly learn their taste. After 20 interactions, the collaborative filter has enough signal.
- **Epsilon-greedy exploration** — with probability ε, ignore the model's recommendation and show a random item from an under-explored category. This intentionally sacrifices some short-term engagement to learn the user's full taste. Standard in multi-armed bandit and reinforcement learning.
- **Content-based bridge** — for cold-start items: use only content features (genre, creator, title embedding) to find which users would most likely enjoy the item, and show it to those users. Once it accumulates 50+ interactions, the collaborative signal kicks in.
- **Bayesian average rating** — for new items with very few ratings: `bayesian_rating = (C × m + Σ_ratings) / (C + n)` where m = global average rating, C = shrinkage factor, n = number of ratings. With few ratings (n << C), the estimate is pulled toward the global mean. With many ratings, it converges to the true mean. Prevents new items from having misleading extreme scores.

```python
import numpy as np

# ── Solution A: New User Cold Start via Onboarding Survey ────────────────────
def handle_new_user_survey(preferred_genres: list, preferred_creators: list,
                            account_device: str = 'mobile',
                            n_recs: int = 20) -> list:
    """
    For brand-new users: map survey answers to a content-based profile.
    No interaction history needed.

    preferred_genres:  e.g., ['Action', 'SciFi']
    preferred_creators: e.g., ['creator_0042', 'creator_0107']
    """
    # Build a synthetic profile vector from survey answers
    profile = np.zeros(item_content_norm.shape[1], dtype=np.float32)

    # Genre preference: activate the corresponding genre dimensions
    for genre in preferred_genres:
        if genre in GENRES:
            genre_idx = GENRES.index(genre)
            profile[genre_idx] += 2.0     # weight 2× as in our item vectors

    # Creator preference: use the creator embedding
    for creator in preferred_creators:
        if creator in creator_emb:
            creator_start = genre_ohe.shape[1] + numeric_scaled.shape[1]
            profile[creator_start:creator_start+16] += creator_emb[creator]

    # Normalise to unit length
    norm = np.linalg.norm(profile)
    if norm > 0:
        profile /= norm

    # Find items most similar to the stated preferences
    recs = find_similar_content(profile, k=n_recs)
    return recs

# Test cold start for a new user who likes SciFi and Action
new_user_recs = handle_new_user_survey(
    preferred_genres   = ['SciFi', 'Action'],
    preferred_creators = ['creator_0042'],
    n_recs             = 10
)
print("── New User Cold Start (survey: SciFi + Action) ──")
for item_id, score in new_user_recs[:5]:
    genre = items_df.loc[item_id, 'genre']
    print(f"  Item {item_id:5d}  [{genre:<12}]  content_sim={score:.4f}")

# ── Solution B: New Item Cold Start ─────────────────────────────────────────
def promote_new_item(item_id: int, n_target_users: int = 1000) -> list:
    """
    For a brand-new item with zero interactions:
    Find users whose taste profile best matches this item's content,
    and target them for the item's initial distribution.

    This is how Spotify pushes new releases to likely fans,
    and how YouTube decides who sees a new video first.
    """
    item_vec  = item_content_norm[item_id]   # content vector for the new item
    # For each user, compute similarity between their content profile and this item
    target_users = []
    for u in range(min(5000, N_USERS)):      # sample for speed
        profile = build_user_content_profile(u)
        if np.linalg.norm(profile) == 0:
            continue
        sim = float(item_vec @ (profile / (np.linalg.norm(profile) + 1e-8)))
        target_users.append((u, sim))

    target_users.sort(key=lambda x: -x[1])
    return target_users[:n_target_users]

new_item = N_ITEMS - 1   # last item = newly added
genre = items_df.loc[new_item, 'genre']
print(f"\n── New Item Cold Start: Item {new_item} [{genre}] ──")
targets = promote_new_item(new_item, n_target_users=5)
print(f"Top 5 users to show this item to first:")
for user_id, score in targets:
    tg = users_df.loc[user_id, 'taste_group']
    print(f"  User {user_id:5d}  (taste group {tg})  match_score={score:.4f}")

# ── Solution C: Bayesian average for new item scoring ────────────────────────
def bayesian_avg_rating(item_id: int, C: float = 50.0) -> float:
    """
    Stabilise ratings for items with few interactions.
    C = 50: an item needs 50 ratings before its average differs meaningfully
    from the global average. A 1-rating 5-star item gets only slightly above average.
    """
    global_mean = interactions['score'].mean()
    item_scores = interactions[interactions['item_id'] == item_id]['score']
    n           = len(item_scores)
    if n == 0:
        return global_mean
    return (C * global_mean + item_scores.sum()) / (C + n)

# Compare raw avg vs bayesian avg for items with different interaction counts
print("\n── Bayesian Rating Smoothing ──")
print(f"Global mean score: {interactions['score'].mean():.3f}")
for item_id in [0, 100, 500, N_ITEMS-1]:
    n_ratings   = len(interactions[interactions['item_id'] == item_id])
    raw_avg     = interactions[interactions['item_id']==item_id]['score'].mean() if n_ratings else 0
    bayes_avg   = bayesian_avg_rating(item_id)
    print(f"  Item {item_id:5d}  n_ratings={n_ratings:4d}  "
          f"raw_avg={raw_avg:.3f}  bayesian_avg={bayes_avg:.3f}")

print("""
Cold Start Strategy Summary:
  New User (0 interactions)   → Onboarding survey → content-based profile
  New User (1–20 interactions)→ Content-based + popularity + epsilon exploration
  New User (20+ interactions) → Full collaborative + two-tower kicks in
  New Item (0 interactions)   → Content-based targeting (reach likely fans first)
  New Item (50+ interactions) → ALS collaborative signal becomes available
  New Item (500+ interactions)→ Two-tower embedding becomes reliable
""")
```

---

## 12. Production Architecture

**What this code does:**
Shows the full production pipeline as it exists at YouTube/Netflix/Spotify scale. This section explains the engineering architecture — how a recommendation request flows through multiple systems in under 100ms — and how the models are kept up-to-date as user preferences change.

**Key terms explained:**
- **Feature store** — a centralised database (usually Redis + a data warehouse like BigQuery/Snowflake) that stores pre-computed user and item features. When a recommendation request arrives, the user's feature vector is fetched from Redis (<1ms) rather than computed from scratch.
- **Embedding store (Vector DB)** — stores pre-computed item embedding vectors (from the two-tower item tower and ALS item factors). Queried with ANN search (HNSW) at inference time. Updated nightly when models are retrained.
- **Near-real-time feature updates** — user features (recent watch history, current session context) are updated every few minutes via a streaming pipeline (Kafka + Flink). Ensures the model knows the user just watched a horror movie and should shift recommendations accordingly.
- **Model serving** — the user tower and ALS model are deployed as microservices (TorchServe, TensorFlow Serving). They receive a user feature vector and return an embedding in <5ms.
- **Cascade architecture** — multiple ranking stages each operating on fewer candidates than the previous. Each stage is more expensive but more accurate. Stage 1 (retrieval): 10M → 500. Stage 2 (ranking): 500 → 20. Stage 3 (re-ranking): 20 → 20 (reordered).
- **Continuous training** — retrain models on a schedule (daily for fast-moving content like news; weekly for slower-moving content like movies). New interaction data constantly arrives; models that don't retrain become stale.
- **Shadow mode / canary deployment** — deploy a new model to 1–5% of users first while the old model serves everyone else. Compare engagement metrics. If the new model is better, gradually increase its traffic share. If it's worse, roll back instantly.

```python
print("""
╔══════════════════════════════════════════════════════════════════════════╗
║        PRODUCTION RECOMMENDATION SYSTEM ARCHITECTURE                    ║
║        (YouTube / Netflix / Spotify scale)                               ║
╚══════════════════════════════════════════════════════════════════════════╝

USER REQUEST FLOW (target < 100ms end-to-end)
═════════════════════════════════════════════

  User opens app
        ↓ [~5ms]
  Feature Store (Redis)
    → fetch user_id → user feature vector (genre affinities, recent history)
    → fetch current session context (device, time of day, last 5 items played)

        ↓ [~5ms]
  Retrieval Service (Stage 1: 10M → 500 candidates)
    → User Tower inference: user features → 128-dim user embedding
    → HNSW ANN search on pre-computed item embeddings
    → Runs in PARALLEL:
        A. Two-tower ANN: top-200 by embedding similarity
        B. ALS collaborative: top-200 by matrix factorization score
        C. Content-based: top-100 by content vector similarity
    → Union + dedup = ~400 candidates

        ↓ [~30ms]
  Ranking Service (Stage 2: 500 → 20)
    → LambdaRank / Neural ranker with all available features:
        User features: genre affinities, watch history depth, device
        Item features: genre, creator, duration, popularity, recency
        Context features: time of day, day of week, session length
        Cross features: (user_genre_pref × item_genre), (user_device × item_type)
    → Output: 500 items with precise relevance scores

        ↓ [~5ms]
  Re-ranking (Stage 3: 20 → 20 reordered)
    → Diversity: max 3 items per genre, max 2 per creator
    → Freshness: boost items published < 7 days ago
    → Novelty: penalise items too similar to last 5 played
    → Business rules: sponsored slots, A/B test variants

        ↓
  Response: top-20 item IDs + their features (for UI rendering)

════════════════════════════════════════════════════════════════════════════
OFFLINE PIPELINES (keep models and features fresh)
════════════════════════════════════════════════════════════════════════════

  Every 5 minutes:
    • Kafka streams new interactions → update user feature vectors in Redis
    • Session context updated in real-time (last 5 items, current device)

  Every night (incremental):
    • Re-train ALS on last 90 days of interactions  [~2 hours on 8 GPUs]
    • Re-run two-tower item tower on all items  [~30 min]
    • Update HNSW index with new item embeddings  [~15 min]
    • Push new item embeddings to embedding store  [~5 min]

  Every week (full retrain):
    • Full two-tower training from scratch  [~4 hours on 8 GPUs]
    • A/B test new model on 5% of users  [7-day test]
    • If NDCG@10 improves > 0.5%: promote to 100% traffic

════════════════════════════════════════════════════════════════════════════
METRICS DASHBOARD (what the team monitors in production)
════════════════════════════════════════════════════════════════════════════

  Engagement metrics (primary):
    • CTR (click-through rate):          % of recommendations clicked
    • Watch-through rate:                % of item consumed after click
    • Session length:                    total time in app per session
    • Return rate:                       % of users who come back next day

  Recommendation quality metrics:
    • NDCG@10:                           online, sampled daily
    • Recall@20:                         offline, computed weekly
    • Catalog coverage:                  % of items recommended to ≥1 user/week

  Business health metrics:
    • Cold-start rate:                   % of requests served with < 10 interactions
    • Serendipity index:                 % of recommendations outside user's typical genres
    • Creator diversity:                 Gini coefficient of creator-level impressions
    • Feedback loop metric:              genre concentration over 30 days (detect filter bubble)
""")

# ── Simulate online A/B test analysis ───────────────────────────────────────
def simulate_ab_test(control_ndcg: float, treatment_ndcg: float,
                     n_users: int = 50000, alpha: float = 0.05):
    """
    Statistical significance test for A/B test results.
    Tests whether the treatment model's NDCG improvement is statistically significant.

    Standard error of mean NDCG ≈ std_dev / sqrt(n_users)
    T-test: t = (treatment_mean - control_mean) / pooled_std_error
    """
    from scipy import stats

    # Simulate NDCG scores for each user group (normal distribution around means)
    std_dev = 0.12     # typical per-user NDCG standard deviation
    ctrl    = np.random.normal(control_ndcg,   std_dev, n_users)
    treat   = np.random.normal(treatment_ndcg, std_dev, n_users)

    # Two-sample t-test: are the means significantly different?
    t_stat, p_val = stats.ttest_ind(treat, ctrl)
    rel_lift      = (treatment_ndcg - control_ndcg) / control_ndcg

    print(f"\n── A/B Test Analysis ──")
    print(f"  Control NDCG@10:   {ctrl.mean():.4f} ± {ctrl.std():.4f}")
    print(f"  Treatment NDCG@10: {treat.mean():.4f} ± {treat.std():.4f}")
    print(f"  Relative lift:     {rel_lift:+.2%}")
    print(f"  p-value:           {p_val:.4f}")
    decision = "SHIP IT ✅" if p_val < alpha and rel_lift > 0 else "DO NOT SHIP ❌"
    print(f"  Decision (α={alpha}): {decision}")

simulate_ab_test(control_ndcg=0.220, treatment_ndcg=0.234, n_users=50000)
```

---

*Last updated: May 2026 | Pramod Modi — Staff ML Engineer, ServiceNow*
