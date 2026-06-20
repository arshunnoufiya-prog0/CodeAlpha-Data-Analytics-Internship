"""
=============================================================
CodeAlpha Internship — Task 4: Sentiment Analysis
=============================================================
Uses VADER (rule-based, no training needed) + optional
TextBlob for comparison. Analyses Amazon-style product
reviews. Generates full charts and a summary report.
=============================================================
Install dependencies:
    pip install vaderSentiment textblob pandas matplotlib seaborn wordcloud nltk
    python -m nltk.downloader punkt stopwords
=============================================================
"""

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

# ── 1. Install check / import VADER ───────────────────────
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    vader_available = True
except ImportError:
    print("⚠️  vaderSentiment not found. Run: pip install vaderSentiment")
    vader_available = False

try:
    from textblob import TextBlob
    textblob_available = True
except ImportError:
    textblob_available = False

try:
    from wordcloud import WordCloud
    wordcloud_available = True
except ImportError:
    wordcloud_available = False

SAVE_DPI = 150
sns.set_theme(style="whitegrid")


# ── 2. Sample dataset ─────────────────────────────────────
SAMPLE_REVIEWS = [
    {"review": "Absolutely love this product! Works perfectly and arrived fast.", "stars": 5},
    {"review": "Great quality for the price. Would definitely buy again.", "stars": 5},
    {"review": "Good product overall, though the packaging was a bit damaged.", "stars": 4},
    {"review": "Decent item. Does what it says on the tin.", "stars": 3},
    {"review": "It's okay, nothing special. Expected better for this price.", "stars": 3},
    {"review": "Not what I expected. Instructions were confusing and unclear.", "stars": 2},
    {"review": "Stopped working after two days. Very disappointing.", "stars": 1},
    {"review": "Terrible quality! Complete waste of money. Returning immediately.", "stars": 1},
    {"review": "Fantastic! Exceeded all expectations. Highly recommend to everyone.", "stars": 5},
    {"review": "Product is fine but delivery took forever. Very frustrating experience.", "stars": 2},
    {"review": "Amazing build quality and great customer service. Five stars!", "stars": 5},
    {"review": "Average product. Nothing outstanding but it works as expected.", "stars": 3},
    {"review": "Broke after one week. Poor manufacturing and weak materials.", "stars": 1},
    {"review": "Super happy with this purchase! Looks exactly like in the photos.", "stars": 5},
    {"review": "Not bad, but I've seen better products at the same price point.", "stars": 3},
    {"review": "Perfect gift! My friend loved it. Very fast shipping too.", "stars": 5},
    {"review": "The colour is slightly off from what was shown, but otherwise fine.", "stars": 3},
    {"review": "Absolute rubbish. Don't buy this. Total scam.", "stars": 1},
    {"review": "Really satisfied. Solid construction and easy to use.", "stars": 4},
    {"review": "It's okay for occasional use, not for heavy-duty applications.", "stars": 3},
    {"review": "Wonderful product! Will be ordering more for gifts.", "stars": 5},
    {"review": "Arrived broken. Customer support was unhelpful and rude.", "stars": 1},
    {"review": "Good value. Performs well and looks nice on my desk.", "stars": 4},
    {"review": "Mediocre at best. Feels cheap and flimsy.", "stars": 2},
    {"review": "Outstanding! Best purchase I've made this year without doubt.", "stars": 5},
]


# ── 3. Text pre-processing ────────────────────────────────
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)       # remove URLs
    text = re.sub(r"[^a-z\s']", "", text)             # keep letters & apostrophes
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── 4. VADER Sentiment Analysis ───────────────────────────
def vader_sentiment(text, analyzer):
    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]
    if compound >= 0.05:
        label = "Positive"
    elif compound <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"
    return compound, label, scores["pos"], scores["neg"], scores["neu"]


# ── 5. TextBlob (optional backup) ─────────────────────────
def textblob_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity        # -1 to 1
    subjectivity = blob.sentiment.subjectivity
    if polarity > 0.05:
        label = "Positive"
    elif polarity < -0.05:
        label = "Negative"
    else:
        label = "Neutral"
    return polarity, subjectivity, label


# ── 6. Build result dataframe ─────────────────────────────
def analyse_reviews(reviews):
    analyzer = SentimentIntensityAnalyzer() if vader_available else None
    results = []

    for r in reviews:
        raw = r["review"]
        clean = clean_text(raw)
        row = {"Review": raw, "Stars": r["stars"], "Cleaned": clean}

        if analyzer:
            cmp, lbl, pos, neg, neu = vader_sentiment(clean, analyzer)
            row.update({"VADER_Compound": cmp, "VADER_Label": lbl,
                        "VADER_Pos": pos, "VADER_Neg": neg, "VADER_Neu": neu})

        if textblob_available:
            pol, subj, tb_lbl = textblob_sentiment(clean)
            row.update({"TB_Polarity": pol, "TB_Subjectivity": subj, "TB_Label": tb_lbl})

        results.append(row)

    return pd.DataFrame(results)


# ── 7. Visualisations ──────────────────────────────────────
def save(name):
    plt.tight_layout()
    plt.savefig(name, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  💾 Saved: {name}")


def chart_sentiment_distribution(df):
    counts = df["VADER_Label"].value_counts()
    colors = {"Positive": "#2ECC71", "Neutral": "#F39C12", "Negative": "#E74C3C"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bar
    axes[0].bar(counts.index, counts.values,
                color=[colors[l] for l in counts.index], edgecolor="black")
    axes[0].set_title("Sentiment Distribution (Bar)")
    axes[0].set_ylabel("Count")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 0.2, str(v), ha="center", fontweight="bold")

    # Pie
    axes[1].pie(counts.values, labels=counts.index, autopct="%1.1f%%",
                colors=[colors[l] for l in counts.index],
                wedgeprops=dict(edgecolor="white", linewidth=2), startangle=90)
    axes[1].set_title("Sentiment Distribution (Pie)")

    plt.suptitle("VADER Sentiment Analysis Results", fontsize=14, fontweight="bold")
    save("sentiment1_distribution.png")


def chart_compound_scores(df):
    df_sorted = df.sort_values("VADER_Compound").reset_index(drop=True)
    colors = df_sorted["VADER_Label"].map(
        {"Positive": "#2ECC71", "Neutral": "#F39C12", "Negative": "#E74C3C"})

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(df_sorted)), df_sorted["VADER_Compound"], color=colors, edgecolor="none")
    ax.axhline(0.05, color="green", linestyle="--", linewidth=1, label="Positive threshold")
    ax.axhline(-0.05, color="red", linestyle="--", linewidth=1, label="Negative threshold")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("VADER Compound Scores per Review (sorted)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Review Index")
    ax.set_ylabel("Compound Score (−1 to +1)")
    ax.legend()
    save("sentiment2_compound_scores.png")


def chart_stars_vs_compound(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x="Stars", y="VADER_Compound", palette="RdYlGn", ax=ax)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("VADER Compound Score vs. Star Rating", fontsize=14, fontweight="bold")
    ax.set_xlabel("Star Rating")
    ax.set_ylabel("Compound Score")
    save("sentiment3_stars_vs_compound.png")


def chart_emotion_breakdown(df):
    avg = df[["VADER_Pos", "VADER_Neg", "VADER_Neu"]].mean()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(["Positive", "Negative", "Neutral"], avg.values,
           color=["#2ECC71", "#E74C3C", "#95A5A6"], edgecolor="black")
    ax.set_title("Average Emotion Scores Across All Reviews", fontsize=14, fontweight="bold")
    ax.set_ylabel("Average Score")
    for i, v in enumerate(avg.values):
        ax.text(i, v + 0.003, f"{v:.3f}", ha="center", fontweight="bold")
    save("sentiment4_emotion_breakdown.png")


def chart_wordcloud(df, sentiment_label, filename):
    if not wordcloud_available:
        print(f"  ⚠️  wordcloud not installed — skipping {filename}")
        return
    subset = df[df["VADER_Label"] == sentiment_label]["Cleaned"]
    text = " ".join(subset)
    if not text.strip():
        return
    # Simple stopword removal
    stopwords = {"the", "a", "an", "and", "is", "it", "in", "of", "to",
                 "for", "this", "was", "i", "my", "but", "on", "with", "very"}
    words = [w for w in text.split() if w not in stopwords and len(w) > 2]
    text_clean = " ".join(words)

    color_map = {"Positive": "Greens", "Negative": "Reds", "Neutral": "Oranges"}
    wc = WordCloud(width=800, height=400, background_color="white",
                   colormap=color_map.get(sentiment_label, "Blues"),
                   max_words=80).generate(text_clean)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(f"Word Cloud — {sentiment_label} Reviews",
                 fontsize=14, fontweight="bold")
    save(filename)


def chart_textblob_comparison(df):
    if not textblob_available:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Agreement
    agree = (df["VADER_Label"] == df["TB_Label"]).sum()
    disagree = len(df) - agree
    axes[0].pie([agree, disagree], labels=["Agree", "Disagree"],
                autopct="%1.0f%%", colors=["#2ECC71", "#E74C3C"],
                wedgeprops=dict(edgecolor="white"))
    axes[0].set_title("VADER vs TextBlob — Agreement")

    # Scatter — compound vs polarity
    axes[1].scatter(df["VADER_Compound"], df["TB_Polarity"],
                    alpha=0.6, c=df["Stars"], cmap="RdYlGn", s=60)
    axes[1].axhline(0, color="gray", linewidth=0.7, linestyle="--")
    axes[1].axvline(0, color="gray", linewidth=0.7, linestyle="--")
    axes[1].set_xlabel("VADER Compound Score")
    axes[1].set_ylabel("TextBlob Polarity")
    axes[1].set_title("VADER vs TextBlob Polarity Correlation")
    plt.colorbar(axes[1].collections[0], ax=axes[1], label="Star Rating")

    plt.suptitle("Model Comparison: VADER vs TextBlob", fontsize=14, fontweight="bold")
    save("sentiment6_model_comparison.png")


def print_summary(df):
    print("\n" + "=" * 55)
    print("  SENTIMENT ANALYSIS — SUMMARY REPORT")
    print("=" * 55)
    print(f"  Total reviews analysed : {len(df)}")
    if "VADER_Label" in df.columns:
        vc = df["VADER_Label"].value_counts()
        for lbl, cnt in vc.items():
            pct = cnt / len(df) * 100
            print(f"  {lbl:10s}: {cnt:3d} ({pct:.1f}%)")
        print(f"\n  Avg compound score : {df['VADER_Compound'].mean():.4f}")
        print(f"  Most positive review:")
        best = df.loc[df["VADER_Compound"].idxmax()]
        print(f"    \"{best['Review'][:80]}...\"")
        print(f"    Score: {best['VADER_Compound']:.4f}")
        print(f"\n  Most negative review:")
        worst = df.loc[df["VADER_Compound"].idxmin()]
        print(f"    \"{worst['Review'][:80]}...\"")
        print(f"    Score: {worst['VADER_Compound']:.4f}")


# ── Main ──────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  CodeAlpha Task 4 — Sentiment Analysis")
    print("=" * 55)

    if not vader_available:
        print("  ❌ Cannot continue without vaderSentiment.")
        print("     Run: pip install vaderSentiment")
        return

    print(f"\n  Analysing {len(SAMPLE_REVIEWS)} product reviews...\n")
    df = analyse_reviews(SAMPLE_REVIEWS)

    # Save results
    df.to_csv("sentiment_results.csv", index=False)
    print("  💾 Results saved to: sentiment_results.csv\n")

    # Charts
    chart_sentiment_distribution(df)
    chart_compound_scores(df)
    chart_stars_vs_compound(df)
    chart_emotion_breakdown(df)
    chart_wordcloud(df, "Positive", "sentiment5a_wordcloud_positive.png")
    chart_wordcloud(df, "Negative", "sentiment5b_wordcloud_negative.png")
    chart_textblob_comparison(df)

    print_summary(df)
    print("\n✅ Task 4 complete!")
    print("   Attach charts + sentiment_results.csv to your GitHub repo.")


if __name__ == "__main__":
    main()
