"""
=============================================================
CodeAlpha Internship — Task 3: Data Visualization
=============================================================
Generates 8 distinct, publication-quality charts using the
books_dataset.csv (Task 1 output) or Titanic (auto-fallback).
Covers: bar, pie, scatter, line, heatmap, violin, pairplot,
        and a multi-panel summary dashboard.
=============================================================
Install dependencies:
    pip install pandas numpy matplotlib seaborn
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── Palette & style ───────────────────────────────────────
sns.set_theme(style="whitegrid", palette="Set2")
PALETTE = sns.color_palette("Set2")
SAVE_DPI = 150


def load_data():
    try:
        df = pd.read_csv("books_dataset.csv")
        print("📂 Loaded: books_dataset.csv")
        return df, "books"
    except FileNotFoundError:
        url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
        df = pd.read_csv(url)
        print("📂 Loaded: Titanic dataset (demo fallback)")
        return df, "titanic"


# ── Chart helpers ──────────────────────────────────────────
def save(name):
    plt.tight_layout()
    plt.savefig(name, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  💾 Saved: {name}")


# ============================================================
# BOOKS VISUALIZATIONS
# ============================================================

def books_bar_avg_price_by_rating(df):
    """Bar chart — average price per star rating."""
    avg = df.groupby("Rating (1-5)")["Price (£)"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(avg["Rating (1-5)"].astype(str), avg["Price (£)"],
                  color=PALETTE[:5], edgecolor="black", width=0.6)
    ax.bar_label(bars, fmt="£%.2f", padding=4, fontsize=10)
    ax.set_title("Average Book Price by Star Rating", fontsize=14, fontweight="bold")
    ax.set_xlabel("Rating (Stars)")
    ax.set_ylabel("Average Price (£)")
    save("viz1_bar_avg_price_by_rating.png")


def books_pie_rating_distribution(df):
    """Pie chart — distribution of star ratings."""
    counts = df["Rating (1-5)"].value_counts().sort_index()
    labels = [f"{i} ★" for i in counts.index]
    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        counts, labels=labels, autopct="%1.1f%%",
        colors=PALETTE[:5], startangle=140,
        wedgeprops=dict(edgecolor="white", linewidth=2)
    )
    for at in autotexts:
        at.set_fontsize(11)
    ax.set_title("Distribution of Book Ratings", fontsize=14, fontweight="bold")
    save("viz2_pie_rating_distribution.png")


def books_scatter_price_vs_rating(df):
    """Scatter plot — price vs rating with jitter."""
    jitter = df["Rating (1-5)"] + np.random.uniform(-0.2, 0.2, len(df))
    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(jitter, df["Price (£)"], alpha=0.5, c=df["Rating (1-5)"],
                    cmap="Set1", edgecolors="none", s=40)
    plt.colorbar(sc, ax=ax, label="Rating")
    ax.set_title("Price vs. Rating (with jitter)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Rating (Stars)")
    ax.set_ylabel("Price (£)")
    ax.set_xticks([1, 2, 3, 4, 5])
    save("viz3_scatter_price_vs_rating.png")


def books_line_cumulative_books(df):
    """Line chart — cumulative book count by price (sorted)."""
    sorted_prices = df["Price (£)"].sort_values().reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(sorted_prices.values, range(1, len(sorted_prices) + 1),
            color="#2196F3", linewidth=2)
    ax.fill_between(sorted_prices.values, range(1, len(sorted_prices) + 1),
                    alpha=0.15, color="#2196F3")
    ax.set_title("Cumulative Book Count by Price", fontsize=14, fontweight="bold")
    ax.set_xlabel("Price (£)")
    ax.set_ylabel("Cumulative Books")
    save("viz4_line_cumulative_price.png")


def books_violin_price_by_rating(df):
    """Violin plot — price distribution per rating."""
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.violinplot(data=df, x="Rating (1-5)", y="Price (£)",
                   palette="Set2", inner="quartile", ax=ax)
    ax.set_title("Price Distribution per Rating (Violin)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Rating (Stars)")
    ax.set_ylabel("Price (£)")
    save("viz5_violin_price_by_rating.png")


def books_availability_bar(df):
    """Horizontal bar — availability breakdown."""
    avail = df["Availability"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 4))
    avail.plot(kind="barh", color=PALETTE[:len(avail)], edgecolor="black", ax=ax)
    ax.set_title("Book Availability Breakdown", fontsize=14, fontweight="bold")
    ax.set_xlabel("Count")
    save("viz6_bar_availability.png")


def books_price_histogram(df):
    """Histogram + KDE — price distribution."""
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(df["Price (£)"], bins=30, kde=True, color="#4C72B0",
                 edgecolor="white", ax=ax)
    ax.axvline(df["Price (£)"].mean(), color="red", linestyle="--", label=f"Mean £{df['Price (£)'].mean():.2f}")
    ax.axvline(df["Price (£)"].median(), color="orange", linestyle="--", label=f"Median £{df['Price (£)'].median():.2f}")
    ax.legend()
    ax.set_title("Book Price Distribution with KDE", fontsize=14, fontweight="bold")
    ax.set_xlabel("Price (£)")
    ax.set_ylabel("Frequency")
    save("viz7_histogram_price.png")


def books_dashboard(df):
    """4-panel summary dashboard."""
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("📚 Books Dataset — Summary Dashboard", fontsize=18, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # Panel 1 — Avg price by rating
    ax1 = fig.add_subplot(gs[0, 0])
    avg = df.groupby("Rating (1-5)")["Price (£)"].mean()
    avg.plot(kind="bar", ax=ax1, color=PALETTE[:5], edgecolor="black")
    ax1.set_title("Avg Price by Rating")
    ax1.set_xlabel("Stars")
    ax1.set_ylabel("Avg Price (£)")
    ax1.tick_params(axis="x", rotation=0)

    # Panel 2 — Rating distribution pie
    ax2 = fig.add_subplot(gs[0, 1])
    counts = df["Rating (1-5)"].value_counts().sort_index()
    ax2.pie(counts, labels=[f"{i}★" for i in counts.index],
            autopct="%1.0f%%", colors=PALETTE[:5],
            wedgeprops=dict(edgecolor="white"))
    ax2.set_title("Rating Distribution")

    # Panel 3 — Price histogram
    ax3 = fig.add_subplot(gs[1, 0])
    sns.histplot(df["Price (£)"], bins=25, kde=True, color="#4C72B0",
                 edgecolor="white", ax=ax3)
    ax3.set_title("Price Distribution")
    ax3.set_xlabel("Price (£)")

    # Panel 4 — Box plot price per rating
    ax4 = fig.add_subplot(gs[1, 1])
    df.boxplot(column="Price (£)", by="Rating (1-5)", ax=ax4,
               patch_artist=True, grid=False)
    plt.sca(ax4)
    plt.title("Price Spread per Rating")
    plt.suptitle("")
    ax4.set_xlabel("Rating (Stars)")
    ax4.set_ylabel("Price (£)")

    save("viz8_dashboard.png")


# ============================================================
# TITANIC VISUALIZATIONS (fallback)
# ============================================================

def titanic_visualizations(df):
    """Full set of 8 charts for Titanic dataset."""
    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Embarked"].fillna("S", inplace=True)

    # 1. Survival count
    fig, ax = plt.subplots(figsize=(7, 5))
    df["Survived"].value_counts().rename({0: "Did not survive", 1: "Survived"}).plot(
        kind="bar", color=["#E74C3C", "#2ECC71"], edgecolor="black", ax=ax)
    ax.set_title("Survival Count", fontsize=14, fontweight="bold")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=0)
    save("viz1_survival_count.png")

    # 2. Pie — sex distribution
    fig, ax = plt.subplots(figsize=(6, 6))
    df["Sex"].value_counts().plot(kind="pie", autopct="%1.1f%%",
                                  colors=["#3498DB", "#E91E63"],
                                  wedgeprops=dict(edgecolor="white"), ax=ax)
    ax.set_ylabel("")
    ax.set_title("Passenger Sex Distribution", fontsize=14, fontweight="bold")
    save("viz2_sex_pie.png")

    # 3. Survival rate by class
    fig, ax = plt.subplots(figsize=(8, 5))
    sr = df.groupby("Pclass")["Survived"].mean() * 100
    sr.plot(kind="bar", color=PALETTE[:3], edgecolor="black", ax=ax)
    ax.set_title("Survival Rate by Passenger Class (%)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Passenger Class")
    ax.set_ylabel("Survival Rate (%)")
    ax.tick_params(axis="x", rotation=0)
    save("viz3_survival_by_class.png")

    # 4. Age distribution by survival
    fig, ax = plt.subplots(figsize=(9, 5))
    for s, label, c in [(0, "Did not survive", "#E74C3C"), (1, "Survived", "#2ECC71")]:
        sns.kdeplot(df[df["Survived"] == s]["Age"], fill=True, alpha=0.4,
                    label=label, color=c, ax=ax)
    ax.set_title("Age Distribution by Survival", fontsize=14, fontweight="bold")
    ax.legend()
    save("viz4_age_kde_survival.png")

    # 5. Heatmap — survival by sex & class
    fig, ax = plt.subplots(figsize=(7, 5))
    pivot = df.pivot_table("Survived", "Sex", "Pclass")
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", ax=ax)
    ax.set_title("Survival Rate — Sex × Class", fontsize=14, fontweight="bold")
    save("viz5_heatmap_sex_class.png")

    # 6. Violin — fare by class
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.violinplot(data=df, x="Pclass", y="Fare", palette="Set2",
                   inner="quartile", ax=ax)
    ax.set_title("Fare Distribution by Class (Violin)", fontsize=14, fontweight="bold")
    save("viz6_violin_fare_class.png")

    # 7. Bar — embarked counts
    fig, ax = plt.subplots(figsize=(7, 5))
    df["Embarked"].value_counts().plot(kind="bar", color=PALETTE[:3],
                                       edgecolor="black", ax=ax)
    ax.set_title("Passengers by Embarkation Port", fontsize=14, fontweight="bold")
    ax.set_xlabel("Port (C=Cherbourg, Q=Queenstown, S=Southampton)")
    ax.tick_params(axis="x", rotation=0)
    save("viz7_embarked_bar.png")

    # 8. Dashboard
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("🚢 Titanic — Summary Dashboard", fontsize=18, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    df["Survived"].value_counts().rename({0: "No", 1: "Yes"}).plot(
        kind="bar", color=["#E74C3C", "#2ECC71"], edgecolor="black", ax=ax1)
    ax1.set_title("Survived?"); ax1.tick_params(axis="x", rotation=0)

    ax2 = fig.add_subplot(gs[0, 1])
    sns.histplot(df["Age"], bins=30, kde=True, color="#4C72B0", ax=ax2)
    ax2.set_title("Age Distribution")

    ax3 = fig.add_subplot(gs[1, 0])
    sns.barplot(data=df, x="Pclass", y="Survived", palette="Set2", ax=ax3)
    ax3.set_title("Survival Rate by Class")
    ax3.set_ylabel("Survival Rate")

    ax4 = fig.add_subplot(gs[1, 1])
    pivot = df.pivot_table("Survived", "Sex", "Pclass")
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", ax=ax4)
    ax4.set_title("Survival: Sex × Class")

    save("viz8_dashboard.png")


# ── Main ──────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  CodeAlpha Task 3 — Data Visualization")
    print("=" * 55)
    df, dataset_type = load_data()
    print(f"\n  Generating 8 charts for '{dataset_type}' dataset...\n")

    if dataset_type == "books":
        books_bar_avg_price_by_rating(df)
        books_pie_rating_distribution(df)
        books_scatter_price_vs_rating(df)
        books_line_cumulative_books(df)
        books_violin_price_by_rating(df)
        books_availability_bar(df)
        books_price_histogram(df)
        books_dashboard(df)
    else:
        titanic_visualizations(df)

    print("\n✅ All 8 visualizations saved successfully!")
    print("   Include these in your LinkedIn post & GitHub repo.")


if __name__ == "__main__":
    main()
