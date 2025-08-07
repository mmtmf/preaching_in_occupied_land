import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sentida import Sentida
import re
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.stats import ttest_1samp
import seaborn as sns
from scipy.stats import sem
import ruptures as rpt
from scipy.stats import ttest_ind

# === Define output folder and create it ===
OUTPUT_FOLDER = r"C:\Users\au546005\OneDrive - Aarhus universitet\Documents\PhD\sentiment_project\Code\src\word_share_v1\results\results_sentiment"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Define file paths
SERMONS_FOLDER = r"C:\Users\au546005\OneDrive - Aarhus universitet\Documents\PhD\sentiment_project\Code\src\word_share_v1\data\præ_fin"
sentiment_file = os.path.join(SERMONS_FOLDER, "sermons_sentiment_results.csv")
metadata_file = r"C:\Users\au546005\OneDrive - Aarhus universitet\Documents\PhD\sentiment_project\Code\src\word_share_v1\data\metadata_v40.CSV"


# Initialize SENTIDA model
sentida_model = Sentida()

# Function to extract first 10 sentences
def get_first_10_sentences(text):
    sentences = re.split(r'\.\s+', text)
    cleaned = [s.strip() for s in sentences if s.strip()]
    return ". ".join(cleaned[:10]).strip() + "."

# Function to extract last 10 sentences
def get_last_10_sentences(text):
    sentences = re.split(r'\.\s+', text)
    cleaned = [s.strip() for s in sentences if s.strip()]
    return ". ".join(cleaned[-10:]).strip() + "."

def analyze_sermons(folder_path):
    sermon_data = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read().strip()

            first_10 = get_first_10_sentences(text)
            last_10 = get_last_10_sentences(text)

            score_first = sentida_model.sentida(first_10, normal=True)
            score_last = sentida_model.sentida(last_10, normal=True)

            sermon_data.append({
                "id_dok": filename.replace(".txt", ""),
                "sentiment_first10": score_first,
                "sentiment_last10": score_last
            })

    return pd.DataFrame(sermon_data)

# Step 1: Perform Sentiment Analysis
sermon_df = analyze_sermons(SERMONS_FOLDER)
sermon_df.to_csv(sentiment_file, index=False)

# Step 2: Merge with metadata
metadata_df = pd.read_csv(metadata_file, sep=";", encoding="ISO-8859-1", engine="python")
metadata_df['id_dok'] = metadata_df['id_dok'].astype(str).str.strip()
sermon_df['id_dok'] = sermon_df['id_dok'].astype(str).str.strip()

# Merge sentiment with metadata
merged_df = pd.merge(sermon_df, metadata_df, on="id_dok", how="inner")

# Parse date and extract month
merged_df["dato"] = pd.to_datetime(merged_df["dato"], errors="coerce")
merged_df = merged_df.dropna(subset=["dato"])
merged_df["month"] = merged_df["dato"].dt.to_period("M").dt.to_timestamp()

# Group by month
monthly = merged_df.groupby("month")[["sentiment_first10", "sentiment_last10"]].mean()
monthly_smoothed = monthly.rolling(window=2, min_periods=1).mean()

# === 1) First 10 and last 10 sentences
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 18

# Plot only smoothed trend
plt.figure(figsize=(12, 6))
first_color = '#1b7837'  # muted green
last_color = '#2166ac'  # muted blue

plt.plot(monthly_smoothed.index, monthly_smoothed["sentiment_first10"],
         label="First 10 sentences", color=first_color, linewidth=2)
plt.plot(monthly_smoothed.index, monthly_smoothed["sentiment_last10"],
         label="Last 10 sentences", color=last_color, linewidth=2)

plt.xlabel("Date")
plt.ylabel("Average sentiment score")
plt.title("Smoothed sentiment in first and last 10 sentences of sermons")
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)
plt.legend()
plt.tight_layout()

plot_file = os.path.join(OUTPUT_FOLDER, "sentiment_first_last_10.png")
plt.savefig(plot_file, dpi=300)
plt.show()

print(f"✅ Plot with both sentiment lines saved: {plot_file}")

# === 2) Trend lines (10 first and last 10 sentences) ===
# Prepare data
dates_numeric = (monthly_smoothed.index - monthly_smoothed.index[0]).days

# LOWESS smoothing
lowess_first = lowess(monthly_smoothed["sentiment_first10"], dates_numeric, frac=0.2)
lowess_last = lowess(monthly_smoothed["sentiment_last10"], dates_numeric, frac=0.2)

# Plot with LOWESS trends
plt.figure(figsize=(12, 6))
plt.plot(monthly_smoothed.index, monthly_smoothed["sentiment_first10"],
         color=first_color, label="First 10 sentences", linewidth=1.5)
plt.plot(monthly_smoothed.index, monthly_smoothed["sentiment_last10"],
         color=last_color, label="Last 10 sentences", linewidth=1.5)

# LOWESS lines
plt.plot(monthly_smoothed.index, lowess_first[:, 1],
         color=first_color, linestyle="--", linewidth=2.5, label="LOWESS trendline")
plt.plot(monthly_smoothed.index, lowess_last[:, 1],
         color=last_color, linestyle="--", linewidth=2.5, label="LOWESS trendline")

plt.xlabel("Date")
plt.ylabel("Average sentiment score")
plt.title("Sermon sentiment")
plt.legend()
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()

# Save the plot
trend_plot_file = os.path.join(OUTPUT_FOLDER, "sentiment_lowess_trend.png")
plt.savefig(trend_plot_file, dpi=300)
plt.show()

print(f"📈 LOWESS trend plot saved: {trend_plot_file}")

# === 3) Sentiment per Affiliation ===
# Font setup
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 22

# Define relevant main affiliation prefixes
main_affiliations = ["ind", "gru", "tid",]

# Map abbreviations to full names
affiliation_names = {
    "ind": "Indre Mission",
    "gru": "Grundtvigske",
    "tid": "Tidehverv"
}

# Collapse detailed affiliations into main groups
def simplify_affiliation(aff):
    if pd.isna(aff):
        return None
    for prefix in main_affiliations:
        if aff.startswith(prefix):
            return prefix
    return None

# Apply simplification
merged_df["affiliation_std"] = merged_df["affiliation"].apply(simplify_affiliation)

# Drop rows with no valid affiliation
filtered_df = merged_df.dropna(subset=["affiliation_std"])

# Group by month and standardized affiliation
grouped = filtered_df.groupby(["month", "affiliation_std"])[["sentiment_first10", "sentiment_last10"]].mean().reset_index()

# Get unique affiliations
affiliations = sorted(grouped["affiliation_std"].unique())

# Plotting setup
color_palette = {
    "ind": "#1b7837",  # muted green
    "gru": "#2166ac",  # muted blue
    "tid": "#b2182b",  # muted red
}
plt.figure(figsize=(16, 9))

# Plot for each affiliation
for aff in affiliations:
    subset = grouped[grouped["affiliation_std"] == aff]
    dates = pd.to_datetime(subset["month"])
    dates_numeric = (dates - dates.min()).dt.days

    if len(subset) >= 3:
        lowess_first = lowess(subset["sentiment_first10"], dates_numeric, frac=0.2)
        lowess_last = lowess(subset["sentiment_last10"], dates_numeric, frac=0.2)

        full_label = affiliation_names[aff]
        plt.plot(dates, lowess_first[:, 1], linestyle='--', linewidth=2,
                 label=f"{full_label} – first 10", color=color_palette[aff])
        plt.plot(dates, lowess_last[:, 1], linestyle='-', linewidth=2,
                 label=f"{full_label} – last 10", color=color_palette[aff])


# Final plot formatting
plt.xlabel("Date")
plt.ylabel("Average sentiment score")
plt.title("Sermon sentiment by affiliation")
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()


# Save
plot_file = os.path.join(OUTPUT_FOLDER, "sentiment_affiliation_lowess.png")
plt.savefig(plot_file, dpi=300)
plt.show()

print(f"📊 Sentiment-by-affiliation LOWESS plot saved: {plot_file}")

# === 4) Bar Plot of Sentiment Delta by Affiliation (Matplotlib) ===
# Font settings
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 14

# Compute delta
merged_df["sentiment_delta"] = merged_df["sentiment_last10"] - merged_df["sentiment_first10"]

# Group and compute stats
delta_stats = merged_df.groupby("affiliation_std")["sentiment_delta"].agg(["mean", "count", "std"]).reset_index()
delta_stats["sem"] = delta_stats["std"] / delta_stats["count"]**0.5
delta_stats.sort_values("mean", ascending=False, inplace=True)

# === Map abbreviations to full names ===
affiliation_names = {
    "ind": "Indre Mission",
    "gru": "Grundtvigske",
    "tid": "Tidehverv"
}
delta_stats["affiliation_full"] = delta_stats["affiliation_std"].map(affiliation_names)

# Plot
plt.figure(figsize=(9, 5))
x = np.arange(len(delta_stats))
means = delta_stats["mean"].values
errors = delta_stats["sem"].values
labels = delta_stats["affiliation_full"].values
bar_color = '#2166ac'
error_color = 'gray' 
bars = plt.bar(x, means, color=bar_color, edgecolor='black', linewidth=0.8)
plt.errorbar(x, means, yerr=errors, fmt='none', ecolor=error_color, elinewidth=2, capsize=7)
plt.xticks(x, labels, rotation=20, ha='right')
plt.ylabel("Mean sentiment change\n(last 10 – first 10 sentences)")
plt.xlabel("Affiliation")
plt.title("Average sentiment change by affiliation with standard error bars")
plt.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.3)
plt.tight_layout()

# Save
delta_plot_file = os.path.join(OUTPUT_FOLDER, "sentiment_delta_by_affiliation.png")
plt.savefig(delta_plot_file, dpi=300)
plt.show()

print(f"📊 Sentiment delta bar plot saved: {delta_plot_file}")

# === 5) Change Point Detection using Ruptures (RBF model) ===

def plot_combined_changepoints(series1, series2, label1, label2, model="rbf", pen1=5, pen2=5,
                                title="Combined Change Point Detection",
                                save_path=None, figsize=(14, 6)):
    """
    Plots two time series with their respective change points in the same figure.

    Parameters:
    - series1, series2: Sentiment series (same index)
    - label1, label2: Labels for each line (e.g., 'First 10', 'Last 10')
    - model: 'rbf' or 'l2'
    - pen1, pen2: penalty values for each series
    - title: Title for plot
    - save_path: path to save
    - figsize: Figure size
    """
    import matplotlib.pyplot as plt
    import ruptures as rpt

    # Drop NaNs for alignment
    series1 = series1.dropna()
    series2 = series2.dropna()
    idx = series1.index.intersection(series2.index)

    signal1 = series1[idx].values
    signal2 = series2[idx].values

    # Run change point detection
    result1 = rpt.Pelt(model=model).fit(signal1).predict(pen=pen1)
    result2 = rpt.Pelt(model=model).fit(signal2).predict(pen=pen2)

    # Plot both series
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(idx, signal1, label=label1, color="green", linestyle="--")
    ax.plot(idx, signal2, label=label2, color="blue", linestyle="-")

    # Add vertical lines for changepoints
    for cp in result1[:-1]:
        ax.axvline(x=idx[cp], color="green", linestyle=":", alpha=0.7)
    for cp in result2[:-1]:
        ax.axvline(x=idx[cp], color="blue", linestyle=":", alpha=0.7)

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Average Sentiment Score")
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)
        print(f"📸 Combined plot saved: {save_path}")

    plt.show()

    return result1, result2

# Run change point detection
combined_plot_file = os.path.join(OUTPUT_FOLDER, "changepoint_combined_rbf.png")

result_first10, result_last10 = plot_combined_changepoints(
    series1=monthly["sentiment_first10"],
    series2=monthly["sentiment_last10"],
    label1="First 10 Sentences",
    label2="Last 10 Sentences",
    model="rbf",
    pen1=5,
    pen2=5,
    title="Change Point Detection (rbf) – First vs Last 10 Sentences",
    save_path=combined_plot_file
)

print("\n📆 Detected change points (First 10):")
for i in result_first10[:-1]:  # skip final boundary
    print(f"{monthly_smoothed.index[i-1].date()} ➡ {monthly_smoothed.index[i].date()}")

print("\n📆 Detected change points (Last 10):")
for i in result_last10[:-1]:
    print(f"{monthly_smoothed.index[i-1].date()} ➡ {monthly_smoothed.index[i].date()}")

# === 6) Change Point Detection in relation to specific events ===
# Event windows to test
event_tests = {
    "Telegram Crisis": "1942-10-01",
    "Norwegian Ban": "1943-02-01",
    "Collapse of Government": "1943-08-29",
    "Pastoral Letter": "1943-10-01",
    "Kaj Munk Murder": "1944-01-04"
}

# Convert monthly sentiment to long format
long_df = merged_df.copy()
long_df["month"] = pd.to_datetime(long_df["month"])

# Function to get 3-month before/after window
def get_window(date_str, months=3):
    event_date = pd.to_datetime(date_str)
    before_start = event_date - pd.DateOffset(months=months)
    before_end = event_date - pd.DateOffset(days=1)
    after_start = event_date
    after_end = event_date + pd.DateOffset(months=months-1)
    return (before_start, before_end), (after_start, after_end)

# Loop through events
for event_name, event_date in event_tests.items():
    (before_start, before_end), (after_start, after_end) = get_window(event_date)

    before = long_df[(long_df["month"] >= before_start) & (long_df["month"] <= before_end)]
    after = long_df[(long_df["month"] >= after_start) & (long_df["month"] <= after_end)]

    # Last 10 sentence sentiment comparison
    t_stat, p_val = ttest_ind(after["sentiment_last10"], before["sentiment_last10"], equal_var=False)
    print(f"🧪 {event_name} (Last 10 Sentences): t={t_stat:.3f}, p={p_val:.3f}")

    # First 10 sentence sentiment comparison
    t_stat, p_val = ttest_ind(after["sentiment_first10"], before["sentiment_first10"], equal_var=False)
    print(f"🧪 {event_name} (First 10 Sentences): t={t_stat:.3f}, p={p_val:.3f}\n")

#Visualisation
def plot_pre_post_sentiment(df, event_name, event_date, months=3, save_folder=OUTPUT_FOLDER):
    """
    Creates boxplot + stripplot of sentiment before/after a specific event.
    """
    event_date = pd.to_datetime(event_date)
    before_start = event_date - pd.DateOffset(months=months)
    before_end = event_date - pd.DateOffset(days=1)
    after_start = event_date
    after_end = event_date + pd.DateOffset(months=months-1)

    # Filter data
    before_df = df[(df["month"] >= before_start) & (df["month"] <= before_end)].copy()
    after_df = df[(df["month"] >= after_start) & (df["month"] <= after_end)].copy()

    before_df["period"] = "Before"
    after_df["period"] = "After"

    combined = pd.concat([before_df, after_df])
    combined["event"] = event_name

    # Plot both first10 and last10
    for col in ["sentiment_first10", "sentiment_last10"]:
        plt.figure(figsize=(7, 5))
        sns.boxplot(data=combined, x="period", y=col, palette="pastel")
        sns.stripplot(data=combined, x="period", y=col, color="black", alpha=0.6, jitter=0.2)
        plt.title(f"{col.replace('_', ' ').title()} – {event_name}")
        plt.ylabel("Sentiment Score")
        plt.xlabel("")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        filename = f"{col}_{event_name.lower().replace(' ', '_')}_boxplot.png"
        path = os.path.join(save_folder, filename)
        plt.savefig(path, dpi=300)
        plt.close()
        print(f"📸 Saved: {path}")

# Run for all events
for event_name, event_date in event_tests.items():
    plot_pre_post_sentiment(merged_df, event_name, event_date)

# === 6) Sentiment changes prior and post august 1943 ===
# Define cutoff date
cutoff = pd.to_datetime("1943-08-29")

# Label periods
merged_df["occupation_period"] = merged_df["month"].apply(
    lambda x: "Pre-August 1943" if x < cutoff else "Post-August 1943"
)

# Convert to long format for plotting both sentiment columns
plot_df = merged_df.melt(
    id_vars=["occupation_period"],
    value_vars=["sentiment_first10", "sentiment_last10"],
    var_name="Position",
    value_name="Sentiment"
)

# Clean up labels
plot_df["Position"] = plot_df["Position"].str.replace("sentiment_", "").str.replace("10", " 10").str.title()

# Plot mean + 95% CI instead of boxplot
plt.figure(figsize=(8, 5))
sns.pointplot(data=plot_df, x="occupation_period", y="Sentiment", hue="Position",
              dodge=0.3, join=True, capsize=0.1, errwidth=1.5, palette="Set2")

plt.title("Mean Sentiment Before and After Government Collapse (Aug 29, 1943)")
plt.ylabel("Average Sentiment Score")
plt.xlabel("")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.legend(title="Sentence Position")
plt.tight_layout()

# Save figure
prepost_pointplot_file = os.path.join(OUTPUT_FOLDER, "sentiment_prepost_aug1943_pointplot.png")
plt.savefig(prepost_pointplot_file, dpi=300)
plt.show()

print(f"📊 Point plot with CI saved: {prepost_pointplot_file}")

# === 7) Compute and interpret Cohen’s d for Pre/Post August 1943 ===

# Function to compute Cohen's d
def cohen_d(x1, x2):
    mean1, mean2 = np.mean(x1), np.mean(x2)
    std1, std2 = np.std(x1, ddof=1), np.std(x2, ddof=1)
    n1, n2 = len(x1), len(x2)
    pooled_std = np.sqrt(((n1 - 1)*std1**2 + (n2 - 1)*std2**2) / (n1 + n2 - 2))
    d = (mean2 - mean1) / pooled_std
    return d

# Optional: interpretation helper
def interpret_d(d):
    if abs(d) < 0.2:
        return "negligible"
    elif abs(d) < 0.5:
        return "small"
    elif abs(d) < 0.8:
        return "medium"
    else:
        return "large"

# Subsets
pre_df = merged_df[merged_df["month"] < cutoff]
post_df = merged_df[merged_df["month"] >= cutoff]

# Compute Cohen's d for first and last 10
d_first = cohen_d(pre_df["sentiment_first10"], post_df["sentiment_first10"])
d_last = cohen_d(pre_df["sentiment_last10"], post_df["sentiment_last10"])

# Output
print(f"\n🧮 Cohen's d (First 10): {d_first:.3f} → {interpret_d(d_first)} effect")
print(f"🧮 Cohen's d (Last 10): {d_last:.3f} → {interpret_d(d_last)} effect")

# === 8) Faceted Pre/Post Point Plot for Each Event ===
# Font settings
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 26

# === Full display names for events ===
event_name_map = {
    "Telegram Crisis": "The Telegram Crisis",
    "Norwegian Ban": "The Norwegian Ban",
    "Collapse of Government": "The Collapse of Government",
    "Pastoral Letter": "The first Pastoral letter",
    "Kaj Munk Murder": "The murder on Kaj Munk"
}

# === Function to compute Cohen's d ===
def cohen_d(x1, x2):
    mean1, mean2 = np.mean(x1), np.mean(x2)
    std1, std2 = np.std(x1, ddof=1), np.std(x2, ddof=1)
    n1, n2 = len(x1), len(x2)
    pooled_std = np.sqrt(((n1 - 1)*std1**2 + (n2 - 1)*std2**2) / (n1 + n2 - 2))
    return (mean2 - mean1) / pooled_std

# === Gather data for plotting and annotation ===
event_plot_data = []
annotations = {}

for event_key, event_date in event_tests.items():
    event_date = pd.to_datetime(event_date)
    before = merged_df[(merged_df["month"] >= event_date - pd.DateOffset(months=3)) &
                       (merged_df["month"] < event_date)].copy()
    after = merged_df[(merged_df["month"] >= event_date) &
                      (merged_df["month"] <= event_date + pd.DateOffset(months=2))].copy()

    before["period"] = "before"
    after["period"] = "after"
    combined = pd.concat([before, after])
    combined["event"] = event_name_map[event_key]
    event_plot_data.append(combined)

    # Stats
    t_stat, p_val = ttest_ind(after["sentiment_last10"], before["sentiment_last10"], equal_var=False)
    d = cohen_d(before["sentiment_last10"], after["sentiment_last10"])
    annotations[event_name_map[event_key]] = f"d = {d:.2f}\np = {p_val:.3f}"

# === Combine and clean dataset ===
event_df_all = pd.concat(event_plot_data, ignore_index=True)
event_df_all = event_df_all.loc[:, ~event_df_all.columns.duplicated()]
event_df_all["period"] = pd.Categorical(event_df_all["period"], categories=["before", "after"], ordered=True)

# === Plot ===
custom_palette = {"before": "#1b7837", "after": "#2166ac"}

g = sns.catplot(
    data=event_df_all,
    x="period", y="sentiment_last10",
    hue="period",
    col="event", col_wrap=3,
    kind="point", dodge=True,
    capsize=0.1,
    err_kws={'linewidth': 1.5},
    palette=custom_palette,
    height=4, aspect=1,
    sharex=False,
    legend=False
)

# === Annotate each subplot ===
for ax, title in zip(g.axes.flat, g.col_names):
    label = annotations.get(title)
    if label:
        if title == "The Collapse of Government":
            ax.text(0.5, 0.95, label,
                    transform=ax.transAxes,
                    ha="center", va="top",
                    fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.4",
                              facecolor="#f0f0f0",  # light gray background
                              edgecolor="black",
                              linewidth=1.2))
        else:
            ax.text(0.5, 0.95, label,
                    transform=ax.transAxes,
                    ha="center", va="top",
                    fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor="white",
                              edgecolor="gray",
                              linewidth=0.8))

        
# === Format axes and spacing ===
for ax in g.axes.flat:
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["before", "after"])
    ax.set_xlabel("")
    ax.set_ylabel("")

g.set_titles("{col_name}")
g.fig.subplots_adjust(top=0.88, wspace=0.3, hspace=0.4)
g.fig.suptitle(
    "Mean sentiment (the last 10 sentences) before and after key events\n"
    "- Cohen’s d and p-values included", fontsize=26
)

# === Save ===
annotated_plot_file = os.path.join(OUTPUT_FOLDER, "sentiment_event_prepost_annotated.png")
g.fig.savefig(annotated_plot_file, dpi=300)
plt.show()

print(f"📊 Annotated event sentiment plot saved: {annotated_plot_file}")