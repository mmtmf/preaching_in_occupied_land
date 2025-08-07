import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
import numpy as np
from itertools import product
import ruptures as rpt
import re
from scipy.stats import norm
import os

# === Define output folder and create it ===
OUTPUT_FOLDER = r"C:\Users\au546005\OneDrive - Aarhus universitet\Documents\PhD\sentiment_project\Code\src\word_share_v1\results\results_topic"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# === Load metadata ===
metadata = pd.read_csv(
    r"C:\Users\au546005\OneDrive - Aarhus universitet\Documents\PhD\sentiment_project\Code\src\word_share_v1\data\metadata_v40.CSV",
    sep=";",
    encoding="ISO-8859-1"
)

# === Load topic distributions ===
topics = pd.read_csv(
    r"C:\Users\au546005\OneDrive - Aarhus universitet\Documents\PhD\sentiment_project\Code\src\word_share_v1\data\document_topic-distribution_csv.CSV",
    sep=";",              
    encoding="ISO-8859-1"
)

# === Identify topic columns ===
topic_columns = topics.columns.drop('id_dok')  # all except id_dok

# === Ensure numeric topic values ===
topics[topic_columns] = topics[topic_columns].apply(pd.to_numeric, errors='coerce')

# === Merge metadata with topic data ===
df = pd.merge(metadata, topics, on="id_dok")

# === Parse date column ===
df['dato'] = pd.to_datetime(df['dato'], format='%Y-%m-%d', errors='coerce')
df = df.dropna(subset=['dato'])

# === Create time periods (monthly) ===
df['month'] = df['dato'].dt.to_period('M')

# === Group by month and average topic proportions ===
monthly_avg = df.groupby('month')[topic_columns].mean()
monthly_avg.index = monthly_avg.index.to_timestamp()

# === Apply Rolling Average (2-month window) ===
rolling_avg = monthly_avg.rolling(window=2, min_periods=1).mean()

# === SELECTED TOPICS TO PLOT ===
selected_topics = ["Power and freedom", "The Danish Church and National Identity", "The Word and Peace of God in Turmoil", "Divine Will and Worldly Challenges"]

for topic in selected_topics:
    if topic not in monthly_avg.columns:
        raise ValueError(f"Topic '{topic}' not found in the topic columns.")

# === Historical Events ===
historical_events = {
    "Telegram Crisis": "1942-10-01",
    "The Norwegian Ban": "1943-02-01",
    "Collapse of Government": "1943-08-29",
    "The first Pastoral letter": "1943-10-01",
    "Murder of Kaj Munk": "1944-01-04",
}

# === 1. All Topics Without Historical Events ===
plt.figure(figsize=(14, 7))
for topic in topic_columns:
    plt.plot(rolling_avg.index, rolling_avg[topic], label=topic)

plt.title("All Topic Prevalence Over Time (Smoothed, No Events)")
plt.xlabel("Date")
plt.ylabel("Average Topic Proportion")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper right', fontsize='small', ncol=2)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "all_topics_no_events.png"), dpi=300)
plt.show()

# === 2. All Topics With Historical Events ===
plt.figure(figsize=(14, 7))
for topic in topic_columns:
    plt.plot(rolling_avg.index, rolling_avg[topic], label=topic)

for event, date_str in historical_events.items():
    date = pd.to_datetime(date_str)
    plt.axvline(x=date, color='gray', linestyle='--', linewidth=1)
    plt.text(date, plt.ylim()[1]*0.95, event, rotation=90, verticalalignment='top', fontsize=9, color='black')

plt.title("All Topic Prevalence Over Time with Historical Events (Smoothed)")
plt.xlabel("Date")
plt.ylabel("Average Topic Proportion")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper right', fontsize='small', ncol=2)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "all_topics_with_events.png"), dpi=300)
plt.show()

# === 3. Selected Topics With Historical Events ===
plt.figure(figsize=(14, 7))
for topic in selected_topics:
    plt.plot(rolling_avg.index, rolling_avg[topic], label=topic)

for event, date_str in historical_events.items():
    date = pd.to_datetime(date_str)
    plt.axvline(x=date, color='gray', linestyle='--', linewidth=1)
    plt.text(date, plt.ylim()[1]*0.95, event, rotation=90, verticalalignment='top', fontsize=9, color='black')

plt.title("Selected Topic Prevalence Over Time with Historical Events (Smoothed)")
plt.xlabel("Date")
plt.ylabel("Average Topic Proportion")
plt.grid(True, linestyle='--', alpha=0.5)
plt.ylim(0, rolling_avg[selected_topics].max().max() * 1.1)  # tighten y-axis
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "selected_topics_with_events.png"), dpi=300)
plt.show()

# === 4. Top 10 Most Prevalent Topics (Smoothed) ===
# Calculate overall mean for each topic
top10_topics = rolling_avg.mean().sort_values(ascending=False).head(10).index.tolist()

plt.figure(figsize=(14, 7))
for topic in top10_topics:
    plt.plot(rolling_avg.index, rolling_avg[topic], label=topic)

for event, date_str in historical_events.items():
    date = pd.to_datetime(date_str)
    plt.axvline(x=date, color='gray', linestyle='--', linewidth=1)
    plt.text(date, plt.ylim()[1]*0.95, event, rotation=90, verticalalignment='top', fontsize=9, color='black')


plt.title("Top 10 Most Prevalent Topics Over Time (Smoothed)")
plt.xlabel("Date")
plt.ylabel("Average Topic Proportion")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper right', fontsize='small', ncol=2)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "top10_topics_smoothed.png"), dpi=300)
plt.show()


# === 5. T-test Topic Changes (Collapse of Government) ===

# === Parameters ===
collapse_date = pd.to_datetime("1943-08-29")
liberation_date = pd.to_datetime("1945-05-04")

# === Filter data to the comparison window ===
filtered_df = df[(df['dato'] >= pd.to_datetime("1940-01-01")) & (df['dato'] <= liberation_date)].copy()
filtered_df['period'] = np.where(filtered_df['dato'] < collapse_date, 'before', 'after')

# === Prepare to collect t-test results ===
results = []

for topic in topic_columns:
    before_vals = filtered_df[filtered_df['period'] == 'before'][topic].dropna()
    after_vals = filtered_df[filtered_df['period'] == 'after'][topic].dropna()

    # Skip if too few data points
    if len(before_vals) < 10 or len(after_vals) < 10:
        continue

    stat, pval = ttest_ind(before_vals, after_vals, equal_var=False)
    mean_before = before_vals.mean()
    mean_after = after_vals.mean()
    mean_diff = mean_after - mean_before
    direction = "increase" if mean_diff > 0 else "decrease"

    results.append({
        "topic": topic,
        "mean_before": mean_before,
        "mean_after": mean_after,
        "mean_diff": mean_diff,
        "p_value": pval,
        "direction": direction
    })

# === Convert to DataFrame and apply FDR correction ===
ttest_df = pd.DataFrame(results)
ttest_df['p_adj'] = multipletests(ttest_df['p_value'], method='fdr_bh')[1]
ttest_df['significant'] = ttest_df['p_adj'] < 0.05
ttest_df = ttest_df.sort_values("p_adj")

# === Print results ===
print("\nT-test results with FDR correction (Collapse of Government):")
print(ttest_df[['topic', 'mean_before', 'mean_after', 'mean_diff', 'direction', 'p_value', 'p_adj', 'significant']])

# === Save to CSV ===
ttest_df.to_csv(os.path.join(OUTPUT_FOLDER, "t_test_topic_changes_with_fdr.csv"), index=False)

# === Visualization ===
# === Sort by absolute change and pick top N ===
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 18
top_n = 10
sorted_df = ttest_df.sort_values("mean_diff", key=abs, ascending=False).head(top_n)

# === Plot ===
plt.figure(figsize=(12, top_n * 0.6))
colors = ['darkgreen' if d > 0 else 'darkred' for d in sorted_df['mean_diff']]
bars = plt.barh(sorted_df['topic'], sorted_df['mean_diff'], color=colors)

# Highlight significant bars with bold black edges
for i, sig in enumerate(sorted_df['significant']):
    if sig:
        bars[i].set_hatch('///')
        bars[i].set_edgecolor('black')
        bars[i].set_linewidth(1.2)

plt.xlabel("Change in topic emphasis (after - before collapse)")
plt.title(f"Top {top_n} changes in sermon topics after The Collapse of Government", pad=15)
plt.axvline(0, color='gray', linewidth=1)
plt.grid(axis='x', linestyle='--', linewidth=0.5, alpha=0.3)
plt.gca().invert_yaxis()
plt.tight_layout()

plt.savefig(os.path.join(OUTPUT_FOLDER, "top_topic_changes_ttest_fdr.png"), dpi=300)
plt.show()

# === 6. T-test Topic Changes (Telegram Crisis) ===

telegram_date = pd.to_datetime("1942-10-01")
liberation_date = pd.to_datetime("1945-05-04")

# Filter data between 1940 and Liberation
filtered_df_telegram = df[(df['dato'] >= pd.to_datetime("1940-01-01")) & (df['dato'] <= liberation_date)].copy()
filtered_df_telegram['period'] = np.where(filtered_df_telegram['dato'] < telegram_date, 'before', 'after')

results_telegram = []

for topic in topic_columns:
    before_vals = filtered_df_telegram[filtered_df_telegram['period'] == 'before'][topic].dropna()
    after_vals = filtered_df_telegram[filtered_df_telegram['period'] == 'after'][topic].dropna()

    if len(before_vals) < 10 or len(after_vals) < 10:
        continue

    stat, pval = ttest_ind(before_vals, after_vals, equal_var=False)
    mean_before = before_vals.mean()
    mean_after = after_vals.mean()
    mean_diff = mean_after - mean_before
    direction = "increase" if mean_diff > 0 else "decrease"

    results_telegram.append({
        "topic": topic,
        "mean_before": mean_before,
        "mean_after": mean_after,
        "mean_diff": mean_diff,
        "p_value": pval,
        "direction": direction
    })

# Convert to DataFrame and apply FDR correction
ttest_telegram_df = pd.DataFrame(results_telegram)
ttest_telegram_df['p_adj'] = multipletests(ttest_telegram_df['p_value'], method='fdr_bh')[1]
ttest_telegram_df['significant'] = ttest_telegram_df['p_adj'] < 0.05
ttest_telegram_df = ttest_telegram_df.sort_values("p_adj")

# Output
print("\nT-test results with FDR correction (Telegram Crisis):")
print(ttest_telegram_df[['topic', 'mean_before', 'mean_after', 'mean_diff', 'direction', 'p_value', 'p_adj', 'significant']])

# Save to CSV
ttest_telegram_df.to_csv(os.path.join(OUTPUT_FOLDER, "t_test_topic_changes_telegram_fdr.csv"), index=False)

# === Visualization ===
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 18

top_n_telegram = 10
sorted_telegram_df = ttest_telegram_df.sort_values("mean_diff", key=abs, ascending=False).head(top_n_telegram)

plt.figure(figsize=(12, top_n_telegram * 0.6))

colors = ['darkgreen' if d > 0 else 'darkred' for d in sorted_telegram_df['mean_diff']]
bars = plt.barh(sorted_telegram_df['topic'], sorted_telegram_df['mean_diff'], color=colors)

# Highlight significant bars
for i, sig in enumerate(sorted_telegram_df['significant']):
    if sig:
        bars[i].set_hatch('///')
        bars[i].set_edgecolor('black')
        bars[i].set_linewidth(1.2)

plt.xlabel("Change in topic emphasis (after - before crisis)")
plt.title(f"Top {top_n_telegram} changes in sermon topics after The Telegram Crisis", pad=15)
plt.axvline(0, color='gray', linewidth=1)
plt.grid(axis='x', linestyle='--', linewidth=0.5, alpha=0.3)
plt.gca().invert_yaxis()
plt.tight_layout()

plt.savefig(os.path.join(OUTPUT_FOLDER, "top_topic_changes_telegram_ttest_fdr.png"), dpi=300)
plt.show()

# === 7) Per-Affiliation T-tests for ALL Topics (Telegram Crisis) ===

telegram_date = pd.to_datetime("1942-10-01")
liberation_date = pd.to_datetime("1945-05-04")

# === Collapse sub-affiliations into main groups ===
main_affiliations = ['fje', 'gru', 'ind', 'kce', 'kfo', 'kfu', 'tid']
df['affiliation'] = df['affiliation'].apply(
    lambda x: next((main for main in main_affiliations if isinstance(x, str) and x.startswith(main)), x)
)

# Prepare data
affil_df = df[(df['dato'] >= pd.to_datetime("1940-01-01")) & (df['dato'] <= liberation_date)].copy()
affil_df = affil_df.dropna(subset=['affiliation'])  # drop missing affiliations
affil_df['period'] = np.where(affil_df['dato'] < telegram_date, 'before', 'after')

# Count number of sermons per affiliation in each period
sermon_counts = affil_df.groupby(['affiliation', 'period'])['id_dok'].count().unstack(fill_value=0)

# Display the table
print("\n📊 Number of sermons per affiliation (before/after Telegram Crisis):")
print(sermon_counts)

# Filter: only use affiliations with at least 10 sermons in both periods
eligible_affils = sermon_counts[(sermon_counts['before'] >= 10) & (sermon_counts['after'] >= 10)].index
print(f"\n✅ Eligible affiliations (≥10 sermons before AND after):\n{eligible_affils.tolist()}")

# Results list
affil_results = []

# Loop through each eligible affiliation and topic
for affil, group in affil_df.groupby('affiliation'):
    if affil not in eligible_affils:
        continue
    group = group.copy()
    for topic in topic_columns:
        if topic not in group.columns or group[topic].isna().all():
            continue

        before = group[group['period'] == 'before'][topic].dropna()
        after = group[group['period'] == 'after'][topic].dropna()

        if len(before) < 10 or len(after) < 10:
            continue  # just in case

        stat, pval = ttest_ind(before, after, equal_var=False)
        mean_before = before.mean()
        mean_after = after.mean()
        mean_diff = mean_after - mean_before
        direction = "increase" if mean_diff > 0 else "decrease"

        affil_results.append({
            "affiliation": affil,
            "topic": topic,
            "mean_before": mean_before,
            "mean_after": mean_after,
            "mean_diff": mean_diff,
            "p_value": pval,
            "direction": direction
        })

# Create DataFrame
affil_topic_df = pd.DataFrame(affil_results)

# Apply FDR correction
affil_topic_df['p_adj'] = multipletests(affil_topic_df['p_value'], method='fdr_bh')[1]
affil_topic_df['significant'] = affil_topic_df['p_adj'] < 0.05
affil_topic_df = affil_topic_df.sort_values("p_adj")

# Save to CSV
affil_topic_df.to_csv(os.path.join(OUTPUT_FOLDER, "affiliationwise_topic_changes_telegram_fdr.csv"), index=False)

# Print results
print("\nMost significant topic shifts per theological affiliation (Telegram Crisis):")
print(affil_topic_df[affil_topic_df['significant']].head(10))

# Optional: summary table
affil_summary = affil_topic_df[affil_topic_df['significant']].groupby('topic')['affiliation'].nunique().reset_index()
affil_summary.columns = ['topic', 'num_affiliations_changed']
affil_summary = affil_summary.sort_values('num_affiliations_changed', ascending=False)

print("\nTopics with most affiliations showing significant change:")
print(affil_summary)

# Save summary
affil_summary.to_csv(os.path.join(OUTPUT_FOLDER, "topics_changed_most_affiliations.csv"), index=False)

# === 8) Per-Affiliation T-tests for ALL Topics (Collapse of Government) ===

collapse_date = pd.to_datetime("1943-08-29")
liberation_date = pd.to_datetime("1945-05-04")

# === Collapse sub-affiliations into main groups ===
main_affiliations = ['fje', 'gru', 'ind', 'kce', 'kfo', 'kfu', 'tid']
df['affiliation'] = df['affiliation'].apply(
    lambda x: next((main for main in main_affiliations if isinstance(x, str) and x.startswith(main)), x)
)

# Prepare data
affil_df = df[(df['dato'] >= pd.to_datetime("1940-01-01")) & (df['dato'] <= liberation_date)].copy()
affil_df = affil_df.dropna(subset=['affiliation'])  # drop missing affiliations
affil_df['period'] = np.where(affil_df['dato'] < collapse_date, 'before', 'after')

# Count number of sermons per affiliation in each period
sermon_counts = affil_df.groupby(['affiliation', 'period'])['id_dok'].count().unstack(fill_value=0)

# Display the table
print("\n📊 Number of sermons per affiliation (before/after Collapse of Government):")
print(sermon_counts)

# Filter: only use affiliations with at least 10 sermons in both periods
eligible_affils = sermon_counts[(sermon_counts['before'] >= 10) & (sermon_counts['after'] >= 10)].index
print(f"\n✅ Eligible affiliations (≥10 sermons before AND after):\n{eligible_affils.tolist()}")

# Results list
affil_results = []

# Loop through each eligible affiliation and topic
for affil, group in affil_df.groupby('affiliation'):
    if affil not in eligible_affils:
        continue
    group = group.copy()
    for topic in topic_columns:
        if topic not in group.columns or group[topic].isna().all():
            continue

        before = group[group['period'] == 'before'][topic].dropna()
        after = group[group['period'] == 'after'][topic].dropna()

        if len(before) < 10 or len(after) < 10:
            continue

        stat, pval = ttest_ind(before, after, equal_var=False)
        mean_before = before.mean()
        mean_after = after.mean()
        mean_diff = mean_after - mean_before
        direction = "increase" if mean_diff > 0 else "decrease"

        affil_results.append({
            "affiliation": affil,
            "topic": topic,
            "mean_before": mean_before,
            "mean_after": mean_after,
            "mean_diff": mean_diff,
            "p_value": pval,
            "direction": direction
        })

# Create DataFrame
affil_topic_df = pd.DataFrame(affil_results)

# Apply FDR correction
affil_topic_df['p_adj'] = multipletests(affil_topic_df['p_value'], method='fdr_bh')[1]
affil_topic_df['significant'] = affil_topic_df['p_adj'] < 0.05
affil_topic_df = affil_topic_df.sort_values("p_adj")

# Save to CSV
affil_topic_df.to_csv(os.path.join(OUTPUT_FOLDER, "affiliationwise_topic_changes_collapse_fdr.csv"), index=False)

# Print results
print("\nMost significant topic shifts per theological affiliation (Collapse of Government):")
print(affil_topic_df[affil_topic_df['significant']].head(10))

# Optional: summary table
affil_summary = affil_topic_df[affil_topic_df['significant']].groupby('topic')['affiliation'].nunique().reset_index()
affil_summary.columns = ['topic', 'num_affiliations_changed']
affil_summary = affil_summary.sort_values('num_affiliations_changed', ascending=False)

print("\nTopics with most affiliations showing significant change:")
print(affil_summary)

# Save summary
affil_summary.to_csv(os.path.join(OUTPUT_FOLDER, "topics_changed_most_affiliations_collapse.csv"), index=False)

# === 9) Visualization of T-tests (#7 and #8) ===
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 14

# === Load Results ===
telegram_df = pd.read_csv(os.path.join(OUTPUT_FOLDER, "affiliationwise_topic_changes_telegram_fdr.csv"))
collapse_df = pd.read_csv(os.path.join(OUTPUT_FOLDER, "affiliationwise_topic_changes_collapse_fdr.csv"))

# === Sort by absolute mean_diff and take top N ===
top_n = 10
telegram_top = telegram_df.sort_values("mean_diff", key=abs, ascending=False).head(top_n)
collapse_top = collapse_df.sort_values("mean_diff", key=abs, ascending=False).head(top_n)

# === Helper: Plot function ===
def plot_top_changes(df, title, filename):
    plt.figure(figsize=(10, max(5, len(df) * 0.6)))
    
    labels = df['affiliation'] + " – " + df['topic']
    values = df['mean_diff']
    colors = ['darkgreen' if x > 0 else 'darkred' for x in values]
    
    bars = plt.barh(labels, values, color=colors)
    
    # Apply hatching for significance
    for i, bar in enumerate(bars):
        if df.iloc[i]['significant']:
            bar.set_hatch('///')
            bar.set_edgecolor('black')
            bar.set_linewidth(1.2)
    
    plt.axvline(0, color='gray', linewidth=1)
    plt.xlabel("Change in topic emphasis (after - before)")
    plt.title(title, pad=15)
    plt.grid(axis='x', linestyle='--', linewidth=0.5, alpha=0.3)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, filename), dpi=300)
    plt.show()

# === Plot for Telegram Crisis ===
plot_top_changes(
    telegram_top,
    "Top shifts in sermon topics by affiliation after the Telegram Crisis",
    "top_topic_shifts_telegram.png"
)

# === Plot for Collapse of Government ===
plot_top_changes(
    collapse_top,
    "Top changes in sermon topics by affiliation\n after The Collapse of Government",
    "top_topic_shifts_collapse.png"
)

# === 10) Change Point Detection for ALL Topics (using RBF model) ===
print("\n🔍 Running change point detection for all topics using 'rbf' model...")

# Parameters
model = "rbf"         # radial basis function: better for nonlinear, trend, variance changes
penalty = 5          # higher penalty for RBF to prevent overfitting
min_size = 3          # min number of points between breakpoints

# Prepare list of topics and results
topics_to_check = topic_columns
change_point_results = []

for topic in topics_to_check:
    series = monthly_avg[topic].dropna().values
    dates = monthly_avg[topic].dropna().index

    if len(series) < min_size * 2:
        continue  # skip short series

    try:
        algo = rpt.Pelt(model=model, min_size=min_size).fit(series)
        change_points = algo.predict(pen=penalty)
        detected_dates = [dates[i - 1] for i in change_points[:-1]]  # exclude final point

        print(f"\n📈 Topic: {topic}")
        print("Change points:", [d.strftime('%Y-%m') for d in detected_dates])

        # Plot
        fig, axarr = rpt.display(series, change_points, figsize=(12, 4))
        fig.suptitle(f"Change Points in '{topic}' (RBF Model)")
        plt.xticks(
            ticks=np.arange(len(dates)),
            labels=[d.strftime('%Y-%m') for d in dates],
            rotation=45,
            fontsize=8
        )
        plt.tight_layout()
        safe_topic = re.sub(r'[^\w\-_.]', '_', topic)
        plt.savefig(os.path.join(OUTPUT_FOLDER, f"changepoints_{safe_topic}.png"), dpi=300)
        plt.close()

        # Store results
        change_point_results.append({
            "topic": topic,
            "change_points": "; ".join([d.strftime('%Y-%m') for d in detected_dates])
        })

    except Exception as e:
        print(f"⚠️ Could not process topic '{topic}': {e}")

# Save summary
cp_df = pd.DataFrame(change_point_results)
cp_df.to_csv(os.path.join(OUTPUT_FOLDER, "change_points_all_topics.csv"), index=False)

print("\n✅ Change point detection (RBF) completed for all topics. Plots and summary CSV saved.")


# === 11) Change Point Detection on First Derivative of ALL Topics ===
print("\n🔍 Running change point detection on first derivatives (rate of change)...")

# Parameters
model = "l2"
penalty = 0.5
min_size = 3

topics_to_check = topic_columns
change_point_results_diff = []

for topic in topics_to_check:
    # Use first derivative (rate of change)
    series = monthly_avg[topic].dropna().diff().dropna().values
    dates = monthly_avg[topic].dropna().index[1:]  # because diff() shifts by 1

    if len(series) < min_size * 2:
        continue

    try:
        algo = rpt.Pelt(model=model, min_size=min_size).fit(series)
        change_points = algo.predict(pen=penalty)
        detected_dates = [dates[i - 1] for i in change_points[:-1]]

        print(f"\n📈 Topic (1st derivative): {topic}")
        print("Change points:", [d.strftime('%Y-%m') for d in detected_dates])

        # Plot
        fig, axarr = rpt.display(series, change_points, figsize=(12, 4))
        fig.suptitle(f"Change Points in Rate of Change for '{topic}'")
        plt.xticks(
            ticks=np.arange(len(dates)),
            labels=[d.strftime('%Y-%m') for d in dates],
            rotation=45,
            fontsize=8
        )
        plt.tight_layout()
        safe_topic = re.sub(r'[^\w\-_.]', '_', topic)
        plt.savefig(os.path.join(OUTPUT_FOLDER, f"changepoints_diff_{safe_topic}.png"), dpi=300)
        plt.close()

        # Store results
        change_point_results_diff.append({
            "topic": topic,
            "change_points": "; ".join([d.strftime('%Y-%m') for d in detected_dates])
        })

    except Exception as e:
        print(f"⚠️ Could not process topic '{topic}': {e}")

# Save results
cp_diff_df = pd.DataFrame(change_point_results_diff)
cp_diff_df.to_csv(os.path.join(OUTPUT_FOLDER, "change_points_all_topics_diff.csv"), index=False)

print("\n✅ Change point detection (1st derivative) completed. Plots and summary CSV saved.")

