"""
eda.py — Exploratory Data Analysis for stock price prediction project
======================================================================
Run this script first to understand the dataset before building models.

Sections
--------
1. Basic dataset info (shape, types, missing values, statistics)
2. Price history with volume
3. Technical indicators visualisation
4. Daily returns distribution
5. Feature correlation heatmap
6. Target (BUY / SELL) class balance
7. Feature distributions split by class
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from fetch_data import fetch_data
from indicators import add_all_ind
from config import TICKER, PERIOD, ML_FEATURES

sns.set_theme(style="darkgrid", palette="muted")

# ── Load data ───────────────────────────────────────────────────────
print("=" * 60)
print(f"  EDA for {TICKER}  (period = {PERIOD})")
print("=" * 60)

data = fetch_data(TICKER, period=PERIOD)
if data is None:
    raise SystemExit("Data fetch failed.")

data = add_all_ind(data)

# Binary target: 1 = price goes UP next day, 0 = stays flat / goes DOWN
data["Target"] = (data['Close'].shift(-1) > data['Close']).astype(int)
data_clean = data.dropna().copy()

# ── Section 1: Basic info ───────────────────────────────────────────
print(f"\n{'─'*60}")
print("  1. DATASET OVERVIEW")
print(f"{'─'*60}")
print(f"  Shape            : {data_clean.shape}")
print(f"  Date range       : {data_clean.index[0].date()}  →  {data_clean.index[-1].date()}")
print(f"  Trading days     : {len(data_clean)}")
print(f"\n  Column dtypes:\n{data_clean.dtypes}\n")
print(f"  Missing values after dropna: {data_clean.isnull().sum().sum()}")
print(f"\n  Descriptive statistics (ML features):\n")
print(data_clean[ML_FEATURES].describe().round(4).to_string())

# ── Section 2: Price history + volume ──────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
fig.suptitle(f"{TICKER} — Price History & Volume ({PERIOD})", fontsize=14, fontweight='bold')

# Close with SMA/EMA overlays
axes[0].plot(data_clean.index, data_clean['Close'],
             label='Close', linewidth=0.9, color='steelblue')
axes[0].plot(data_clean.index, data_clean['SMA_20'],
             label='SMA 20', linewidth=1.2, color='orange', alpha=0.85)
axes[0].plot(data_clean.index, data_clean['EMA_50'],
             label='EMA 50', linewidth=1.2, color='red', alpha=0.85)
axes[0].fill_between(data_clean.index,
                      data_clean['Upper_band_20'],
                      data_clean['Lower_band_20'],
                      alpha=0.08, color='grey', label='Bollinger Bands')
axes[0].set_ylabel("Price ($)")
axes[0].legend(loc='upper left', fontsize=8)

# Daily returns
axes[1].bar(data_clean.index, data_clean['Daily_Return'],
            color=np.where(data_clean['Daily_Return'] >= 0, 'forestgreen', 'tomato'),
            width=1, label='Daily Return (%)')
axes[1].axhline(0, color='black', linewidth=0.6)
axes[1].set_ylabel("Daily Return (%)")
axes[1].legend(loc='upper left', fontsize=8)

# Volume
axes[2].bar(data_clean.index, data_clean['Volume'] / 1e6,
            color='mediumpurple', width=1, alpha=0.7, label='Volume (M)')
axes[2].set_ylabel("Volume (millions)")
axes[2].set_xlabel("Date")
axes[2].legend(loc='upper left', fontsize=8)

plt.tight_layout()
plt.show()

# ── Section 3: Technical indicators ────────────────────────────────
fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
fig.suptitle(f"{TICKER} — Technical Indicators", fontsize=14, fontweight='bold')

# RSI
axes[0].plot(data_clean.index, data_clean['RSI_14'],
             color='darkorange', linewidth=0.9, label='RSI 14')
axes[0].axhline(70, color='red',   linestyle='--', linewidth=0.8, alpha=0.7, label='Overbought (70)')
axes[0].axhline(30, color='green', linestyle='--', linewidth=0.8, alpha=0.7, label='Oversold (30)')
axes[0].axhline(50, color='grey',  linestyle=':',  linewidth=0.6)
axes[0].set_ylabel("RSI")
axes[0].set_ylim(0, 100)
axes[0].legend(loc='upper left', fontsize=8)

# MACD
axes[1].plot(data_clean.index, data_clean['MACD'],
             color='steelblue', linewidth=0.9, label='MACD')
axes[1].plot(data_clean.index, data_clean['Signal_Line'],
             color='tomato', linewidth=0.9, label='Signal Line')
axes[1].bar(data_clean.index, data_clean['MACD_Histogram'],
            color=np.where(data_clean['MACD_Histogram'] >= 0, 'forestgreen', 'tomato'),
            width=1, alpha=0.5, label='Histogram')
axes[1].axhline(0, color='black', linewidth=0.5)
axes[1].set_ylabel("MACD")
axes[1].legend(loc='upper left', fontsize=8)

# Volatility
axes[2].plot(data_clean.index, data_clean['Volatility_20'],
             color='slategray', linewidth=0.9, label='Volatility 20d (std %)')
axes[2].set_ylabel("Volatility (%)")
axes[2].legend(loc='upper left', fontsize=8)

# Price vs EMA (relative)
axes[3].plot(data_clean.index, data_clean['Price_vs_EMA_20'],
             color='mediumpurple', linewidth=0.9, label='Price / EMA20')
axes[3].axhline(1.0, color='black', linestyle='--', linewidth=0.7)
axes[3].set_ylabel("Price / EMA20")
axes[3].set_xlabel("Date")
axes[3].legend(loc='upper left', fontsize=8)

plt.tight_layout()
plt.show()

# ── Section 4: Daily returns distribution ──────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f"{TICKER} — Daily Returns Distribution", fontsize=14, fontweight='bold')

# Histogram + KDE
sns.histplot(data_clean['Daily_Return'], bins=80, kde=True,
             ax=axes[0], color='steelblue', edgecolor='white', linewidth=0.3)
axes[0].axvline(0, color='red', linestyle='--', linewidth=1)
axes[0].set_xlabel("Daily Return (%)")
axes[0].set_title(f"Mean={data_clean['Daily_Return'].mean():.3f}%  "
                  f"Std={data_clean['Daily_Return'].std():.3f}%  "
                  f"Skew={data_clean['Daily_Return'].skew():.3f}")

# Box-plot by year
data_clean['Year'] = data_clean.index.year
sns.boxplot(data=data_clean, x='Year', y='Daily_Return',
            ax=axes[1], palette='coolwarm', linewidth=0.8)
axes[1].axhline(0, color='black', linewidth=0.6)
axes[1].set_xlabel("Year")
axes[1].set_ylabel("Daily Return (%)")
axes[1].set_title("Annual Distribution of Daily Returns")
axes[1].tick_params(axis='x', rotation=45)
data_clean.drop(columns='Year', inplace=True)

plt.tight_layout()
plt.show()

# ── Section 5: Correlation heatmap ─────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 9))
corr = data_clean[ML_FEATURES + ["Target"]].corr()

mask = np.triu(np.ones_like(corr, dtype=bool))   # hide upper triangle
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
            cmap='coolwarm', center=0, linewidths=0.5,
            annot_kws={"size": 8}, ax=ax)
ax.set_title(f"{TICKER} — Feature Correlation Matrix", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

# ── Section 6: Target class balance ────────────────────────────────
counts = data_clean['Target'].value_counts().sort_index()
labels = ['SELL / HOLD (0)', 'BUY (1)']
colors = ['tomato', 'forestgreen']

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle("Target Class Distribution", fontsize=13, fontweight='bold')

axes[0].bar(labels, counts.values, color=colors, edgecolor='white', linewidth=0.8)
axes[0].set_ylabel("Count")
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 5, str(v), ha='center', fontsize=11, fontweight='bold')

axes[1].pie(counts.values, labels=labels, colors=colors,
            autopct='%1.1f%%', startangle=90, pctdistance=0.75)
axes[1].set_title("Proportion")

print(f"\n  Class balance:\n  {counts.to_dict()}")
plt.tight_layout()
plt.show()

# ── Section 7: Feature distributions by class ──────────────────────
n_features = len(ML_FEATURES)
ncols = 2
nrows = (n_features + 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3.5))
axes = axes.flatten()
fig.suptitle(f"{TICKER} — Feature Distributions by Class", fontsize=13, fontweight='bold')

for i, feat in enumerate(ML_FEATURES):
    sns.kdeplot(data=data_clean, x=feat, hue='Target',
                palette={0: 'tomato', 1: 'forestgreen'},
                fill=True, alpha=0.35, ax=axes[i], common_norm=False,
                legend=(i == 0))
    axes[i].set_title(feat, fontsize=9)
    axes[i].set_xlabel("")

# Hide any unused subplot panels
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

handles, _ = axes[0].get_legend_handles_labels()
fig.legend(handles, ['SELL/HOLD (0)', 'BUY (1)'],
           loc='lower right', fontsize=10)

plt.tight_layout()
plt.show()

print("\nEDA complete.")
