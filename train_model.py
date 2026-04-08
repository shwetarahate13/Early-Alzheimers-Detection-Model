

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve, f1_score
)
from xgboost import XGBClassifier


COLORS = {
    'teal'   : '#1D9E75',
    'coral'  : '#D85A30',
    'blue'   : '#378ADD',
    'purple' : '#7F77DD',
    'amber'  : '#EF9F27',
    'gray'   : '#888780',
    'light'  : '#F1EFE8',
}
MODEL_COLORS = [COLORS['teal'], COLORS['blue'], COLORS['coral'], COLORS['purple']]
sns.set_style("whitegrid")
plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.spines.top': False,
                     'axes.spines.right': False})


#  SECTION 1 — LOAD & CLEAN DATA

print("\n" + "="*60)
print("  ALZHEIMER PREDICTION — MULTI-MODEL + EDA")
print("="*60)

df = pd.read_csv("/content/drive/MyDrive/oasis_cross-sectional.csv")
print(f"\n Raw dataset: {df.shape[0]} rows × {df.shape[1]} cols")

# Drop unused columns
drop_cols = ['ID', 'Hand', 'Delay', 'MR Delay']
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

# Drop rows with missing CDR (target)
df.dropna(subset=['CDR'], inplace=True)

# Fill missing
df['SES'].fillna(df['SES'].median(), inplace=True)
df['MMSE'].fillna(df['MMSE'].median(), inplace=True)

# Encode gender
df['Gender'] = df['M/F'].map({'M': 1, 'F': 0})

# Binary target
df['Alzheimer'] = (df['CDR'] > 0).astype(int)
df['Diagnosis'] = df['Alzheimer'].map({0: 'No Alzheimer', 1: 'Has Alzheimer'})

print(f" Cleaned dataset: {df.shape[0]} rows")
print(f" Class balance  : {df['Alzheimer'].value_counts().to_dict()}")



#  SECTION 2 — GRAPHICAL EDA  (10 plots across 3 figures)
print("\n\n--- Generating EDA Plots ---")

DIAG_PALETTE = {'No Alzheimer': COLORS['teal'], 'Has Alzheimer': COLORS['coral']}

# ── EDA Figure 1: Overview (3×2) ────────────────────────────
fig1, axes = plt.subplots(2, 3, figsize=(16, 10))
fig1.suptitle("EDA — Patient Overview", fontsize=17, fontweight='bold', y=1.01)

# 1a. Class distribution (donut)
ax = axes[0, 0]
counts = df['Diagnosis'].value_counts()
wedge_colors = [COLORS['teal'], COLORS['coral']]
wedges, texts, autotexts = ax.pie(
    counts, labels=counts.index, autopct='%1.1f%%',
    colors=wedge_colors, startangle=90,
    wedgeprops=dict(width=0.55), textprops={'fontsize': 11}
)
for at in autotexts:
    at.set_fontsize(11); at.set_fontweight('bold')
ax.set_title('Class Distribution', fontsize=13, pad=10)

# 1b. Age distribution by diagnosis
ax = axes[0, 1]
for label, grp in df.groupby('Diagnosis'):
    ax.hist(grp['Age'], bins=15, alpha=0.7,
            color=DIAG_PALETTE[label], label=label, edgecolor='white')
ax.set_title('Age Distribution', fontsize=13)
ax.set_xlabel('Age'); ax.set_ylabel('Count')
ax.legend(fontsize=9)

# 1c. Gender breakdown
ax = axes[0, 2]
gender_diag = df.groupby(['M/F', 'Diagnosis']).size().unstack(fill_value=0)
gender_diag.plot(kind='bar', ax=ax, color=[COLORS['teal'], COLORS['coral']],
                 edgecolor='white', width=0.6)
ax.set_title('Gender vs Diagnosis', fontsize=13)
ax.set_xlabel('Gender'); ax.set_ylabel('Count')
ax.set_xticklabels(['Female', 'Male'], rotation=0)
ax.legend(fontsize=9)

# 1d. MMSE score by diagnosis (violin)
ax = axes[1, 0]
for i, (label, grp) in enumerate(df.groupby('Diagnosis')):
    parts = ax.violinplot(grp['MMSE'].dropna(), positions=[i], widths=0.6,
                          showmedians=True)
    for pc in parts['bodies']:
        pc.set_facecolor(DIAG_PALETTE[label]); pc.set_alpha(0.75)
    parts['cmedians'].set_color('black')
ax.set_xticks([0, 1])
ax.set_xticklabels(['No Alzheimer', 'Has Alzheimer'])
ax.set_title('MMSE Score Distribution', fontsize=13)
ax.set_ylabel('MMSE Score')

# 1e. Education level breakdown
ax = axes[1, 1]
educ_order = sorted(df['Educ'].dropna().unique())
educ_df = df.groupby(['Educ', 'Diagnosis']).size().unstack(fill_value=0)
educ_df.plot(kind='bar', ax=ax, color=[COLORS['teal'], COLORS['coral']],
             edgecolor='white', width=0.7)
ax.set_title('Education Level vs Diagnosis', fontsize=13)
ax.set_xlabel('Education Level (1=low, 5=high)')
ax.set_ylabel('Count')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.legend(fontsize=9)

# 1f. Brain volume (nWBV) by diagnosis (box)
ax = axes[1, 2]
data_no  = df[df['Alzheimer'] == 0]['nWBV'].dropna()
data_yes = df[df['Alzheimer'] == 1]['nWBV'].dropna()
bp = ax.boxplot([data_no, data_yes], patch_artist=True,
                medianprops=dict(color='black', linewidth=2))
bp['boxes'][0].set_facecolor(COLORS['teal'])
bp['boxes'][1].set_facecolor(COLORS['coral'])
for box in bp['boxes']:
    box.set_alpha(0.7)
ax.set_xticks([1, 2])
ax.set_xticklabels(['No Alzheimer', 'Has Alzheimer'])
ax.set_title('Normalized Brain Volume (nWBV)', fontsize=13)
ax.set_ylabel('nWBV')

plt.tight_layout()
plt.savefig('eda_overview.png', dpi=150, bbox_inches='tight')
plt.show()
print("  Saved: eda_overview.png")


# ── EDA Figure 2: Correlation & Relationships ───────────────
fig2, axes = plt.subplots(2, 2, figsize=(14, 11))
fig2.suptitle("EDA — Feature Relationships & Correlation", fontsize=17, fontweight='bold')

num_cols = ['Age', 'Educ', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF', 'Alzheimer']
num_df   = df[num_cols].dropna()

# 2a. Correlation heatmap
ax = axes[0, 0]
corr = num_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, ax=ax, cmap='RdYlGn', center=0,
            annot=True, fmt='.2f', square=True, linewidths=0.5,
            cbar_kws={'shrink': 0.8}, annot_kws={'size': 9})
ax.set_title('Feature Correlation Heatmap', fontsize=13)

# 2b. MMSE vs Age scatter
ax = axes[0, 1]
for label, grp in df.groupby('Diagnosis'):
    ax.scatter(grp['Age'], grp['MMSE'], alpha=0.6, s=35,
               color=DIAG_PALETTE[label], label=label)
ax.set_xlabel('Age'); ax.set_ylabel('MMSE Score')
ax.set_title('Age vs MMSE Score', fontsize=13)
ax.legend(fontsize=9)

# 2c. nWBV vs Age
ax = axes[1, 0]
for label, grp in df.groupby('Diagnosis'):
    ax.scatter(grp['Age'], grp['nWBV'], alpha=0.6, s=35,
               color=DIAG_PALETTE[label], label=label)
    z = np.polyfit(grp['Age'].dropna(), grp['nWBV'].dropna(), 1)
    p = np.poly1d(z)
    x_line = np.linspace(grp['Age'].min(), grp['Age'].max(), 100)
    ax.plot(x_line, p(x_line), color=DIAG_PALETTE[label], lw=1.5, alpha=0.9)
ax.set_xlabel('Age'); ax.set_ylabel('Normalized Brain Volume')
ax.set_title('Brain Volume Decline with Age', fontsize=13)
ax.legend(fontsize=9)

# 2d. SES vs MMSE heatmap-style
ax = axes[1, 1]
pivot = df.groupby(['SES', 'Educ'])['MMSE'].mean().unstack(fill_value=np.nan)
sns.heatmap(pivot, ax=ax, cmap='YlOrRd_r', annot=True, fmt='.1f',
            linewidths=0.5, cbar_kws={'label': 'Mean MMSE Score'})
ax.set_title('Mean MMSE by SES & Education', fontsize=13)
ax.set_xlabel('Education Level'); ax.set_ylabel('SES (1=high, 5=low)')

plt.tight_layout()
plt.savefig('eda_correlation.png', dpi=150, bbox_inches='tight')
plt.show()
print("  Saved: eda_correlation.png")


# ── EDA Figure 3: Feature Distributions (KDE) ───────────────
fig3, axes = plt.subplots(2, 3, figsize=(16, 9))
fig3.suptitle("EDA — Feature Distributions by Diagnosis", fontsize=17, fontweight='bold')

kde_features = ['Age', 'MMSE', 'nWBV', 'eTIV', 'ASF', 'SES']
for ax, feat in zip(axes.flat, kde_features):
    for label, grp in df.groupby('Diagnosis'):
        data = grp[feat].dropna()
        ax.hist(data, bins=18, density=True, alpha=0.35,
                color=DIAG_PALETTE[label])
        data.plot.kde(ax=ax, color=DIAG_PALETTE[label],
                      label=label, linewidth=2)
    ax.set_title(f'{feat} Distribution', fontsize=12)
    ax.set_xlabel(feat); ax.set_ylabel('Density')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('eda_distributions.png', dpi=150, bbox_inches='tight')
plt.show()
print("  Saved: eda_distributions.png")



#  SECTION 3 — MODEL TRAINING

print("\n\n--- Training 4 Models ---")

FEATURES = [f for f in ['Gender', 'Age', 'Educ', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']
            if f in df.columns]
X = df[FEATURES]
y = df['Alzheimer']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale for SVM & KNN
scaler  = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── Define Models ────────────────────────────────────────────
models = {
    'Random Forest': {
        'model': RandomForestClassifier(
            n_estimators=200, max_depth=10,
            min_samples_split=5, random_state=42, class_weight='balanced'),
        'scaled': False,
        'color': COLORS['teal']
    },
    'XGBoost': {
        'model': XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, use_label_encoder=False,
            eval_metric='logloss', random_state=42),
        'scaled': False,
        'color': COLORS['blue']
    },
    'SVM': {
        'model': SVC(
            C=1.0, kernel='rbf', gamma='scale',
            probability=True, random_state=42, class_weight='balanced'),
        'scaled': True,
        'color': COLORS['coral']
    },
    'KNN': {
        'model': KNeighborsClassifier(
            n_neighbors=7, metric='minkowski', weights='distance'),
        'scaled': True,
        'color': COLORS['purple']
    },
}

# ── Train & Collect Results ──────────────────────────────────
results = {}

for name, cfg in models.items():
    mdl = cfg['model']
    Xtr = X_train_sc if cfg['scaled'] else X_train
    Xte = X_test_sc  if cfg['scaled'] else X_test
    Xall = scaler.transform(X) if cfg['scaled'] else X

    mdl.fit(Xtr, y_train)
    y_pred  = mdl.predict(Xte)
    y_proba = mdl.predict_proba(Xte)[:, 1]

    cv = cross_val_score(mdl, Xall, y, cv=StratifiedKFold(5), scoring='accuracy')

    results[name] = {
        'model'    : mdl,
        'y_pred'   : y_pred,
        'y_proba'  : y_proba,
        'accuracy' : accuracy_score(y_test, y_pred),
        'roc_auc'  : roc_auc_score(y_test, y_proba),
        'f1'       : f1_score(y_test, y_pred),
        'cv_mean'  : cv.mean(),
        'cv_std'   : cv.std(),
        'color'    : cfg['color'],
    }
    print(f"  {name:<16} Acc={results[name]['accuracy']*100:.1f}%  "
          f"AUC={results[name]['roc_auc']:.3f}  "
          f"CV={results[name]['cv_mean']*100:.1f}±{results[name]['cv_std']*100:.1f}%")



#  SECTION 4 — MODEL COMPARISON PLOTS

print("\n--- Generating Model Comparison Plots ---")

names  = list(results.keys())
colors = [results[n]['color'] for n in names]

# ── Figure 4: Model Comparison Dashboard ────────────────────
fig4 = plt.figure(figsize=(18, 14))
fig4.suptitle("Model Comparison Dashboard", fontsize=18, fontweight='bold', y=1.01)
gs = gridspec.GridSpec(3, 4, figure=fig4, hspace=0.45, wspace=0.35)

# 4a. Accuracy bar chart
ax1 = fig4.add_subplot(gs[0, :2])
accs = [results[n]['accuracy']*100 for n in names]
bars = ax1.bar(names, accs, color=colors, edgecolor='white', width=0.55)
ax1.set_ylim(0, 110)
ax1.set_title('Test Accuracy (%)', fontsize=13)
ax1.set_ylabel('Accuracy (%)')
for bar, val in zip(bars, accs):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
best_idx = accs.index(max(accs))
bars[best_idx].set_edgecolor('gold'); bars[best_idx].set_linewidth(2.5)

# 4b. ROC-AUC bar
ax2 = fig4.add_subplot(gs[0, 2:])
aucs = [results[n]['roc_auc'] for n in names]
bars2 = ax2.bar(names, aucs, color=colors, edgecolor='white', width=0.55)
ax2.set_ylim(0, 1.15)
ax2.set_title('ROC-AUC Score', fontsize=13)
ax2.set_ylabel('AUC Score')
for bar, val in zip(bars2, aucs):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# 4c. ROC Curves (all 4 on one plot)
ax3 = fig4.add_subplot(gs[1, :2])
for name in names:
    fpr, tpr, _ = roc_curve(y_test, results[name]['y_proba'])
    ax3.plot(fpr, tpr, color=results[name]['color'], lw=2,
             label=f"{name} (AUC={results[name]['roc_auc']:.2f})")
ax3.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
ax3.set_xlabel('False Positive Rate'); ax3.set_ylabel('True Positive Rate')
ax3.set_title('ROC Curves — All Models', fontsize=13)
ax3.legend(fontsize=9, loc='lower right')

# 4d. Cross-validation scores
ax4 = fig4.add_subplot(gs[1, 2:])
cv_means = [results[n]['cv_mean']*100 for n in names]
cv_stds  = [results[n]['cv_std']*100  for n in names]
ax4.bar(names, cv_means, color=colors, edgecolor='white', width=0.55, alpha=0.85)
ax4.errorbar(names, cv_means, yerr=cv_stds, fmt='none',
             color='black', capsize=6, linewidth=1.5)
ax4.set_ylim(0, 110)
ax4.set_title('5-Fold Cross-Validation Accuracy', fontsize=13)
ax4.set_ylabel('CV Accuracy (%)')
for i, (m, s) in enumerate(zip(cv_means, cv_stds)):
    ax4.text(i, m + s + 2, f'{m:.1f}%', ha='center', fontsize=10, fontweight='bold')

# 4e-4h. Confusion matrices (one per model)
for i, name in enumerate(names):
    ax = fig4.add_subplot(gs[2, i])
    cm = confusion_matrix(y_test, results[name]['y_pred'])
    cmap = sns.light_palette(results[name]['color'], as_cmap=True)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax,
                xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'],
                linewidths=0.5, cbar=False, annot_kws={'size': 12})
    ax.set_title(f'{name}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=9)
    ax.set_ylabel('Actual', fontsize=9)

plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("  Saved: model_comparison.png")


# ── Figure 5: Metrics Summary + Feature Importance ──────────
fig5, axes = plt.subplots(1, 2, figsize=(15, 6))
fig5.suptitle("Metrics Summary & Feature Importance", fontsize=15, fontweight='bold')

# 5a. Grouped metric bars
metric_data = {
    'Accuracy' : [results[n]['accuracy'] for n in names],
    'ROC-AUC'  : [results[n]['roc_auc']  for n in names],
    'F1 Score' : [results[n]['f1']        for n in names],
}
x = np.arange(len(names))
width = 0.26
bar_colors = [COLORS['teal'], COLORS['blue'], COLORS['amber']]
ax = axes[0]
for i, (metric, vals) in enumerate(metric_data.items()):
    ax.bar(x + i*width, vals, width, label=metric,
           color=bar_colors[i], edgecolor='white', alpha=0.9)
ax.set_xticks(x + width)
ax.set_xticklabels(names, fontsize=10)
ax.set_ylim(0, 1.15)
ax.set_title('All Metrics per Model', fontsize=13)
ax.set_ylabel('Score')
ax.legend(fontsize=10)
ax.axhline(0.8, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
ax.text(3.6, 0.81, '0.80 baseline', fontsize=8, color='gray')

# 5b. Random Forest feature importance
ax = axes[1]
rf_model = results['Random Forest']['model']
importances = pd.Series(rf_model.feature_importances_, index=FEATURES).sort_values()
bar_colors_feat = [COLORS['coral'] if v == importances.max() else COLORS['blue']
                   for v in importances]
importances.plot(kind='barh', ax=ax, color=bar_colors_feat, edgecolor='white')
ax.set_title('Feature Importance (Random Forest)', fontsize=13)
ax.set_xlabel('Importance Score')
for i, v in enumerate(importances):
    ax.text(v + 0.002, i, f'{v:.3f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('metrics_features.png', dpi=150, bbox_inches='tight')
plt.show()
print("  Saved: metrics_features.png")



#  SECTION 5 — PREDICT ON NEW PATIENT (ALL MODELS)

print("\n\n--- Predict New Patient (All Models) ---")

new_patient_raw = pd.DataFrame([{
    'Gender': 1,   # Male
    'Age'   : 72,
    'Educ'  : 2,
    'SES'   : 3,
    'MMSE'  : 24,
    'eTIV'  : 1500,
    'nWBV'  : 0.72,
    'ASF'   : 1.1,
}])[FEATURES]

new_patient_sc = scaler.transform(new_patient_raw)

print(f"\n  Patient: Male, 72y, MMSE=24, nWBV=0.72\n")
for name, cfg in models.items():
    inp   = new_patient_sc if cfg['scaled'] else new_patient_raw
    pred  = results[name]['model'].predict(inp)[0]
    proba = results[name]['model'].predict_proba(inp)[0][1]
    risk  = 'HIGH' if proba > 0.6 else 'MEDIUM' if proba > 0.3 else 'LOW'
    label = 'HAS ALZHEIMER' if pred == 1 else 'NO ALZHEIMER'
    print(f"  {name:<16} → {label:<16} | Prob: {proba*100:.1f}%  Risk: {risk}")



#  FINAL SUMMARY TABLE

print("\n\n" + "="*65)
print("  FINAL MODEL COMPARISON SUMMARY")
print("="*65)
print(f"  {'Model':<18} {'Accuracy':>10} {'ROC-AUC':>10} {'F1':>8} {'CV Acc':>12}")
print("  " + "-"*60)
for name in names:
    r = results[name]
    print(f"  {name:<18} {r['accuracy']*100:>9.1f}%"
          f" {r['roc_auc']:>10.3f}"
          f" {r['f1']:>8.3f}"
          f" {r['cv_mean']*100:>8.1f}±{r['cv_std']*100:.1f}%")
best = max(results, key=lambda n: results[n]['roc_auc'])
print("="*65)
print(f"  Best Model: {best} (ROC-AUC = {results[best]['roc_auc']:.3f})")
print("="*65)
print("\n  Plots saved:")
print("    eda_overview.png      — class, age, gender, MMSE, education, brain vol")
print("    eda_correlation.png   — heatmap, scatter plots, SES vs education")
print("    eda_distributions.png — KDE distributions for all 6 features")
print("    model_comparison.png  — accuracy, AUC, ROC curves, confusion matrices")
print("    metrics_features.png  — grouped metrics + feature importance")
print("\n  NOTE: Educational purposes only. Consult a medical professional.")
print("="*65)
