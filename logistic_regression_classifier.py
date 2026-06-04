"""
Task 4: Binary Classification with Logistic Regression
Dataset: Breast Cancer Wisconsin Dataset (built-in from sklearn)
Tools: Scikit-learn, Pandas, Matplotlib, Seaborn
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve,
    accuracy_score, precision_score, recall_score, f1_score
)

# ══════════════════════════════════════════════════════
# 1. LOAD DATASET
# ══════════════════════════════════════════════════════
print("=" * 55)
print("   Logistic Regression – Breast Cancer Classification")
print("=" * 55)

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target   # 0 = Malignant, 1 = Benign

df.to_csv('breast_cancer.csv', index=False)
print(f"\n[1] Dataset loaded: {df.shape[0]} samples, {df.shape[1]-1} features")
print(f"    Classes — Malignant: {(df.target==0).sum()}  |  Benign: {(df.target==1).sum()}")

# ══════════════════════════════════════════════════════
# 2. TRAIN / TEST SPLIT & STANDARDIZE FEATURES
# ══════════════════════════════════════════════════════
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"\n[2] Train/Test split: {len(X_train)} train | {len(X_test)} test")
print("    Features standardized with StandardScaler")

# ══════════════════════════════════════════════════════
# 3. FIT LOGISTIC REGRESSION MODEL
# ══════════════════════════════════════════════════════
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_sc, y_train)

y_pred = model.predict(X_test_sc)
y_prob = model.predict_proba(X_test_sc)[:, 1]

print("\n[3] Logistic Regression model trained successfully.")

# ══════════════════════════════════════════════════════
# 4. EVALUATION METRICS
# ══════════════════════════════════════════════════════
acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)
auc  = roc_auc_score(y_test, y_prob)
cm   = confusion_matrix(y_test, y_pred)

print("\n[4] Evaluation Metrics:")
print(f"    Accuracy  : {acc:.4f}")
print(f"    Precision : {prec:.4f}")
print(f"    Recall    : {rec:.4f}")
print(f"    F1-Score  : {f1:.4f}")
print(f"    ROC-AUC   : {auc:.4f}")
print("\n    Confusion Matrix:")
print(f"    {cm}")
print("\n    Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Malignant', 'Benign']))

# ══════════════════════════════════════════════════════
# 5. THRESHOLD TUNING
# ══════════════════════════════════════════════════════
thresholds = np.arange(0.1, 0.9, 0.01)
precisions, recalls, f1s = [], [], []

for t in thresholds:
    yp = (y_prob >= t).astype(int)
    precisions.append(precision_score(y_test, yp, zero_division=0))
    recalls.append(recall_score(y_test, yp, zero_division=0))
    f1s.append(f1_score(y_test, yp, zero_division=0))

best_t = thresholds[np.argmax(f1s)]
print(f"[5] Optimal decision threshold (max F1): {best_t:.2f}")

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob)

# ══════════════════════════════════════════════════════
# 6. VISUALISATIONS
# ══════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor('#0f172a')
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)

TITLE_C  = '#f8fafc'
LABEL_C  = '#cbd5e1'
GRID_C   = '#1e293b'
ACC_C    = '#38bdf8'
MALIGN_C = '#f87171'
BENIGN_C = '#4ade80'
ROC_C    = '#a78bfa'
SIG_C    = '#fb923c'
THR_C    = '#facc15'

def style_ax(ax, title=''):
    ax.set_facecolor(GRID_C)
    for spine in ax.spines.values():
        spine.set_edgecolor('#334155')
    ax.tick_params(colors=LABEL_C, labelsize=8)
    ax.xaxis.label.set_color(LABEL_C)
    ax.yaxis.label.set_color(LABEL_C)
    if title:
        ax.set_title(title, color=TITLE_C, fontsize=10, fontweight='bold', pad=8)
    ax.grid(color='#1e3a5f', linestyle='--', linewidth=0.5, alpha=0.6)

# (A) Confusion Matrix
ax0 = fig.add_subplot(gs[0, 0])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Malignant', 'Benign'],
            yticklabels=['Malignant', 'Benign'],
            ax=ax0, cbar=False,
            annot_kws={'size': 14, 'weight': 'bold', 'color': 'white'})
ax0.set_facecolor(GRID_C)
ax0.set_xlabel('Predicted', color=LABEL_C, fontsize=9)
ax0.set_ylabel('Actual',    color=LABEL_C, fontsize=9)
ax0.set_title('Confusion Matrix', color=TITLE_C, fontsize=10, fontweight='bold', pad=8)
ax0.tick_params(colors=LABEL_C)

# (B) ROC Curve
ax1 = fig.add_subplot(gs[0, 1])
ax1.plot(fpr, tpr, color=ROC_C, lw=2, label=f'AUC = {auc:.4f}')
ax1.plot([0, 1], [0, 1], 'w--', lw=1, alpha=0.4)
ax1.fill_between(fpr, tpr, alpha=0.15, color=ROC_C)
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.legend(facecolor='#1e293b', edgecolor='none', labelcolor=LABEL_C, fontsize=8)
style_ax(ax1, 'ROC Curve')

# (C) Sigmoid Function
ax2 = fig.add_subplot(gs[0, 2])
z   = np.linspace(-8, 8, 300)
sig = 1 / (1 + np.exp(-z))
ax2.plot(z, sig, color=SIG_C, lw=2.5)
ax2.axhline(0.5, color='white', lw=1, ls='--', alpha=0.5)
ax2.axvline(0,   color='white', lw=1, ls='--', alpha=0.5)
ax2.fill_between(z, sig, 0.5, where=(z > 0), alpha=0.15, color=BENIGN_C)
ax2.fill_between(z, sig, 0.5, where=(z < 0), alpha=0.15, color=MALIGN_C)
ax2.set_xlabel('z  (linear combination)')
ax2.set_ylabel('sigma(z)')
style_ax(ax2, 'Sigmoid Function  s(z) = 1 / (1 + e^-z)')

# (D) Precision / Recall / F1 vs Threshold
ax3 = fig.add_subplot(gs[1, 0:2])
ax3.plot(thresholds, precisions, color=ACC_C,    lw=2, label='Precision')
ax3.plot(thresholds, recalls,    color=BENIGN_C, lw=2, label='Recall')
ax3.plot(thresholds, f1s,        color=THR_C,    lw=2, label='F1-Score')
ax3.axvline(best_t, color='white', ls='--', lw=1.5, alpha=0.8,
            label=f'Best threshold = {best_t:.2f}')
ax3.set_xlabel('Threshold')
ax3.set_ylabel('Score')
ax3.legend(facecolor='#1e293b', edgecolor='none', labelcolor=LABEL_C, fontsize=8, ncol=4)
ax3.set_xlim(0.1, 0.9)
ax3.set_ylim(0, 1.05)
style_ax(ax3, 'Precision / Recall / F1-Score vs Decision Threshold')

# (E) Metrics Bar Chart
ax4 = fig.add_subplot(gs[1, 2])
metrics_dict = {'Accuracy': acc, 'Precision': prec,
                'Recall': rec, 'F1': f1, 'ROC-AUC': auc}
colors = [ACC_C, BENIGN_C, MALIGN_C, THR_C, ROC_C]
bars = ax4.barh(list(metrics_dict.keys()), list(metrics_dict.values()),
                color=colors, height=0.5)
for bar, val in zip(bars, metrics_dict.values()):
    ax4.text(val - 0.02, bar.get_y() + bar.get_height() / 2,
             f'{val:.4f}', va='center', ha='right',
             color='#0f172a', fontsize=9, fontweight='bold')
ax4.set_xlim(0, 1.05)
style_ax(ax4, 'Evaluation Metrics')

# (F) Top 10 Feature Coefficients
ax5 = fig.add_subplot(gs[2, :])
coefs = pd.Series(model.coef_[0], index=data.feature_names)
top10 = coefs.abs().nlargest(10).index
coefs_top  = coefs[top10].sort_values()
bar_colors = [BENIGN_C if v > 0 else MALIGN_C for v in coefs_top]
ax5.barh(coefs_top.index, coefs_top.values, color=bar_colors, height=0.55)
ax5.axvline(0, color='white', lw=1, alpha=0.5)
ax5.set_xlabel('Coefficient Value')
style_ax(ax5,
    'Top 10 Feature Coefficients  '
    '(Green = Benign indicator  |  Red = Malignant indicator)')

fig.suptitle('Logistic Regression — Breast Cancer Classification',
             color=TITLE_C, fontsize=15, fontweight='bold', y=0.98)

output_path = 'logistic_regression_results.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close()
print(f"\n[6] Visualisation saved → {output_path}")
print("\nDone! All outputs generated successfully.")
