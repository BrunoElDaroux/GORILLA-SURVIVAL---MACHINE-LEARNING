"""
src/model_utils.py
===================
Reusable modeling, evaluation, and interpretation utilities.

Import in any notebook:
    import sys; sys.path.append('..')
    from src.model_utils import build_rf, evaluate_model, plot_feature_importance
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle, os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    average_precision_score, precision_recall_curve, RocCurveDisplay,
    ConfusionMatrixDisplay
)
from sklearn.inspection import permutation_importance


# ── MODEL FACTORY ─────────────────────────────────────────────────────────────

def build_rf(n_estimators=500, min_samples_leaf=4, random_state=42):
    """
    Return a configured RandomForestClassifier with class balancing.

    Parameters mirror paper-recommended settings for ecological survival data
    (imbalanced classes, correlated features, moderate sample size).
    """
    return RandomForestClassifier(
        n_estimators     = n_estimators,
        max_depth        = None,
        min_samples_leaf = min_samples_leaf,
        max_features     = 'sqrt',
        class_weight     = 'balanced',
        oob_score        = True,
        n_jobs           = -1,
        random_state     = random_state,
    )


# ── CROSS-VALIDATION ──────────────────────────────────────────────────────────

def run_cv(model, X, y, n_splits=5, random_state=42):
    """
    Run stratified k-fold CV and return a summary DataFrame.

    Returns
    -------
    cv_df : pd.DataFrame with fold-level and mean scores
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results = cross_validate(
        model, X, y, cv=cv,
        scoring=['roc_auc', 'average_precision', 'f1', 'balanced_accuracy'],
        n_jobs=-1
    )
    summary = {}
    for k, v in results.items():
        if k.startswith('test_'):
            name = k.replace('test_', '')
            summary[name] = {'mean': v.mean(), 'std': v.std(), 'folds': v.tolist()}

    df = pd.DataFrame({m: [d['mean'], d['std']] for m, d in summary.items()},
                      index=['mean', 'std']).T
    return df, results


# ── EVALUATION ────────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, plot=True, save_path=None):
    """
    Full evaluation suite: classification report + ROC + PR + confusion matrix.

    Parameters
    ----------
    model     : fitted sklearn estimator
    X_test    : test features
    y_test    : true labels
    plot      : whether to render matplotlib figure
    save_path : optional path to save figure PNG

    Returns
    -------
    metrics_dict : dict with roc_auc, avg_precision, y_pred, y_prob
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_prob)
    avg_prec = average_precision_score(y_test, y_prob)

    print("── Classification Report ────────────────────────────────────────────")
    print(classification_report(y_test, y_pred, target_names=['Died (0)', 'Survived (1)']))
    print(f"  ROC-AUC          : {roc_auc:.4f}")
    print(f"  Average Precision: {avg_prec:.4f}")

    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        ConfusionMatrixDisplay(cm, display_labels=['Died', 'Survived']).plot(
            ax=axes[0], cmap='Blues', colorbar=False)
        axes[0].set_title('Confusion Matrix', fontsize=12, fontweight='bold')

        # ROC
        RocCurveDisplay.from_predictions(y_test, y_prob, ax=axes[1],
            name=f'RF (AUC={roc_auc:.3f})', color='#2980b9')
        axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[1].set_title('ROC Curve', fontsize=12, fontweight='bold')

        # PR
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        axes[2].plot(rec, prec, color='#27ae60', linewidth=2, label=f'AP={avg_prec:.3f}')
        axes[2].axhline(y_test.mean(), color='red', linestyle='--', alpha=0.5,
                        label=f'Baseline={y_test.mean():.3f}')
        axes[2].set_xlabel('Recall')
        axes[2].set_ylabel('Precision')
        axes[2].set_title('Precision–Recall Curve', fontsize=12, fontweight='bold')
        axes[2].legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved: {save_path}")
        plt.show()

    return {'roc_auc': roc_auc, 'avg_precision': avg_prec,
            'y_pred': y_pred, 'y_prob': y_prob}


# ── FEATURE IMPORTANCE ────────────────────────────────────────────────────────

def plot_feature_importance(model, feature_names, top_n=20, title='Feature Importance (MDI)',
                             save_path=None):
    """Plot horizontal bar chart of MDI feature importances."""
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.nlargest(top_n).sort_values()

    q75 = importances.quantile(0.75)
    colors = ['#c0392b' if v >= q75 else '#3498db' for v in importances.values]

    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.4)))
    ax.barh(importances.index, importances.values, color=colors,
            edgecolor='black', alpha=0.85)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Importance Score')

    red_patch   = mpatches.Patch(color='#c0392b', label='Top quartile')
    blue_patch  = mpatches.Patch(color='#3498db', label='Other')
    ax.legend(handles=[red_patch, blue_patch])

    for spine in ['top', 'right']: ax.spines[spine].set_visible(False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return importances


def compute_permutation_importance(model, X_test, y_test, feature_names,
                                   n_repeats=30, random_state=42, plot=True,
                                   save_path=None):
    """Compute and plot permutation importance."""
    result = permutation_importance(
        model, X_test, y_test, n_repeats=n_repeats,
        scoring='roc_auc', n_jobs=-1, random_state=random_state
    )
    perm_df = pd.DataFrame({
        'feature'  : feature_names,
        'mean_imp' : result.importances_mean,
        'std_imp'  : result.importances_std,
    }).sort_values('mean_imp', ascending=False).reset_index(drop=True)

    if plot:
        perm_sorted = perm_df.sort_values('mean_imp')
        fig, ax = plt.subplots(figsize=(9, max(5, len(perm_df) * 0.35)))
        colors = ['#e74c3c' if m > 0 else '#95a5a6' for m in perm_sorted['mean_imp']]
        ax.barh(perm_sorted['feature'], perm_sorted['mean_imp'],
                xerr=perm_sorted['std_imp'], color=colors, edgecolor='black', alpha=0.8)
        ax.axvline(0, color='black', linewidth=1)
        ax.set_title('Permutation Importance (ROC-AUC drop)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Mean decrease in ROC-AUC')
        for spine in ['top', 'right']: ax.spines[spine].set_visible(False)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    return perm_df


# ── MODEL PERSISTENCE ─────────────────────────────────────────────────────────

def save_model(model, path):
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✅ Model saved: {path}")


def load_model(path):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    print(f"✅ Model loaded: {path}")
    return model
