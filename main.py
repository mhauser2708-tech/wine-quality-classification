"""Wine quality classification: KNN, SVM, Decision Tree, and a tuned Random Forest.

Loads the UCI red + white Vinho Verde wine datasets, combines them into one
multiclass quality-prediction problem (quality 3-9), and compares baseline
vs. hyperparameter-tuned classifiers.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

SEED = 42
ROOT = Path(__file__).parent
DATA_DIR = ROOT / 'data'
PLOTS_DIR = ROOT / 'figures'
TABLES_DIR = ROOT / 'tables'
PLOTS_DIR.mkdir(exist_ok=True)
TABLES_DIR.mkdir(exist_ok=True)

np.random.seed(SEED)

# ── Data ──────────────────────────────────────────────────────────────────────
df_red = pd.read_csv(DATA_DIR / 'winequality-red.csv', sep=';')
df_white = pd.read_csv(DATA_DIR / 'winequality-white.csv', sep=';')

df_red['type'] = 0    # red
df_white['type'] = 1  # white
df = pd.concat([df_red, df_white], ignore_index=True)

X = df.drop(columns='quality')
y = df['quality']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


def save_report(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, zero_division=0)
    text = f"--- {name.replace('_', ' ')} ---\nAccuracy: {acc:.4f}\nClassification Report:\n{report}"
    (TABLES_DIR / f'classification_report_{name}.txt').write_text(text)
    print(text)
    return acc


def save_confusion_matrix(name, y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels,
                yticklabels=labels, ax=ax)
    ax.set_title(f'Confusion Matrix — {name.replace("_", " ")}')
    ax.set_xlabel('Predicted quality')
    ax.set_ylabel('True quality')
    plt.tight_layout()
    idx = save_confusion_matrix.counter
    fig.savefig(PLOTS_DIR / f'plot_{idx:02d}_confusion_matrix_{name}.png', dpi=150)
    plt.close(fig)
    save_confusion_matrix.counter += 1


save_confusion_matrix.counter = 2  # plot_01 is the feature-importance plot, below

labels = sorted(y.unique())

# ── Baseline models ──────────────────────────────────────────────────────────
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
save_report('K-Nearest_Neighbors', y_test, knn.predict(X_test_scaled))
save_confusion_matrix('K-Nearest_Neighbors', y_test, knn.predict(X_test_scaled), labels)

svm = SVC(kernel='rbf', random_state=SEED)
svm.fit(X_train_scaled, y_train)
save_report('Support_Vector_Machine', y_test, svm.predict(X_test_scaled))
save_confusion_matrix('Support_Vector_Machine', y_test, svm.predict(X_test_scaled), labels)

tree = DecisionTreeClassifier(random_state=SEED)
tree.fit(X_train, y_train)
save_report('Decision_Tree', y_test, tree.predict(X_test))
save_confusion_matrix('Decision_Tree', y_test, tree.predict(X_test), labels)

# Feature importance plot (plot_01)
importances = pd.Series(tree.feature_importances_, index=X.columns).sort_values()
fig, ax = plt.subplots(figsize=(7, 6))
importances.plot.barh(ax=ax, color='steelblue')
ax.set_title('Decision Tree — Feature Importance')
ax.set_xlabel('Importance')
plt.tight_layout()
fig.savefig(PLOTS_DIR / 'plot_01_feature_importance_decision_tree.png', dpi=150)
plt.close(fig)

# ── Hyperparameter-tuned models ──────────────────────────────────────────────
tuned_rf = GridSearchCV(
    RandomForestClassifier(random_state=SEED),
    param_grid={'n_estimators': [100, 200], 'max_depth': [None, 10, 20]},
    cv=3, n_jobs=-1,
)
tuned_rf.fit(X_train, y_train)
save_report('Tuned_Random_Forest', y_test, tuned_rf.predict(X_test))
save_confusion_matrix('Tuned_Random_Forest', y_test, tuned_rf.predict(X_test), labels)

tuned_knn = GridSearchCV(
    KNeighborsClassifier(),
    param_grid={'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']},
    cv=3, n_jobs=-1,
)
tuned_knn.fit(X_train_scaled, y_train)
save_report('Tuned_K-Nearest_Neighbors', y_test, tuned_knn.predict(X_test_scaled))
save_confusion_matrix('Tuned_K-Nearest_Neighbors', y_test, tuned_knn.predict(X_test_scaled), labels)

tuned_svm = GridSearchCV(
    SVC(random_state=SEED),
    param_grid={'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']},
    cv=3, n_jobs=-1,
)
tuned_svm.fit(X_train_scaled, y_train)
save_report('Tuned_Support_Vector_Machine', y_test, tuned_svm.predict(X_test_scaled))
save_confusion_matrix('Tuned_Support_Vector_Machine', y_test, tuned_svm.predict(X_test_scaled), labels)

tuned_tree = GridSearchCV(
    DecisionTreeClassifier(random_state=SEED),
    param_grid={'max_depth': [5, 10, 20, None], 'min_samples_leaf': [1, 5, 10]},
    cv=3, n_jobs=-1,
)
tuned_tree.fit(X_train, y_train)
save_report('Tuned_Decision_Tree', y_test, tuned_tree.predict(X_test))
save_confusion_matrix('Tuned_Decision_Tree', y_test, tuned_tree.predict(X_test), labels)

print('\nAll classification reports written to tables/, all plots written to figures/.')
