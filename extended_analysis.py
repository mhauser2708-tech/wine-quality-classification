"""Extra statistics for main.py's results: confidence intervals, macro-F1, sensitivity of
each model to its hyperparameters, and a check for duplicate rows shared by train and test.

Uses exactly the same data, split, scaling and grids as main.py. Writes tables/extended_analysis.txt.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

SEED = 42
ROOT = Path(__file__).parent
TABLES_DIR = ROOT / 'tables'
TABLES_DIR.mkdir(exist_ok=True)
lines = []


def log(text=''):
    print(text)
    lines.append(text)


def wilson(correct, n, z=1.96):
    p = correct / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centre - half, centre + half


# ── Same data and split as main.py ───────────────────────────────────────────
red = pd.read_csv(ROOT / 'data' / 'winequality-red.csv', sep=';')
white = pd.read_csv(ROOT / 'data' / 'winequality-white.csv', sep=';')
red['type'], white['type'] = 0, 1
df = pd.concat([red, white], ignore_index=True)
X, y = df.drop(columns='quality'), df['quality']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
y_true = y_test.values
n = len(y_true)

# ── Duplicates between train and test ────────────────────────────────────────
train_keys = set(map(tuple, X_train.values.round(6)))
in_train = np.array([tuple(row) in train_keys for row in X_test.values.round(6)])
log(f'Rows: {len(df)}, exact duplicate rows in the full data: {df.duplicated().sum()}')
log(f'Test rows with an identical feature vector in the training set: '
    f'{in_train.sum()} of {n} ({in_train.mean():.1%})')
log()

# ── Models (same settings as main.py) ────────────────────────────────────────
preds, grids = {}, {}
preds['KNN (baseline)'] = KNeighborsClassifier(n_neighbors=5).fit(X_train_s, y_train).predict(X_test_s)
preds['SVM (baseline)'] = SVC(kernel='rbf', random_state=SEED).fit(X_train_s, y_train).predict(X_test_s)
preds['Decision Tree (baseline)'] = DecisionTreeClassifier(random_state=SEED).fit(X_train, y_train).predict(X_test)

searches = {
    'Tuned Random Forest': (RandomForestClassifier(random_state=SEED),
                            {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]}, False),
    'Tuned KNN': (KNeighborsClassifier(),
                  {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']}, True),
    'Tuned SVM': (SVC(random_state=SEED),
                  {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}, True),
    'Tuned Decision Tree': (DecisionTreeClassifier(random_state=SEED),
                            {'max_depth': [5, 10, 20, None], 'min_samples_leaf': [1, 5, 10]}, False),
}
for name, (est, grid, scaled) in searches.items():
    gs = GridSearchCV(est, grid, cv=3, n_jobs=-1)
    gs.fit(X_train_s if scaled else X_train, y_train)
    preds[name] = gs.predict(X_test_s if scaled else X_test)
    grids[name] = gs

log('Best hyperparameters (3-fold CV on the training set)')
for name, gs in grids.items():
    log(f'  {name}: {gs.best_params_}  (CV accuracy {gs.best_score_:.4f})')
log()

# ── Test accuracy with 95% CI, macro-F1, accuracy by duplicate status ────────
log(f'{"Model":26s} {"Accuracy":>8s}  {"95% CI":>15s}  {"Macro-F1":>8s}  '
    f'{"Acc. (in train)":>15s}  {"Acc. (not in train)":>19s}')
for name, p in preds.items():
    correct = int((p == y_true).sum())
    lo, hi = wilson(correct, n)
    mf1 = f1_score(y_true, p, average='macro', zero_division=0)
    acc_dup = (p[in_train] == y_true[in_train]).mean()
    acc_new = (p[~in_train] == y_true[~in_train]).mean()
    log(f'{name:26s} {correct / n:8.4f}  [{lo:.3f}, {hi:.3f}]  {mf1:8.3f}  '
        f'{acc_dup:15.3f}  {acc_new:19.3f}')
log()

# ── Paired comparison (exact McNemar test) ───────────────────────────────────
def mcnemar_exact(a, b):
    ca, cb = preds[a] == y_true, preds[b] == y_true
    only_a, only_b = int((ca & ~cb).sum()), int((~ca & cb).sum())
    p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    log(f'  {a} vs {b}: only first correct {only_a}, only second correct {only_b}, p = {p:.4f}')


log('Paired comparison on the test set (exact McNemar test)')
mcnemar_exact('Tuned Random Forest', 'Tuned KNN')
mcnemar_exact('Tuned KNN', 'Tuned SVM')
mcnemar_exact('Tuned Random Forest', 'Decision Tree (baseline)')
log()

# ── Sensitivity to hyperparameters (spread of CV accuracy over each grid) ────
log('Sensitivity to hyperparameters: mean CV accuracy over the grid (min / max / range)')
for name, gs in grids.items():
    s = gs.cv_results_['mean_test_score']
    log(f'  {name:22s} {s.min():.4f} / {s.max():.4f} / {s.max() - s.min():.4f}  ({len(s)} settings)')

(TABLES_DIR / 'extended_analysis.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
