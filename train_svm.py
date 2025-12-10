import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
import pickle
import time
import warnings
import os

warnings.filterwarnings('ignore')

class SVMTrainer:
    def __init__(self):
        self.datasets = {
            'Coimbra': 'data/Coimbra.csv',
            'Haberman': 'data/Haberman.csv',
            'WDBC': 'data/WDBC.csv',
            'Wisconsin': 'data/Wisconsin_Original.csv'
        }
        self.kernels = ['linear', 'poly', 'rbf', 'sigmoid']
        self.results = {}

    def load_dataset(self, dataset_name):
        print(f"\n{'='*60}")
        print(f"CHARGEMENT : {dataset_name}")
        print(f"{'='*60}")

        filepath = self.datasets[dataset_name]
        df = pd.read_csv(filepath, header=None if dataset_name != 'Coimbra' else 'infer')

        if dataset_name == 'Coimbra':
            X = df.drop('Classification', axis=1).values
            y = (df['Classification'] == 2).astype(int).values
            print(f"Coimbra → {X.shape[0]} échantillons, {y.sum()} malades")

        elif dataset_name == 'Haberman':
            X = df.iloc[:, :3].values
            y = (df.iloc[:, 3] == 2).astype(int).values
            print(f"Haberman → {X.shape[0]} patientes, {y.sum()} décès")

        elif dataset_name == 'WDBC':
            df.columns = ['id', 'diag'] + [f'f{i}' for i in range(30)]
            X = df.iloc[:, 2:].values
            y = (df['diag'] == 'M').astype(int).values
            print(f"WDBC → {X.shape[0]} patientes, {y.sum()} malignes")

        elif dataset_name == 'Wisconsin':
            df = df.replace('?', np.nan).dropna().astype(int)
            X = df.iloc[:, 1:10].values
            y = (df.iloc[:, 10] == 4).astype(int).values
            print(f"Wisconsin Original → {X.shape[0]} patientes, {y.sum()} malignes")

        return np.array(X, dtype=float), np.array(y, dtype=int)

    def create_pipeline(self, kernel):
        return Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=0.95, random_state=42)),
            ('svm', SVC(kernel=kernel, probability=True, random_state=42, max_iter=30000))
        ])

    def train_single_model(self, X_train, X_test, y_train, y_test, kernel):
        print(f"   → Noyau {kernel.upper():<8} ... ", end="")

        if kernel == 'linear':
            params = {'svm__C': [0.1, 1, 10, 100, 1000]}
        elif kernel == 'poly':
            params = {'svm__C': [1, 10, 100], 'svm__degree': [3], 'svm__coef0': [0, 1], 'svm__gamma': ['scale', 'auto']}
        elif kernel == 'rbf':
            params = {'svm__C': [1, 10, 100, 1000], 'svm__gamma': ['scale', 'auto', 0.01, 0.1]}
        else:  # sigmoid
            params = {'svm__C': [0.1, 1, 10, 100], 'svm__gamma': ['scale', 0.01, 0.1]}

        grid = GridSearchCV(self.create_pipeline(kernel), params, cv=5, scoring='accuracy', n_jobs=-1)
        grid.fit(X_train, y_train)

        pred = grid.predict(X_test)
        proba = grid.predict_proba(X_test)

        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, average='weighted')
        auc = roc_auc_score(y_test, proba[:, 1]) if len(np.unique(y_test)) == 2 else None

        print(f"OK → Accuracy = {acc:.4f}" + (f" | AUC = {auc:.4f}" if auc else ""))

        return grid.best_estimator_, acc, f1, auc

    def train_all_models(self):
        print("\n" + "="*70)
        print("DÉBUT ENTRAÎNEMENT – 4 DATASETS × 4 NOYAUX")
        print("="*70)

        os.makedirs('models/saved_models', exist_ok=True)

        for name in self.datasets.keys():
            try:
                X, y = self.load_dataset(name)

                if len(np.unique(y)) < 2:
                    print("Dataset ignoré : pas assez de classes")
                    continue

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )

                self.results[name] = {}

                for k in self.kernels:
                    try:
                        model, acc, f1, auc = self.train_single_model(X_train, X_test, y_train, y_test, k)
                        self.results[name][k] = {'model': model, 'accuracy': acc, 'f1': f1, 'auc': auc}

                        with open(f'models/saved_models/{name}_{k}_model.pkl', 'wb') as f:
                            pickle.dump(model, f)

                    except Exception as e:
                        print(f"   Échec {k} : {e}")

            except Exception as e:
                print(f"ERREUR FATALE {name} : {e}")

        with open('models/saved_models/all_results.pkl', 'wb') as f:
            pickle.dump(self.results, f)

        print("\n" + "="*70)
        print("TOUT EST TERMINÉ – MODÈLES SAUVEGARDÉS")
        self.print_summary()

    def print_summary(self):
        print("\n" + "="*70)
        print("RÉSUMÉ FINAL – MEILLEUR NOYAU PAR DATASET")
        print("="*70)

        for dataset, res in self.results.items():
            if not res:  # si vide
                print(f"{dataset:<15} → aucun modèle entraîné")
                continue
            best = max(res.items(), key=lambda x: x[1]['accuracy'])
            print(f"{dataset:<15} → {best[0].upper():<8} | Accuracy = {best[1]['accuracy']:.4f}")

        print("\nProjet 100 % fonctionnel !")
        print("Lancez : streamlit run app.py")

# ===================================
if __name__ == "__main__":
    trainer = SVMTrainer()
    trainer.train_all_models()