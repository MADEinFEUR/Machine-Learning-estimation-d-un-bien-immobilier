# ==========================================================
# Version optimisée avec gestion dynamique du type de local
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans

df = pd.read_csv("Machine-Learning-estimation-d-un-bien-immobilier/ValeursFoncieres2024_06000.txt", sep="|", dtype=str)

# Nettoyage des colonnes numériques
num_cols = [
    "Valeur fonciere", "Surface reelle bati", "Surface Carrez du 1er lot",
    "Surface Carrez du 2eme lot", "Surface Carrez du 3eme lot",
    "Surface Carrez du 4eme lot", "Surface Carrez du 5eme lot",
    "Nombre de lots", "Nombre pieces principales", "Surface terrain"
]

for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c].str.replace(" ", "").str.replace(",", "."), errors='coerce')

# Surface totale
surface_cols = [
    "Surface reelle bati", "Surface Carrez du 1er lot", "Surface Carrez du 2eme lot",
    "Surface Carrez du 3eme lot", "Surface Carrez du 4eme lot", "Surface Carrez du 5eme lot"
]
df["Surface totale"] = df[surface_cols].sum(axis=1, skipna=True)

df["Date mutation"] = pd.to_datetime(df["Date mutation"], format="%d/%m/%Y", errors="coerce")
df["Annee mutation"] = df["Date mutation"].dt.year

# --- Nettoyage des valeurs aberrantes ---
# Valeur foncière
q1_val = df["Valeur fonciere"].quantile(0.0)
q99_val = df["Valeur fonciere"].quantile(1)
df = df[(df["Valeur fonciere"] >= q1_val) & (df["Valeur fonciere"] <= q99_val)]

# Surface totale
q1_surf = df["Surface totale"].quantile(0.0)
q99_surf = df["Surface totale"].quantile(1)
df = df[(df["Surface totale"] >= q1_surf) & (df["Surface totale"] <= q99_surf)]

# Nombre de pièces principales (optionnel)
q1_pieces = df["Nombre pieces principales"].quantile(0.0)
q99_pieces = df["Nombre pieces principales"].quantile(1)
df = df[(df["Nombre pieces principales"] >= q1_pieces) & (df["Nombre pieces principales"] <= q99_pieces)]

group_cols = ['Date mutation', 'No voie', 'Type de voie', 'Voie', 'Code postal', 'Commune']
df['id_vente'] = df.groupby(group_cols).ngroup()
dependances = df[df['Type local'] == 'Dépendance'].groupby('id_vente').size().rename('nb_dependances')
df = df.merge(dependances, on='id_vente', how='left')
df['nb_dependances'] = df['nb_dependances'].fillna(0).astype(int)

# Filtre sur les biens principaux : modèle multi-type (tous sauf dépendance)
df = df[df["Type local"] != "Dépendance"]
df["log_valeur"] = np.log1p(df["Valeur fonciere"])
df["habitable"] = ((df["Surface totale"] > 0) & (df["Type local"].isin(["Maison", "Appartement"]))).astype(int)
df = df.dropna(subset=["Valeur fonciere", "Surface totale", "Nombre pieces principales", "Annee mutation"])

surface_bins = [0, 60, 80, 120, 200, np.inf]
surface_labels = ["XS", "S", "M", "L", "XL"]
df["surface_cat"] = pd.cut(df["Surface totale"], bins=surface_bins, labels=surface_labels, include_lowest=True)

# Visualisation de la distribution des types de bien
plt.figure(figsize=(8,4))
sns.countplot(x="Type local", data=df, order=df["Type local"].value_counts().index)
plt.xticks(rotation=30)
plt.title("Répartition des types de locaux")
plt.tight_layout()
plt.show()
print(df["Type local"].value_counts())

# Visualisation surface_cat avant encoding dummies
plt.figure(figsize=(7,5))
sns.boxplot(x="surface_cat", y="Valeur fonciere", data=df)
plt.yscale("log")
plt.xlabel("Catégorie de surface")
plt.ylabel("Valeur foncière (€) (log)")
plt.title("Distribution du prix selon la catégorie de surface")
plt.tight_layout()
plt.show()

# Encodage des catégories et du type de local (dummies)
df = pd.get_dummies(df, columns=["surface_cat", "Type local"], drop_first=True)

features_base = [
    "Surface totale", "Nombre pieces principales", "Annee mutation", "habitable", "nb_dependances"
] + [f"surface_cat_{l}" for l in surface_labels[1:]] + \
    [col for col in df.columns if col.startswith("Type local_")]

scaler_clust = StandardScaler()
X_clust_scaled = scaler_clust.fit_transform(df[features_base].fillna(0))

# Méthode du coude pour choisir k
inertias = []
k_range = range(2, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_clust_scaled)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(8,4))
plt.plot(k_range, inertias, marker='o')
plt.xlabel('Nombre de clusters k')
plt.ylabel('Inertie (somme des distances au carré)')
plt.title("Méthode du coude pour choisir k optimal")
plt.show()

# Choix du nombre de clusters sur base du coude
k_final = 4
kmeans_final = KMeans(n_clusters=k_final, random_state=42)
df["Cluster"] = kmeans_final.fit_predict(X_clust_scaled)

# Modélisation RandomForest optimisée par cluster
results = []
for cluster_id in sorted(df["Cluster"].unique()):
    df_cluster = df[df["Cluster"] == cluster_id].copy()
    X = df_cluster[features_base]
    y = df_cluster["log_valeur"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [4, 6, 10]
    }
    rf = RandomForestRegressor(random_state=42)
    grid = GridSearchCV(rf, param_grid, cv=3)
    grid.fit(X_train_scaled, y_train)
    y_pred = grid.predict(X_test_scaled)
    MAE = mean_absolute_error(y_test, y_pred)
    RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
    R2 = r2_score(y_test, y_pred)
    print(f"\n===== Cluster {cluster_id} (RF optimisé) =====\nBest params: {grid.best_params_}\nMAE={MAE:.3f}, RMSE={RMSE:.3f}, R2={R2:.3f}")
    results.append({"Cluster": cluster_id, "Model": "RandomForest_Opt", "MAE": MAE, "RMSE": RMSE, "R2": R2})

results_df = pd.DataFrame(results)
plt.figure(figsize=(8, 5))
sns.barplot(data=results_df, x="Cluster", y="R2", palette="mako")
plt.title("Perf. R² RandomForest optimisé par cluster")
plt.tight_layout()
plt.show()

# Importance des variables (RandomForest, dernier cluster)
best_rf = grid.best_estimator_
importances = best_rf.feature_importances_
plt.figure(figsize=(10,4))
sns.barplot(x=features_base, y=importances)
plt.xticks(rotation=45)
plt.title("Importance des variables (RandomForest)")
plt.tight_layout()
plt.show()

for cluster_id in sorted(df["Cluster"].unique()):
    df_cluster = df[df["Cluster"] == cluster_id].copy()
    X = df_cluster[features_base]
    y = df_cluster["log_valeur"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [4, 6, 10]
    }

    rf = RandomForestRegressor(random_state=42)
    grid = GridSearchCV(rf, param_grid, cv=3)
    grid.fit(X_train_scaled, y_train)
    y_pred = grid.predict(X_test_scaled)
    MAE = mean_absolute_error(y_test, y_pred)
    RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
    R2 = r2_score(y_test, y_pred)
    print(f"\n===== Cluster {cluster_id} (RF optimisé) =====\nBest params: {grid.best_params_}\nMAE={MAE:.3f}, RMSE={RMSE:.3f}, R2={R2:.3f}")
    results.append({"Cluster": cluster_id, "Model": "RandomForest_Opt", "MAE": MAE, "RMSE": RMSE, "R2": R2})

    # AJOUT DU BLOC DE VISUALISATION DES ERREURS
    plt.figure(figsize=(7,5))
    plt.scatter(np.expm1(y_test), np.expm1(y_pred), alpha=0.3)
    plt.plot([np.expm1(y_test.min()), np.expm1(y_test.max())], [np.expm1(y_test.min()), np.expm1(y_test.max())], color='red', linestyle='--')
    plt.xlabel("Valeur réelle (€)")
    plt.ylabel("Valeur prédite (€)")
    plt.title(f"Cluster {cluster_id} – Réel vs Prédiction (RandomForest)")
    plt.tight_layout()
    plt.show()

