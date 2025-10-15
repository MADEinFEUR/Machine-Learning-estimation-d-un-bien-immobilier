import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set_style('darkgrid')

# --- Chargement et nettoyage des données ---
df = pd.read_csv(
    "C:/Users/sachi/Documents/CODE/machine_learning/test/ValeursFoncieres2024_06000.txt",
    sep="|", dtype=str
)

# Colonnes numériques
num_cols = [
    "Valeur fonciere", "Surface reelle bati", "Surface Carrez du 1er lot",
    "Surface Carrez du 2eme lot", "Surface Carrez du 3eme lot",
    "Surface Carrez du 4eme lot", "Surface Carrez du 5eme lot",
    "Nombre de lots", "Nombre pieces principales", "Surface terrain"
]

for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c].str.replace(" ", "").str.replace(",", "."), errors='coerce')

# Date et année
df["Date mutation"] = pd.to_datetime(df["Date mutation"], format="%d/%m/%Y", errors="coerce")
df["Annee mutation"] = df["Date mutation"].dt.year

# Surface totale
surface_cols = [
    "Surface reelle bati", "Surface Carrez du 1er lot", "Surface Carrez du 2eme lot",
    "Surface Carrez du 3eme lot", "Surface Carrez du 4eme lot", "Surface Carrez du 5eme lot"
]
df["Surface totale"] = df[surface_cols].sum(axis=1, skipna=True)

# Imputation sans warnings
for c in ["Nombre pieces principales", "Surface totale", "Surface reelle bati", "Surface terrain"]:
    if c in df.columns:
        df[c] = df[c].fillna(df[c].median())

for c in ["Code postal", "Type local", "Commune"]:
    df[c] = df[c].fillna('__NA__')

# Suppression des valeurs aberrantes
q_low_val = df['Valeur fonciere'].quantile(0.05)
q_high_val = df['Valeur fonciere'].quantile(0.95)
df = df[(df['Valeur fonciere'] >= q_low_val) & (df['Valeur fonciere'] <= q_high_val)]

q_low_surf = df['Surface totale'].quantile(0.05)
q_high_surf = df['Surface totale'].quantile(0.95)
df = df[(df['Surface totale'] >= q_low_surf) & (df['Surface totale'] <= q_high_surf)]

# Log transformation de la cible
df['log_valeur'] = np.log1p(df['Valeur fonciere'])

# --- Préparation des features ---
features = ["Surface totale", "Nombre pieces principales", "Type local", "Annee mutation"]
X = df[features].copy()
y = df['log_valeur']

# One-hot encoding pour les colonnes catégorielles
X = pd.get_dummies(X, columns=["Type local"], drop_first=True)

# --- Train/Test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Modèle Linéaire ---
linear_reg = LinearRegression()
linear_reg.fit(X_train_scaled, y_train)

# Évaluation
def model_evaluation(model, X_test, y_test, model_name="Model"):
    y_pred = model.predict(X_test)
    MAE = mean_absolute_error(y_test, y_pred)
    MSE = mean_squared_error(y_test, y_pred)
    RMSE = np.sqrt(MSE)
    R2 = r2_score(y_test, y_pred)
    print(f"\n{model_name} Evaluation:")
    print(f"MAE: {MAE:.2f}, RMSE: {RMSE:.2f}, R2: {R2:.3f}")
    return y_pred

y_pred_linear = model_evaluation(linear_reg, X_test_scaled, y_test, "Linear Regression")

# --- Fonction pour prédire la valeur d'un nouveau bien ---
def predict_bien(surface_totale, nb_pieces, type_local, annee_mutation):
    new_data = pd.DataFrame({
        "Surface totale": [surface_totale],
        "Nombre pieces principales": [nb_pieces],
        "Annee mutation": [annee_mutation],
        "Type local": [type_local]
    })
    new_data = pd.get_dummies(new_data, columns=["Type local"], drop_first=True)

    # Ajouter les colonnes manquantes si elles n'existent pas dans le jeu de données
    for col in X.columns:
        if col not in new_data.columns:
            new_data[col] = 0

    # Réordonner les colonnes pour correspondre au modèle
    new_data = new_data[X.columns]

    # Standardisation
    new_scaled = scaler.transform(new_data)

    # Prédiction
    log_pred = linear_reg.predict(new_scaled)[0]
    valeur_pred = np.expm1(log_pred)  # Inverse du log1p
    return valeur_pred

# Exemple :
valeur_estimee = predict_bien(surface_totale=100, nb_pieces=4, type_local='Maison', annee_mutation=2024)
print(valeur_estimee)
