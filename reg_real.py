import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Chargement et nettoyage des données ---
df = pd.read_csv(
    "C:/Users/sachi/Documents/CODE/machine_learning/test/ValeursFoncieres2024_06000.txt",
    sep="|",
    dtype=str
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

# Surface totale (bâti + Carrez)
surface_cols = [
    "Surface reelle bati", "Surface Carrez du 1er lot", "Surface Carrez du 2eme lot",
    "Surface Carrez du 3eme lot", "Surface Carrez du 4eme lot", "Surface Carrez du 5eme lot"
]
df["Surface totale"] = df[surface_cols].sum(axis=1, skipna=True)

# Imputation des valeurs manquantes
df["Nombre pieces principales"] = df["Nombre pieces principales"].fillna(df["Nombre pieces principales"].median())
df["Surface totale"] = df["Surface totale"].fillna(df["Surface totale"].median())
df["Surface reelle bati"] = df["Surface reelle bati"].fillna(df["Surface reelle bati"].median())
df["Surface terrain"] = df["Surface terrain"].fillna(df["Surface terrain"].median())
df["Code postal"] = df["Code postal"].fillna("__NA__")
df["Type local"] = df["Type local"].fillna("__NA__")
df["Commune"] = df["Commune"].fillna("__NA__")

# Suppression des valeurs aberrantes
q_low_val = df['Valeur fonciere'].quantile(0.00)
q_high_val = df['Valeur fonciere'].quantile(0.95)
df = df[(df['Valeur fonciere'] >= q_low_val) & (df['Valeur fonciere'] <= q_high_val)]

q_low_surf = df['Surface totale'].quantile(0.00)
q_high_surf = df['Surface totale'].quantile(0.95)
df = df[(df['Surface totale'] >= q_low_surf) & (df['Surface totale'] <= q_high_surf)]

# Log transformation de la cible
df['log_valeur'] = np.log1p(df['Valeur fonciere'])

# --- Création de la colonne "habitable" ---
df['habitable'] = ((df['Surface totale'] > 0) & (df['Type local'].isin(['Maison', 'Appartement']))).astype(int)

# --- Création de l'ID de vente et du nombre de dépendances ---
# Les ventes sont groupées par Identifiant de document + Date mutation
df['id_vente'] = df.groupby(['Identifiant de document', 'Date mutation']).ngroup()

# Compter le nombre de dépendances par vente
dependances = df[df['Type local'] == 'Dépendance'].groupby('id_vente').size().rename('nb_dependances')
df = df.merge(dependances, on='id_vente', how='left')
df['nb_dependances'] = df['nb_dependances'].fillna(0).astype(int)

# --- Préparation des features ---
features = ["Surface totale", "Surface terrain", "Nombre pieces principales", "habitable", "nb_dependances", "Type local", "Annee mutation"]
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

# --- Évaluation du modèle ---
y_pred = linear_reg.predict(X_test_scaled)
MAE = mean_absolute_error(y_test, y_pred)
MSE = mean_squared_error(y_test, y_pred)
RMSE = np.sqrt(MSE)
R2 = r2_score(y_test, y_pred)

print(f"Linear Regression Evaluation:\nMAE: {MAE:.2f}, RMSE: {RMSE:.2f}, R2: {R2:.3f}")

# --- Fonction pour prédire un nouveau bien ---
def predict_property(surface_totale, surface_terrain, nb_pieces, habitable, nb_dependances, type_local, annee,
                     model=linear_reg, scaler=scaler, X_columns=X.columns):
    new_data = pd.DataFrame({
        "Surface totale": [surface_totale],
        "Surface terrain": [surface_terrain],
        "Nombre pieces principales": [nb_pieces],
        "habitable": [habitable],
        "nb_dependances": [nb_dependances],
        "Type local": [type_local],
        "Annee mutation": [annee]
    })
    # One-hot encoding
    new_data = pd.get_dummies(new_data, columns=["Type local"], drop_first=True)

    # Ajouter les colonnes manquantes si nécessaire
    for col in X_columns:
        if col not in new_data.columns:
            new_data[col] = 0
    new_data = new_data[X_columns]  # Réordonner

    # Standardisation
    new_scaled = scaler.transform(new_data)

    # Prédiction
    log_val = model.predict(new_scaled)[0]
    return np.expm1(log_val)  # Retourne la valeur réelle

# --- Exemple de prédiction automatique ---
valeur_predite = predict_property(
    surface_totale=72,
    surface_terrain=50,
    nb_pieces=4,
    habitable=1,
    nb_dependances=2,  # 2 dépendances vendues avec
    type_local="Appartement",
    annee=2024
)
print(f"Valeur prédite du bien (exemple): {valeur_predite:,.2f} €")

# --- Mode interactif utilisateur ---
print("\n=== Estimation de votre bien immobilier ===")
print("ℹ️  Cette estimation est basée sur les données foncières du code postal 06000 (Nice et alentours).")
print("Les résultats peuvent être moins fiables pour d'autres zones géographiques.\n")

try:
    surface_totale = float(input("Surface totale habitable (m²) : ").replace(",", "."))
    surface_terrain = float(input("Surface du terrain (m², 0 si appartement) : ").replace(",", "."))
    nb_pieces = int(input("Nombre de pièces principales : "))

    type_local = input("Type de bien (Maison / Appartement) : ").capitalize()
    while type_local not in ["Maison", "Appartement"]:
        print("❌ Type invalide. Choisissez 'Maison' ou 'Appartement'.")
        type_local = input("Type de bien (Maison / Appartement) : ").capitalize()

    nb_dependances = int(input("Nombre de dépendances (garage, cave, etc.) : "))
    annee = int(input("Année de la transaction (ex: 2024) : "))

    habitable = 1 if type_local in ["Maison", "Appartement"] else 0

    estimation = predict_property(
        surface_totale=surface_totale,
        surface_terrain=surface_terrain,
        nb_pieces=nb_pieces,
        habitable=habitable,
        nb_dependances=nb_dependances,
        type_local=type_local,
        annee=annee
    )

    print("\n📍 Estimation basée sur le modèle entraîné avec les ventes du secteur 06000 (Nice).")
    print(f"💰 Valeur foncière estimée : {estimation:,.2f} €")

except Exception as e:
    print(f"❌ Une erreur est survenue lors de la saisie : {e}")
