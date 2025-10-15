import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Chargement du dataset ---
df = pd.read_csv(
    "C:/Users/sachi/Documents/CODE/machine_learning/test/ValeursFoncieres2024_06000.txt",
    sep="|",
    dtype=str
)

# --- Colonnes numériques ---
num_cols = [
    "Valeur fonciere", "Surface reelle bati", "Surface Carrez du 1er lot",
    "Surface Carrez du 2eme lot", "Surface Carrez du 3eme lot",
    "Surface Carrez du 4eme lot", "Surface Carrez du 5eme lot",
    "Nombre de lots", "Nombre pieces principales", "Surface terrain",
    "Code voie", "Code postal"
]

for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c].str.replace(" ", "").str.replace(",", "."), errors='coerce')

# --- Date et année ---
df["Date mutation"] = pd.to_datetime(df["Date mutation"], format="%d/%m/%Y", errors="coerce")
df["Annee mutation"] = df["Date mutation"].dt.year

# --- Création de l'identifiant de vente et du lot unique ---
df['id_vente'] = df.groupby(['Identifiant de document', 'Date mutation']).ngroup()
df['lot_unique'] = df.apply(
    lambda x: f"{x['id_vente']}_{x['Section']}_{x['No plan']}_{x.get('1er lot',0)}", axis=1
)

# --- Surface totale par lot (bâti + Carrez) ---
surface_cols = [
    "Surface reelle bati", "Surface Carrez du 1er lot", "Surface Carrez du 2eme lot",
    "Surface Carrez du 3eme lot", "Surface Carrez du 4eme lot", "Surface Carrez du 5eme lot"
]
df["Surface totale"] = df[surface_cols].sum(axis=1, skipna=True)

# --- Imputation ---
df["Nombre pieces principales"] = df["Nombre pieces principales"].fillna(df["Nombre pieces principales"].median())
df["Surface totale"] = df["Surface totale"].fillna(df["Surface totale"].median())
df["Surface reelle bati"] = df["Surface reelle bati"].fillna(df["Surface reelle bati"].median())
df["Surface terrain"] = df["Surface terrain"].fillna(df["Surface terrain"].median())
df["Code postal"] = df["Code postal"].fillna(0).astype(int)
df["Code voie"] = df["Code voie"].fillna(0).astype(int)
df["Type local"] = df["Type local"].fillna("__NA__")
df["Commune"] = df["Commune"].fillna("__NA__")

# --- Comptage des dépendances par lot ---
dependances = df[df['Type local'] == 'Dépendance'].groupby('id_vente').size().rename('nb_dependances')
df = df.merge(dependances, on='id_vente', how='left')
df['nb_dependances'] = df['nb_dependances'].fillna(0).astype(int)

# --- Encodage numérique Type local ---
type_local_map = {"Maison": 1, "Appartement": 2, "Dépendance": 3, "__NA__": 0}
df["Type_local_num"] = df["Type local"].map(type_local_map).fillna(0).astype(int)

# --- On ne filtre plus sur les appartements ---
df_lots = df.copy()  # tous les lots (Appartement, Maison, Dépendance...)

# --- Supprimer les lignes sans valeur foncière ---
df_lots = df_lots.dropna(subset=["Valeur fonciere"]).copy()

# --- Features pour le modèle ---
features = [
    "Surface totale",
    "Surface terrain",
    "Nombre pieces principales",
    "nb_dependances",
    "Annee mutation",
    "Code postal",
    "Code voie",
    "Type_local_num"
]

X = df_lots[features]
y = np.log1p(df_lots["Valeur fonciere"])  # log transformation

# --- Train/Test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- Standardisation ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Modèle linéaire ---
linear_reg = LinearRegression()
linear_reg.fit(X_train_scaled, y_train)

# --- Évaluation ---
y_pred = linear_reg.predict(X_test_scaled)
MAE = mean_absolute_error(y_test, y_pred)
RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
R2 = r2_score(y_test, y_pred)

print(f"Linear Regression Evaluation (tous types de biens):\nMAE: {MAE:.2f}, RMSE: {RMSE:.2f}, R2: {R2:.3f}")

# --- Fonction de prédiction ---
def predict_property(surface_totale, surface_terrain, nb_pieces, nb_dependances,
                     type_local, annee, code_postal, code_voie,
                     model=linear_reg, scaler=scaler):

    type_local_num = type_local_map.get(type_local, 0)

    new_data = pd.DataFrame({
        "Surface totale": [surface_totale],
        "Surface terrain": [surface_terrain],
        "Nombre pieces principales": [nb_pieces],
        "nb_dependances": [nb_dependances],
        "Annee mutation": [annee],
        "Code postal": [int(code_postal)],
        "Code voie": [int(code_voie)],
        "Type_local_num": [type_local_num]
    })

    new_scaled = scaler.transform(new_data)
    log_val = model.predict(new_scaled)[0]
    return np.expm1(log_val)

# --- Exemple de prédiction automatique ---
valeur_predite = predict_property(
    surface_totale=77,
    surface_terrain=0,
    nb_pieces=4,
    nb_dependances=2,
    type_local="Appartement",
    annee=2024,
    code_postal=6000,
    code_voie=3800
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
    code_postal = int(input("Code postal : "))
    code_voie = int(input("Code voie : "))

    estimation = predict_property(
        surface_totale=surface_totale,
        surface_terrain=surface_terrain,
        nb_pieces=nb_pieces,
        nb_dependances=nb_dependances,
        type_local=type_local,
        annee=annee,
        code_postal=code_postal,
        code_voie=code_voie
    )

    print("\n📍 Estimation basée sur le modèle entraîné avec les ventes du secteur 06000 (Nice).")
    print(f"💰 Valeur foncière estimée : {estimation:,.2f} €")

except Exception as e:
    print(f"❌ Une erreur est survenue lors de la saisie : {e}")
