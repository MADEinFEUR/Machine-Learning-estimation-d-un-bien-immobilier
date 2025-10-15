import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# Chargement du dataset
df = pd.read_csv(
    "C:/Users/sachi/Documents/CODE/machine_learning/test/ValeursFoncieres2024_06400.txt",
    sep="|",
    dtype=str
)

# Colonnes numériques
df["Code voie"] = pd.to_numeric(df["Code voie"], errors='coerce')
df["No plan"] = pd.to_numeric(df["No plan"], errors='coerce')
df["Surface reelle bati"] = pd.to_numeric(df["Surface reelle bati"].str.replace(",", "."), errors='coerce')
df["Valeur fonciere"] = pd.to_numeric(df["Valeur fonciere"].str.replace(",", "."), errors='coerce')

# Supprimer les lignes sans info essentielles
df = df.dropna(subset=["Code voie", "No plan", "Valeur fonciere", "Surface reelle bati"])

# Créer un indicateur de surface totale (bâti + Carrez)
surface_cols = ["Surface reelle bati", "Surface Carrez du 1er lot",
                "Surface Carrez du 2eme lot", "Surface Carrez du 3eme lot",
                "Surface Carrez du 4eme lot", "Surface Carrez du 5eme lot"]

for col in surface_cols[1:]:
    df[col] = pd.to_numeric(df[col].str.replace(",", "."), errors='coerce')

df["Surface totale"] = df[surface_cols].sum(axis=1, skipna=True)

# --- Plot 2D ---
plt.figure(figsize=(10, 8))

# Chaque lot comme un point
# X = Code voie, Y = No plan
# Taille = Surface totale, Couleur = Valeur fonciere
sc = plt.scatter(
    df["Code voie"],
    df["No plan"],
    s=df["Surface totale"],
    c=df["Valeur fonciere"],
    cmap="viridis",
    norm=mcolors.LogNorm(),  # <- échelle logarithmique
    alpha=0.6,
    edgecolor="k"
)

plt.xlabel("Code voie")
plt.ylabel("No plan cadastral")
plt.title("Map 2D des lots cadastraux (Nice, 06000)")
plt.show()

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np

# Conversion en numérique (remplacer les NaN par 0)
df["No plan"] = pd.to_numeric(df["No plan"], errors="coerce").fillna(0)
df["1er lot"] = pd.to_numeric(df["1er lot"], errors="coerce").fillna(0)

# Maintenant on peut calculer Y_pos
df["Y_pos"] = df["No plan"] + df["1er lot"] / 10


# Suppose df est déjà chargé et nettoyé
# Factorisation des sections et type de voie
df["Section_num"] = pd.factorize(df["Section"])[0]
df["Type_voie_num"] = pd.factorize(df["Type de voie"].fillna("RUE"))[0]

# Position Y approximative dans la rue
df["Y_pos"] = df["No plan"].fillna(0) + df["1er lot"].fillna(0)/10

plt.figure(figsize=(15,10))
sc = plt.scatter(
    df["Section_num"],
    df["Type_voie_num"] + df["Y_pos"]/100,  # fraction pour éviter chevauchement
    s=df["Surface totale"].fillna(10),
    c=df["Valeur fonciere"].fillna(1),
    cmap="viridis",
    norm=mcolors.LogNorm(),
    alpha=0.6,
    edgecolor="k"
)
plt.colorbar(sc, label="Valeur fonciere (€) [log scale]")
plt.xlabel("Section (factorisée)")
plt.ylabel("Type de voie + position dans la voie")
plt.title("Trieur 2D approximatif des biens par section et par rue")
plt.show()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Conversion des colonnes en numérique
df["No plan"] = pd.to_numeric(df["No plan"], errors="coerce").fillna(0)
df["1er lot"] = pd.to_numeric(df["1er lot"], errors="coerce").fillna(0)
df["Valeur fonciere"] = pd.to_numeric(df["Valeur fonciere"], errors="coerce").fillna(0)

# Axe X : sections
sections = df["Section"].unique()
section_map = {sec: i for i, sec in enumerate(sections)}
df["X_pos"] = df["Section"].map(section_map)

# Axe Y : numéro de plan + fraction du lot
df["Y_pos"] = df["No plan"] + df["1er lot"] / 10

# Couleur selon valeur foncière en log
valeurs_log = np.log1p(df["Valeur fonciere"])  # log(1 + valeur) pour éviter log(0)

plt.figure(figsize=(12, 6))
scatter = plt.scatter(
    df["X_pos"], df["Y_pos"],
    c=valeurs_log, cmap="viridis", alpha=0.7
)
plt.colorbar(scatter, label="Valeur foncière (log €)")

# Labels X avec noms de sections
plt.xticks(ticks=list(section_map.values()), labels=list(section_map.keys()), rotation=45)
plt.xlabel("Section cadastrale")
plt.ylabel("Position lot / plan")
plt.title("Carte 2D des biens par section et numéro de lot (log valeur foncière)")
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Copier df et filtrer les valeurs foncières
df_plot = df.dropna(subset=["Valeur fonciere"]).copy()
df_plot["Valeur_log"] = np.log1p(df_plot["Valeur fonciere"])

# Limiter aux 20 rues les plus fréquentes
top_rues = df_plot["Voie"].value_counts().nlargest(20).index
df_plot = df_plot[df_plot["Voie"].isin(top_rues)]

# Créer des positions numériques pour le type de bien
type_map = {t: i for i, t in enumerate(df_plot["Type local"].unique())}
df_plot["Type_local_num"] = df_plot["Type local"].map(type_map)

plt.figure(figsize=(14, 6))
scatter = plt.scatter(
    x=df_plot["Voie"].map({v: i for i, v in enumerate(top_rues)}),
    y=df_plot["Type_local_num"],
    c=df_plot["Valeur_log"],
    cmap="viridis",
    s=50,  # taille du point
    alpha=0.7
)

plt.colorbar(scatter, label="Valeur foncière (log €)")
plt.xticks(ticks=np.arange(len(top_rues)), labels=top_rues, rotation=45, ha="right")
plt.yticks(ticks=list(type_map.values()), labels=list(type_map.keys()))
plt.xlabel("Nom de la rue")
plt.ylabel("Type de bien")
plt.title("Valeur foncière par type de bien et rue (log)")
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Copier df et filtrer les valeurs foncières
df_plot = df.dropna(subset=["Valeur fonciere", "Surface totale"]).copy()
df_plot["Valeur_log"] = np.log1p(df_plot["Valeur fonciere"])

# Limiter aux 20 rues les plus fréquentes
top_rues = df_plot["Voie"].value_counts().nlargest(20).index
df_plot = df_plot[df_plot["Voie"].isin(top_rues)]

# Créer des positions numériques pour le type de bien
type_map = {t: i for i, t in enumerate(df_plot["Type local"].unique())}
df_plot["Type_local_num"] = df_plot["Type local"].map(type_map)

# Échelle pour la taille des points (surface)
surface_min = df_plot["Surface totale"].min()
surface_max = df_plot["Surface totale"].max()
s_min, s_max = 20, 200  # taille min/max des points
df_plot["Point_size"] = s_min + (df_plot["Surface totale"] - surface_min) / (surface_max - surface_min) * (s_max - s_min)

plt.figure(figsize=(20, 6))
scatter = plt.scatter(
    x=df_plot["Voie"].map({v: i for i, v in enumerate(top_rues)}),
    y=df_plot["Type_local_num"],
    c=df_plot["Valeur_log"],
    s=df_plot["Point_size"],
    cmap="viridis",
    alpha=0.7,
    edgecolors='w'
)

plt.colorbar(scatter, label="Valeur foncière (log €)")
plt.xticks(ticks=np.arange(len(top_rues)), labels=top_rues, rotation=45, ha="right")
plt.yticks(ticks=list(type_map.values()), labels=list(type_map.keys()))
plt.xlabel("Nom de la rue")
plt.ylabel("Type de bien")
plt.title("Valeur foncière et surface par type de bien et rue")
plt.tight_layout()
plt.show()

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Sélection des colonnes pour le clustering
cols = ["Surface totale", "Surface terrain", "Nombre pieces principales", "Valeur fonciere", "Code voie", "Section"]
df_cluster = df.dropna(subset=cols).copy()
df_cluster["Section_num"] = pd.factorize(df_cluster["Section"])[0]  # transformer section en numérique

X_cluster = df_cluster[["Surface totale", "Surface terrain", "Nombre pieces principales",
                        "Valeur fonciere", "Code voie", "Section_num"]]
type_local_map = {"Maison": 1, "Appartement": 2, "Dépendance": 3, "__NA__": 0}
df_cluster["Type_local_num"] = df_cluster["Type local"].map(type_local_map).fillna(0).astype(int)

# Standardisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# Clustering
kmeans = KMeans(n_clusters=5, random_state=42)
df_cluster["cluster"] = kmeans.fit_predict(X_scaled)

import matplotlib.pyplot as plt

# Assurer que la colonne existe
df_cluster["Type_local_num"] = df_cluster["Type local"].map(type_local_map).fillna(0).astype(int)

plt.figure(figsize=(12,6))
plt.scatter(
    x=df_cluster["Code voie"],
    y=df_cluster["Type_local_num"],
    s=df_cluster["Surface totale"],  # taille selon la surface
    c=np.log1p(df_cluster["Valeur fonciere"]),  # couleur selon log de la valeur
    cmap="viridis",
    alpha=0.7
)
plt.colorbar(label="Valeur foncière (log)")
plt.xlabel("Code voie")
plt.ylabel("Type de bien (num)")
plt.title("Biens par type et code voie, taille=surface, couleur=log(Valeur)")
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Log de la valeur foncière
df_plot['log_valeur'] = np.log1p(df_plot['Valeur fonciere'])

plt.figure(figsize=(12, 6))
sns.scatterplot(
    data=df_plot,
    x='Type de voie',
    y='Type local',
    size='Surface totale',   # taille des points
    hue='log_valeur',        # couleur par valeur foncière log
    palette='viridis',
    alpha=0.7,
    sizes=(20, 200)
)
plt.xticks(rotation=45)
plt.title("Distribution des biens par type de voie et type de bien")
plt.xlabel("Type de voie")
plt.ylabel("Type de bien")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

