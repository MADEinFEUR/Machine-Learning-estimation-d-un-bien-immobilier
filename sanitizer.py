import pandas as pd

# Lire le fichier (séparateur = '|')
df = pd.read_csv("C:/Users/sachi/Documents/CODE/machine_learning/test/ValeursFoncieres2024.txt", sep="|", dtype=str)

# Filtrer le code postal
df_06000 = df[df["Code postal"] == "6400"]

# Sauvegarder dans un nouveau fichier
df_06000.to_csv("C:/Users/sachi/Documents/CODE/machine_learning/test/ValeursFoncieres2024_06400.txt", sep="|", index=False)
##

import pandas as pd

input_file = "C:/Users/sachi/Documents/CODE/machine_learning/test/ValeursFoncieres2024_06400.txt"

count = 0
chunksize = 100000  # lecture par morceaux pour gros fichiers

for chunk in pd.read_csv(input_file, sep="|", dtype=str, chunksize=chunksize):
    # Nettoie la colonne et compare
    matches = chunk["Code postal"].astype(str).str.strip().isin(["06400"])
    count += matches.sum()

print(f"📊 Nombre total de lignes avec le code postal 06000 : {count}")




