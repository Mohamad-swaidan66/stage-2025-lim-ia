import pandas as pd

# Charger le fichier Excel
excel_path = "/var/www/RAG/Data/CWD FR/SELLES/CWD - Protocole Sellier_Compte rendu client.xlsx"

# Lire la feuille "Finitions" sans utiliser la première ligne comme entête
df_finitions = pd.read_excel(excel_path, sheet_name="Finitions", header=None)

# Supprimer les lignes vides
df_finitions.dropna(how='all', inplace=True)

# Construire le contenu Markdown
markdown_lines = ["# Finitions.md\n"]

for _, row in df_finitions.iterrows():
    # Extraire l'abréviation à partir de la 1ère colonne (ex: "1. GV" → "GV")
    abbreviation = str(row[0]).split()[-1].strip()
    designation = str(row[1]).strip()
    caracteristique = str(row[2]).strip() if pd.notna(row[2]) else "Non spécifiée"

    # Écrire les lignes Markdown formatées
    markdown_lines.append(f"## {designation}")
    markdown_lines.append(f"- abbreviation : {abbreviation}")
    markdown_lines.append(f"- caracteristique : {caracteristique}\n")

# Sauvegarder le fichier Markdown
output_path = "/var/www/RAG/Data_parse/exel3.md"
with open(output_path, "w", encoding="utf-8") as f:
    f.write("\n".join(markdown_lines))

print(f"Fichier généré : {output_path}")
