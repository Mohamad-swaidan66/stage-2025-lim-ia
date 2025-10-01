import pandas as pd

# Charger le fichier Excel
excel_path = "/var/www/RAG/Data/CWD FR/SELLES/CWD - Protocole Sellier_Compte rendu client.xlsx"

# Lire la feuille "Config panneaux" sans considérer la première ligne comme en-tête
df_config = pd.read_excel(excel_path, sheet_name="Config panneaux", header=None)

# Supprimer les lignes vides
df_config.dropna(how='all', inplace=True)

# Donner des noms explicites aux colonnes
df_config.columns = ['Code', 'Définition']

# Construire le contenu Markdown
markdown_lines = ["# config panneaux.md\n"]

for index, row in df_config.iterrows():
    code = str(row['Code']).strip()
    definition = str(row['Définition']).strip()
    markdown_lines.append(f"## {code}")
    markdown_lines.append(f"Définition : {definition}\n")

# Sauvegarder dans un fichier .md
output_path = "/var/www/RAG/Data_parse/exel2.md"
with open(output_path, "w", encoding="utf-8") as f:
    f.write("\n".join(markdown_lines))

print(f"Fichier généré : {output_path}")
