import pandas as pd
from collections import defaultdict

# === Chemin du fichier Excel source ===
excel_path = "/var/www/RAG/Data/CWD FR/SELLES/CWD_Descriptifs produits_refonte_Fev23 ZBA.xlsx"

# === Lire la feuille 'Modele selle' ===
df_modele_selle = pd.read_excel(excel_path, sheet_name="Modele selle")

# === Nettoyage des colonnes et des lignes ===
df_modele_selle.columns = df_modele_selle.columns.str.strip()
df_modele_selle.dropna(how='all', inplace=True)

# === Structure hiérarchique : Type > Modèle > Descriptif + Caractéristiques ===
data_by_type_model = defaultdict(lambda: defaultdict(lambda: {'description': '', 'features': []}))
current_type = None
current_model = None

for _, row in df_modele_selle.iterrows():
    if pd.notna(row['Type']):
        current_type = row['Type']
    if pd.notna(row['Modèle']):
        current_model = row['Modèle']
    if current_type and current_model:
        if pd.notna(row['Descriptif web actuel']):
            data_by_type_model[current_type][current_model]['description'] = row['Descriptif web actuel']
        if pd.notna(row['Caractéristiques']) and pd.notna(row['Définition des caractéristiques']):
            data_by_type_model[current_type][current_model]['features'].append(
                (row['Caractéristiques'].strip(), row['Définition des caractéristiques'].strip())
            )

# === Génération du contenu Markdown ===
markdown_lines = ["# modeleselle.md\n"]

for type_name, models in data_by_type_model.items():
    markdown_lines.append(f"## {type_name}")
    markdown_lines.append("### Modèle")
    for model_name, details in models.items():
        markdown_lines.append(f"#### {model_name}")
        markdown_lines.append("##### Descriptif web actuel")
        markdown_lines.append(details['description'] + "\n")
        markdown_lines.append("##### Caractéristique")
        for characteristic, definition in details['features']:
            markdown_lines.append(f"● {characteristic} : \n")
            markdown_lines.append(f"Définition : {definition}\n")

# === Sauvegarde dans un fichier Markdown ===
output_path = "/var/www/RAG/Data_parse/exel.md"
with open(output_path, "w", encoding="utf-8") as f:
    f.write("\n".join(markdown_lines))

print(f"Fichier généré : {output_path}")
