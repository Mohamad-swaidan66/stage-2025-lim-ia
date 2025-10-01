from pathlib import Path
from PIL import Image
import pytesseract
import re



# Correspondances code → description
CODES_MATIERES = {
    "GV": "grainé veau",
    "DV": "doublé veau",
    "DGV": "doublé grainé veau",
    "GB": "grainé buffle",
    "DB": "doublé buffle",
    "DGB": "doublé grainé buffle"
}  


def image_to_markdown_paragraphs(input_image_path, output_file=None):
    print(f"🖼️ Traitement de l'image : {input_image_path}")
    try:
        img = Image.open(input_image_path)
    except Exception as e:
        print(f"❌ Erreur ouverture image : {e}")
        return ""

    print("🔍 Extraction du texte avec OCR...")
    custom_config = r'--psm 3 --oem 3 -l fra+eng'
    raw_text = pytesseract.image_to_string(img, config=custom_config)

    print("🧹 Nettoyage OCR...")
    cleaned_text = clean_and_correct_ocr_text(raw_text)

    print("📦 Formatage Markdown structuré...")
    markdown_output = blocs_vers_markdown_par_paragraphe(cleaned_text)

    # Ajout de la description matière si code trouvé
    code_matiere = None
    for code in sorted(CODES_MATIERES.keys(), key=len, reverse=True):
        if code in Path(input_image_path).stem:
            code_matiere = code
            break
    if code_matiere:
        description_matiere = CODES_MATIERES[code_matiere]
        markdown_output = f"**Matière : {description_matiere}**\n\n" + markdown_output


    # Détection de la référence de selle (ex: SE01)
    numero_selle_match = re.search(r'(\d{3})', Path(input_image_path).stem)
    if numero_selle_match:
        numero_selle = numero_selle_match.group(1)
        nom_selle = f"SE{numero_selle}"
        markdown_output = f"**Selle : {nom_selle}**\n" + markdown_output

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(markdown_output)
        print(f"✅ Markdown sauvegardé dans : {output_file}")

    return markdown_output

def clean_and_correct_ocr_text(text):
    """Nettoie les lignes et corrige les erreurs fréquentes OCR."""
    lines = text.splitlines()
    cleaned = []

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        if i > 0 and line == lines[i - 1].strip():
            continue
        if line.endswith('-') and cleaned:
            cleaned[-1] = cleaned[-1].rstrip('-') + line[:-1]
        else:
            cleaned.append(line)

    cleaned_text = "\n".join(cleaned)

    corrections = {
        r'Arcon': 'Arçon',
        r'si[eé]ge': 'Siège',
        r'Panneaux\s+Int[ée]gr[ée]s?': 'Panneaux Intégrés',
        r'Sanglage\s+3\s+Points': 'Sanglage 3 Points',
        r'Close\s+Contact': 'Close Contact',
        r'Enfourchure\s+Large': 'Enfourchure Large'
    }

    for pattern, replacement in corrections.items():
        cleaned_text = re.sub(pattern, replacement, cleaned_text, flags=re.IGNORECASE)

    return cleaned_text



def blocs_vers_markdown_par_paragraphe(text):
    """
    Détecte automatiquement les blocs de type :
    ## Titre
    Contenu (même vide), et sépare plusieurs titres sur une même ligne.
    """
    lignes = [l.strip() for l in text.splitlines() if l.strip()]
    markdown = ""
    titre_actuel = ""
    contenu_actuel = ""

    def est_titre(ligne):
        mots = ligne.split()
        nb_maj = sum(1 for mot in mots if mot[:1].isupper())
        return (
            len(mots) <= 5 and
            nb_maj >= 2 and
            not any(p in ligne for p in [".", ",", ":"])
        )

    def scinder_ligne_multi_titres(ligne):
        """
        Scinde une ligne contenant plusieurs titres collés en plusieurs titres distincts.
        """
        mots_cles = ["Arçon", "Siège", "Panneaux", "Enfourchure", "Close", "Mono", "Petits", "Sanglage"]
        mots = ligne.split()
        blocs = []
        bloc = []

        for mot in mots:
            if mot in mots_cles and bloc:
                blocs.append(" ".join(bloc))
                bloc = [mot]
            else:
                bloc.append(mot)

        if bloc:
            blocs.append(" ".join(bloc))

        return blocs

    for ligne in lignes:
        lignes_a_traiter = scinder_ligne_multi_titres(ligne) if est_titre(ligne) else [ligne]

        for sous_ligne in lignes_a_traiter:
            if est_titre(sous_ligne):
                if titre_actuel or contenu_actuel:
                    markdown += f"## {titre_actuel.strip()}\n{contenu_actuel.strip()}\n\n"
                    contenu_actuel = ""
                titre_actuel = sous_ligne
            else:
                contenu_actuel += " " + sous_ligne

    if titre_actuel or contenu_actuel:
        markdown += f"## {titre_actuel.strip()}\n{contenu_actuel.strip()}\n\n"

    return markdown.strip()
    
#def traiter_images_dossier(input_folder, output_folder):
    """
    Traite toutes les images d'un dossier et génère un fichier Markdown pour chacune.
    """
    input_folder = Path("/var/www/RAG/Data/Descriptif Selles raccourci/Descriptif Selles raccourci")
    output_folder = Path("/var/www/RAG/Data_parse/markdown_outputvar/www/RAG/Data_parse/Descriptif Selles raccourci.md")
    output_folder.mkdir(parents=True, exist_ok=True)

    extensions = (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif")

    for image_path in sorted(input_folder.glob("*")):
        if image_path.suffix.lower() in extensions:
            output_md_path = output_folder / (image_path.stem + ".md")
            print(f"\n🔁 Traitement du fichier : {image_path.name}")
            image_to_markdown_paragraphs(image_path, output_file=output_md_path)

if __name__ == "__main__":
    image_test = "/var/www/RAG/Data/image.png"  # <-- mets ici le chemin de l'image que tu veux tester
    output_path = Path("/var/www/RAG/Data_parse/markdown_outputvar")
    
    markdown_resultat = image_to_markdown_paragraphs(image_test, output_file=output_path)
    
    print("\n📄 Résultat Markdown :\n")
    print(markdown_resultat)



