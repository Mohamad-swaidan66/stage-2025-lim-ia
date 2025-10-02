from pathlib import Path
from PIL import Image
import pytesseract
import re

# =========================================================
# TABLE DE CORRESPONDANCE : codes matière -> description
# ---------------------------------------------------------
# Utilisée pour préfixer le Markdown d’une ligne "Matière : ..."
# en détectant le code dans le nom de fichier (stem du chemin).
# =========================================================
# =========================================================
# FONCTION PRINCIPALE : OCR image -> Markdown structuré
# ---------------------------------------------------------
# - Ouvre l'image
# - OCR via Tesseract (fra+eng)
# - Nettoyage & corrections OCR
# - Mise en forme Markdown par blocs / paragraphes
# - Ajoute (si trouvés) :
#     * la matière (via code dans le nom de fichier, ex. 'GV')
#     * la référence de selle (ex. 'SE123' si '123' apparaît dans le nom)
# - Écrit le Markdown sur disque si output_file fourni
# - Retourne la chaîne Markdown finale
# =========================================================
def image_to_markdown_paragraphs(input_image_path, output_file=None):
    print(f"🖼️ Traitement de l'image : {input_image_path}")
    try:
        img = Image.open(input_image_path)
    except Exception as e:
        print(f"❌ Erreur ouverture image : {e}")
        return ""

    print("🔍 Extraction du texte avec OCR...")
    # --psm 3 : mode "Fully automatic page segmentation"
    # --oem 3 : moteur LSTM + Legacy (auto)
    # -l fra+eng : langues française + anglaise
    custom_config = r'--psm 3 --oem 3 -l fra+eng'
    raw_text = pytesseract.image_to_string(img, config=custom_config)

    print("🧹 Nettoyage OCR...")
    cleaned_text = clean_and_correct_ocr_text(raw_text)

    print("📦 Formatage Markdown structuré...")
    markdown_output = blocs_vers_markdown_par_paragraphe(cleaned_text)

    # === Ajout de la description matière si code trouvé dans le nom de fichier ===
    code_matiere = None
    for code in sorted(CODES_MATIERES.keys(), key=len, reverse=True):
        if code in Path(input_image_path).stem:
            code_matiere = code
            break
    if code_matiere:
        description_matiere = CODES_MATIERES[code_matiere]
        markdown_output = f"**Matière : {description_matiere}**\n\n" + markdown_output

    # === Détection de la référence de selle (ex: SE01) via 3 chiffres dans le nom ===
    numero_selle_match = re.search(r'(\d{3})', Path(input_image_path).stem)
    if numero_selle_match:
        numero_selle = numero_selle_match.group(1)
        nom_selle = f"SE{numero_selle}"
        markdown_output = f"**Selle : {nom_selle}**\n" + markdown_output

    # === Écriture optionnelle sur disque ===
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(markdown_output)
        print(f"✅ Markdown sauvegardé dans : {output_file}")

    return markdown_output

# =========================================================
# NETTOYAGE & CORRECTIONS OCR
# ---------------------------------------------------------
# - Trim lignes vides / doublons exacts successifs
# - Raccorde les césures en fin de ligne (…- + suite)
# - Corrections ciblées (accents, termes techniques, toponymes)
# =========================================================
def clean_and_correct_ocr_text(text):
    """Nettoie les lignes et corrige les erreurs fréquentes OCR."""
    lines = text.splitlines()
    cleaned = []

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        # Filtre doublons stricts (même ligne répétée)
        if i > 0 and line == lines[i - 1].strip():
            continue
        # Recolle les césures (mots coupés en fin de ligne)
        if line.endswith('-') and cleaned:
            cleaned[-1] = cleaned[-1].rstrip('-') + line[:-1]
        else:
            cleaned.append(line)

    cleaned_text = "\n".join(cleaned)

    # Corrections lexicales/typographiques fréquentes
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

# =========================================================
# DÉTECTION DES BLOCS & MISE EN FORME MARKDOWN
# ---------------------------------------------------------
# blocs_vers_markdown_par_paragraphe :
# - Heuristique pour repérer des titres courts en Majuscules
# - Scission d’une ligne contenant plusieurs titres collés
# - Construit des sections "## Titre" suivies du contenu
# =========================================================
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
        # Heuristique : peu de mots, majorité de mots commençant par une majuscule,
        # absence de ponctuation forte (.,,:)
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
        (liste de mots-clés métier indicative)
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

    # Parcours séquentiel avec détection titres / contenu
    for ligne in lignes:
        lignes_a_traiter = scinder_ligne_multi_titres(ligne) if est_titre(ligne) else [ligne]

        for sous_ligne in lignes_a_traiter:
            if est_titre(sous_ligne):
                # Si on a un bloc en cours, on le flush
                if titre_actuel or contenu_actuel:
                    markdown += f"## {titre_actuel.strip()}\n{contenu_actuel.strip()}\n\n"
                    contenu_actuel = ""
                titre_actuel = sous_ligne
            else:
                contenu_actuel += " " + sous_ligne

    # Flush du dernier bloc si nécessaire
    if titre_actuel or contenu_actuel:
        markdown += f"## {titre_actuel.strip()}\n{contenu_actuel.strip()}\n\n"

    return markdown.strip()

# =========================================================
# (OPTION) TRAITEMENT D’UN DOSSIER ENTIER D’IMAGES
# ---------------------------------------------------------
# NOTE : le bloc ci-dessous est conservé tel quel, commenté.
# - Parcourt un dossier d’images
# - Produit un .md par image dans le dossier de sortie
# - Les chemins sont actuellement "en dur" (tels quels)
# =========================================================
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

# =========================================================
# POINT D’ENTRÉE (test unitaire simple)
# ---------------------------------------------------------
# - Spécifie un chemin d'image de test
# - Écrit le Markdown dans un fichier de sortie si 'output_file' fourni
# - Affiche le Markdown en console
# =========================================================
if __name__ == "__main__":
    image_test = "/var/www/RAG/Data/image.png"  # <-- mets ici le chemin de l'image que tu veux tester
    output_path = Path("/var/www/RAG/Data_parse/markdown_outputvar")
    
    markdown_resultat = image_to_markdown_paragraphs(image_test, output_file=output_path)
    
    print("\n📄 Résultat Markdown :\n")
    print(markdown_resultat)
