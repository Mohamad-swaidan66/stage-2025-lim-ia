from pathlib import Path
import requests
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TesseractCliOcrOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption


def convert_pdf(input_doc, output_file=None):
    """
    Convert a PDF document to Markdown format, enriched with OCR and structure analysis.
    """
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_code_enrichment = True
    pipeline_options.do_formula_enrichment = True
    pipeline_options.do_table_structure = True
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True
    pipeline_options.images_scale = 2.0
    pipeline_options.table_structure_options.do_cell_matching = True

    pipeline_options.do_ocr = True
    ocr_options = TesseractCliOcrOptions(force_full_page_ocr=True)
    pipeline_options.ocr_options = ocr_options

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    print(f" Conversion PDF : {input_doc}")
    doc = converter.convert(input_doc).document
    md = doc.export_to_markdown()

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(md)
    return md


def clean_repetitive_lines(md):
    """
    Supprime les lignes répétées trois fois ou plus à la suite (typiquement du bruit OCR).
    """
    lines = md.splitlines()
    cleaned = []
    last_line = None
    repeat_count = 0

    for line in lines:
        line = line.strip()

        if line == last_line:
            repeat_count += 1
            if repeat_count >= 2:
                continue  
        else:
            repeat_count = 0

        last_line = line
        cleaned.append(line)

    return "\n".join(cleaned)


def enrich_markdown_with_ollama(markdown_content, model="llama3:latest"):
    """
    Enrich raw Markdown using a local Ollama model like Mistral or Gemma.
    """
    prompt = f"""
Tu es un outil de post-traitement OCR. Ta mission est de convertir un texte brut issu d’un scan PDF ou photo en une version **propre, lisible et fidèle à l’original**, au format **Markdown**.

---

🎯 **Objectif principal** :
- Produire un contenu **corrigé**, **structuré** et **prêt à l’usage**.
- Respecter **scrupuleusement** la mise en forme d’origine, sans interprétation.

---

🛠️ **Règles de traitement** :
1. **Remplacer** tous les mots "argon" par **"arçon"**.
2. **Supprimer les numéros de ligne** (ex. : "45 Introduction").
3. **Corriger automatiquement** :
   - fautes de frappe,
   - erreurs OCR (lettres, ponctuation, casse, mots coupés),
   - doublons de mots ou de lignes.
4. **Recomposer les paragraphes** logiquement (regrouper les lignes).
5. **Restaurer la hiérarchie** :
   - Titres → `#`, `##`, etc.
   - Paragraphes → blocs Markdown cohérents.
   - Tableaux → en Markdown (ou `|` + `-` si non structurable).
6. **Préserver** :
   - les chiffres, unités, équations, citations,
   - les références scientifiques.
7. **Réorganiser** les blocs de texte dans un **ordre naturel de lecture** (haut → bas, gauche → droite).
8. Ne jamais ajouter de contenu ou de commentaire.
9. Si le PDF est issu d’une **photo** :
   - corriger les distorsions de forme,
   - reconstruire les blocs mal alignés,
   - convertir les éléments visuellement saillants (titres centrés, encadrés) en structures Markdown explicites.

---

📝 **Exemple de conversion** :

**OCR brut** :
45 Introduction  
46 Cette étude analyse l'impact de la position du cavalier.  
47 Le cavalier influence la biomécanique du cheval.

**Markdown attendu** :
# Introduction  
Cette étude analyse l'impact de la position du cavalier. Le cavalier influence la biomécanique du cheval.

---

🧾 **Texte à traiter** :  
Texte OCR :  
{markdown_content}

---

🔁 **Sortie attendue** :
Retourne **uniquement** le texte **corrigé** au format **Markdown propre**, sans aucun ajout, annotation ou explication.
"""





    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_ctx": 8192,
            }
        },
    )

    if response.status_code == 200:
        return response.json()["response"]
    else:
        raise RuntimeError(f"Ollama error: {response.status_code}\n{response.text}")


def convert_office_document(input_doc, output_file=None):
    """
    Convert an Office document (Word, etc.) to Markdown.
    """
    converter = DocumentConverter()
    doc = converter.convert(input_doc).document
    md = doc.export_to_markdown()

    if output_file:

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(md)
    return md

def structured_pdf_pipeline(input_pdf_path, output_md_path, model_name="llama3:latest"):
    print(f"📄 Conversion OCR : {input_pdf_path}")
    raw_md = convert_pdf(input_pdf_path)
    cleaned_md = clean_repetitive_lines(raw_md)
    enriched_md = enrich_markdown_with_ollama(cleaned_md, model=model_name)

    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write(enriched_md)
    print(f"✅ Fichier enrichi : {output_md_path}")
 

def process_and_enrich_markdown(md_raw, output_file, model="llama3:latest"):
    md_cleaned = clean_repetitive_lines(md_raw)
    md_enriched = enrich_markdown_with_ollama(md_cleaned, model=model)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(md_enriched)








if __name__ == "__main__":
    # PDF source
    #input_doc = Path(r"/var/www/RAG/Data/lilian_prompting.pdf")
    input_doc = Path(r"/var/www/RAG/Data/LIM FR/BIOMÉCA/THESE PAULINE/1605861571_2015 - Effect of the rider position during rising trot on the horse_s biomechanics _back and trunk kinematics and pressure under the saddle_ - Pauline Martin.pdf")
    # Fichier de sortie Markdown enrichi
    output_file = Path("/var/www/RAG/Data_parse/test.md")

    # Étape 1 : Conversion PDF → Markdown brut avec OCR complet
    raw_md = convert_pdf(input_doc)

    # Étape 2 : Enrichissement via Ollama 
    print(" Enrichissement via modèle local ...")
    enriched_md = enrich_markdown_with_ollama(raw_md, model="llama3:latest")

    # Étape 3 : Sauvegarde
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(enriched_md)

    print(f"✅ Markdown enrichi sauvegardé dans : {output_file}")
