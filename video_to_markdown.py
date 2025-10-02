# =========================================================
# Imports
# ---------------------------------------------------------
# whisper      : modèle de transcription audio/vidéo
# requests     : appel HTTP à un LLM local (Ollama)
# os, tempfile : gestion de fichiers temporaires et chemins
# Path         : manipulation de chemins (lecture/écriture)
# pydub        : conversion MP3 -> WAV
# moviepy      : extraction piste audio d'une vidéo
# time         : temporisation (ex. boucle alternatives commentées)
# =========================================================
import whisper
import requests
import os
import tempfile
from pathlib import Path
from pydub import AudioSegment
import moviepy.editor as mp
import time

# =========================================================
# Extraction audio depuis une vidéo
# ---------------------------------------------------------
# - Entrée  : chemin vidéo, chemin de sortie audio (wav)
# - Sortie  : chemin du fichier audio généré
# - Effet   : écrit un fichier audio à partir de la piste de la vidéo
# =========================================================
def extract_audio_from_video(video_path: str, output_audio_path: str) -> str:
    """Extrait la piste audio d'une vidéo et l'enregistre en fichier audio."""
    video = mp.VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(output_audio_path, logger=None)
    return output_audio_path

# =========================================================
# Conversion MP3 -> WAV
# ---------------------------------------------------------
# - Entrée  : chemin .mp3
# - Sortie  : chemin .wav
# - Effet   : crée un WAV adjacent (même nom, extension .wav)
# =========================================================
def convert_mp3_to_wav(mp3_path: str) -> str:
    """Convertit un fichier MP3 en WAV (même répertoire/nom)."""
    audio = AudioSegment.from_file(mp3_path, format="mp3")
    wav_path = mp3_path.replace(".mp3", ".wav")
    audio.export(wav_path, format="wav")
    return wav_path

# =========================================================
# Transcription (Whisper)
# ---------------------------------------------------------
# - Entrée  : chemin audio, taille modèle, langue
# - Sortie  : liste de segments (dict) avec 'start'/'end'/'text'
# - Effet   : charge le modèle Whisper et transcrit l'audio
# =========================================================
def transcribe_segments(audio_path: str, model_size="large", language="fr"):
    """Transcrit un fichier audio via Whisper et renvoie les segments."""
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path, language=language, verbose=False)
    return result['segments']

# =========================================================
# Groupement de segments par durée
# ---------------------------------------------------------
# - Entrée  : segments Whisper, durée cible d'un chunk (sec)
# - Sortie  : liste de listes de segments (chunks)
# - Logique : accumule jusqu'à dépasser 'chunk_duration' puis coupe
# =========================================================
def group_segments_by_duration(segments, chunk_duration=60):
    """Regroupe séquentiellement les segments en blocs d'environ 'chunk_duration' secondes."""
    grouped = []
    current_chunk = []
    current_start = 0

    for seg in segments:
        if not current_chunk:
            current_start = seg['start']

        if seg['end'] - current_start > chunk_duration:
            grouped.append(current_chunk)
            current_chunk = [seg]
            current_start = seg['start']
        else:
            current_chunk.append(seg)

    if current_chunk:
        grouped.append(current_chunk)

    return grouped

# =========================================================
# Appel LLM pour produire un Markdown structuré/corrigé
# ---------------------------------------------------------
# - Entrée  : texte brut, nom du modèle local (Ollama)
# - Sortie  : texte Markdown (ou message d'erreur)
# - Effet   : POST /api/generate sur Ollama (stream=False)
# =========================================================
def generate_markdown_summary(text_chunk: str, model_name="mistral") -> str:
    """Demande à un LLM local de corriger/structurer un texte en Markdown strict."""
    prompt = f"""
Tu es un **relecteur-correcteur professionnel** spécialisé en langue française.

Ta mission est de corriger **toutes les fautes** présentes dans le texte suivant : orthographe, grammaire, accords, conjugaisons, ponctuation, typographie, mauvais usages de mots ou termes mal transcrits. Tu dois également améliorer le style pour garantir **clarté, fluidité et cohérence**, sans jamais altérer le sens du contenu.

Tu restitueras le texte corrigé au **format Markdown strict**, structuré comme suit :

### Ce que tu dois produire :
1. Un **titre principal** de niveau 1 (`#`) : clair, concis, informatif, sans ponctuation finale.  
2. Un **sous-titre** de 1 à 2 phrases synthétiques résumant l’ensemble du contenu de manière informatif et donnant envie au lecteur de lire la suite.  
3. Le **corps du texte**, organisé en paragraphes bien séparés par titre de niveaux 2 (`#`) clair, concis, informatif, sans ponctuation finale, en respectant la logique et la progression du texte d’origine.

### Règles strictes à respecter :
- Corrige toutes les fautes d’orthographe, de grammaire, de conjugaison et d’accords.  
- Corrige aussi les fautes de typographie (ponctuation, majuscules, espaces).  
- Améliore le style si nécessaire pour éviter les répétitions ou formulations maladroites.  
- Ne modifie pas le sens des phrases ni les informations techniques.  
- Si un terme semble erroné ou peu clair (ex. « pouvrante »), corrige-le par le bon mot (ex. « poudrante »).  
- Si un nom de lieu est mal orthographié, corrige-le selon la toponymie officielle (ex. « Saint-Pardou-la-Rivière » → « Saint-Pardoux-la-Rivière »).  
- Utilise uniquement **du Markdown strict** : pas de balises HTML, pas de commentaires, pas de formatage superflu. Uniquement `##` pour les titres et des paragraphes.

### Texte à corriger :


Voici le contenu brut à structurer :

```markdown
# 3. Le **corps du texte**, organisé en paragraphes bien séparés par des sauts de ligne, en respectant la logique et la progression du texte d’origine.


{text_chunk}

Réponds uniquement au format Markdown, sans texte additionnel hors structure demandée.
"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model_name, "prompt": prompt, "stream": False}
        )
        output = response.json().get("response", "")
        return output.strip()
    except Exception as e:
        return "## Erreur\n\nImpossible de générer le contenu Markdown structuré."

# =========================================================
# Sauvegarde de la transcription structurée (Markdown)
# ---------------------------------------------------------
# - Entrée  : groupes de segments, chemin de sortie (.md), modèle LLM
# - Sortie  : fichier .md structuré (et option TXT brut si bloc activé)
# - Effet   : concatène tous les segments, appelle le LLM, écrit le .md
# - NOTE    : la variante par-chunk est conservée en commentaire
# =========================================================
def save_structured_transcription_markdown(grouped_segments, output_path, model_name="mistral"):
    """Assemble le texte, génère le Markdown structuré et l'écrit sur disque."""
    output_path = Path(output_path).with_suffix(".md")

    all_text = []
    for i, group in enumerate(grouped_segments):
        all_text.append( " ".join([seg["text"] for seg in group]) )
    all_text = " ".join( all_text )

    # Enregistrer la transcription non structurée
    if False:
        output_path_transcript = Path(output_path).with_suffix(".txt")
        with open(output_path_transcript, "w", encoding="utf-8") as f:
            f.write("# Transcription Non Structurée\n\n")
            f.write(f"{all_text}\n\n---\n\n")

    markdown_summary = generate_markdown_summary(all_text, model_name=model_name)

    with open(output_path, "w", encoding="utf-8") as f:
        # f.write("# Transcription Structurée\n\n")
        f.write(f"{markdown_summary}")
    print(f"[✅] Transcription Markdown enregistrée dans : {output_path}")

    # Variante : génération par chunk (conservée telle quelle, commentée)
    # with open(output_path, "w", encoding="utf-8") as f:
    #     f.write("# Transcription Structurée\n\n")
    #     for i, group in enumerate(grouped_segments):
    #         block_text = " ".join([seg["text"] for seg in group])
    #         markdown_summary = generate_markdown_summary(block_text, model_name=model_name)
    #         time.sleep(1)
    #         f.write(f"{markdown_summary}\n\n---\n\n")
    # print(f"[✅] Transcription Markdown enregistrée dans : {output_path}")

# =========================================================
# Pipeline complet
# ---------------------------------------------------------
# - Étapes :
#   1) extrait audio de la vidéo (temporaire)
#   2) transcrit via Whisper
#   3) regroupe par durée
#   4) génère/sauvegarde le Markdown structuré via LLM local
# - Paramètres :
#   * model_size : taille Whisper (ex. 'large')
#   * language   : langue de transcription ('fr')
#   * chunk_duration : taille des blocs (s) pour regrouper segments
#   * model_name : modèle LLM utilisé par Ollama
# =========================================================
def structured_transcription_pipeline(video_path, output_text_path, model_size="medium", language="fr", chunk_duration=60, model_name="mistral"):
    """Exécute l'enchaînement extraction -> transcription -> regroupement -> structuration Markdown."""
    video_path = str(video_path)
    output_text_path = Path(output_text_path)
    print( video_path )
    print( str(output_text_path) )
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, "audio.wav")
        extract_audio_from_video(video_path, audio_path)
        segments = transcribe_segments(audio_path, model_size=model_size, language=language)
        grouped_segments = group_segments_by_duration(segments, chunk_duration=chunk_duration)
        save_structured_transcription_markdown(grouped_segments, output_text_path, model_name=model_name)

# =========================================================
# Point d'entrée script
# ---------------------------------------------------------
# - Deux exemples de fichiers en entrée/sortie (le second écrase le premier)
# - Lancement pipeline avec paramètres explicites
# =========================================================
if __name__ == "__main__":
    doc_in = "/var/www/RAG/Data/CWD FR/SELLES/ARGUMENTAIRES/ARGUMENTAIRE ARCON MADEMOISELLE.mp4"
    doc_out = "/var/www/RAG/Data_parse/CWD FR/SELLES/ARGUMENTAIRES/transcription_finale.md"


    structured_transcription_pipeline(
        video_path=doc_in ,
        output_text_path=doc_out,
        model_size="large",
        language="fr",
        chunk_duration=60,
        model_name="llama3.3:latest"
    )
