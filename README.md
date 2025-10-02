# stage-2025-lim-ia
Stage 2025 LIM — Prototype RAG (Retrieval-Augmented Generation) : ingestion, retrieval, évaluation

Ce projet regroupe plusieurs modules permettant d’extraire, traiter et interroger des connaissances métiers de la marque **CWD** à partir de différentes sources (images, vidéos, documents texte).  
Il s’appuie sur **Whisper, LangChain, LlamaIndex, ChromaDB**, et des modèles **LLM via Ollama**.

---

## ✨ Fonctionnalités principales

- **OCR vers Markdown**
  - Extraction de texte depuis des images (fiches techniques, descriptifs selles).
  - Nettoyage, corrections automatiques, structuration en Markdown.

- **Transcription vidéo**
  - Extraction audio depuis vidéos (`.mp4`).
  - Transcription avec Whisper (multi-langues).
  - Segmentation, regroupement par durée, et correction/structuration automatique via LLM.

- **RAG (Retrieval-Augmented Generation)**
  - Ingestion des documents texte/Markdown.
  - Indexation vectorielle avec ChromaDB.
  - Recherche sémantique (MMR) et génération de réponses factuelles via LangChain ou LlamaIndex.

- **Interfaces utilisateurs**
  - Terminal (CLI) : poser directement une question en boucle interactive.
  - Gradio Web UI : interface simple avec menu de questions prédéfinies + saisie libre.
  - Comparateur multi-LLM : tester la même question avec plusieurs modèles (Llama3, Mistral, GPT-OSS, etc.) et mesurer temps de réponse.

---

## ⚙️ Prérequis

- Python 3.9+
- Dépendances système :
  - [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) (avec `fra` et `eng`)
  - [ffmpeg](https://ffmpeg.org/) (utilisé par `moviepy` et `pydub`)

### Installation Python
```bash
pip install -r requirements.txt
