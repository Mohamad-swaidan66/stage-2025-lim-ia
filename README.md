# stage-2025-lim-ia
Stage 2025 LIM — Prototype RAG (Retrieval-Augmented Generation) : ingestion, retrieval, évaluation


Ce projet regroupe plusieurs modules permettant d’extraire, traiter et interroger des connaissances métiers de la marque CWD à partir de différentes sources (images, vidéos, documents texte).
Il s’appuie sur Whisper, LangChain, LlamaIndex, ChromaDB, et des modèles LLM via Ollama.

✨ Fonctionnalités principales

OCR vers Markdown

Extraction de texte depuis des images (fiches techniques, descriptifs selles).

Nettoyage, corrections automatiques, structuration en Markdown.

Transcription vidéo

Extraction audio depuis vidéos (.mp4).

Transcription avec Whisper (multi-langues).

Segmentation, regroupement par durée, et correction/structuration automatique via LLM.

RAG (Retrieval-Augmented Generation)

Ingestion des documents texte/Markdown.

Indexation vectorielle avec ChromaDB.

Recherche sémantique (MMR) et génération de réponses factuelles via LangChain ou LlamaIndex.

Interfaces utilisateurs

Terminal (CLI) : poser directement une question en boucle interactive.

Gradio Web UI : interface simple avec menu de questions prédéfinies + saisie libre.

Comparateur multi-LLM : tester la même question avec plusieurs modèles (Llama3, Mistral, GPT-OSS, etc.) et mesurer temps de réponse.

⚙️ Prérequis

Python 3.9+

Dépendances système :

Tesseract OCR
 (avec fra et eng)

ffmpeg
 (utilisé par moviepy et pydub)

Ollama installé et en service local sur http://localhost:11434 avec les modèles nécessaires


Installation Python
pip install -r requirements.txt

📂 Structure du projet
.
├── builder.py                     # Ingestion via LlamaIndex -> Chroma
├── ocr_to_markdown.py              # OCR image -> Markdown structuré
├── video_transcriber.py            # Transcription vidéo -> Markdown
├── rag_terminal.py                  # Interface CLI RAG
├── rag_gradio.py                    # Interface web avec Gradio
├── multi_model_benchmark.py         # Comparaison multi-LLM avec chronométrage
├── /Data                           # Données brutes (images, vidéos, docs)
├── /Data_parse                     # Sorties Markdown et index vectoriels
├── /chroma_index                   # Stockage persistant ChromaDB
└── README.md



Projet interne LIM — développé pour l’assistant technique & commercial CWD.

