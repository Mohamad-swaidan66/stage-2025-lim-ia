# stage-2025-lim-ia
Stage 2025 LIM — Prototype RAG (Retrieval-Augmented Generation) : ingestion, retrieval, évaluation

CWD RAG & OCR Project

Ce projet regroupe plusieurs modules permettant d’extraire, traiter et interroger des connaissances métiers de la marque CWD à partir de différentes sources (images, vidéos, documents texte).
Il s’appuie sur OCR (Tesseract), Whisper, LangChain, LlamaIndex, ChromaDB, et des modèles LLM via Ollama.

✨ Fonctionnalités principales

OCR vers Markdown

Extraction de texte depuis des images (fiches techniques, descriptifs selles).

Nettoyage, corrections automatiques, structuration en Markdown.

Ajout automatique de la matière (GV, DV, DB…) et de la référence de selle (ex: SE123).

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

Ollama installé et en service local sur http://localhost:11434 avec les modèles nécessaires :

llama3:latest

llama3.3:latest

mistral:instruct

gpt-oss:latest

gpt-oss:120b

nomic-embed-text (embeddings)

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

🚀 Utilisation
1. OCR sur image unique
python ocr_to_markdown.py


Résultat : un fichier .md généré à partir de l’image.

2. Transcription vidéo
python video_transcriber.py


Résultat : transcription structurée et corrigée en Markdown.

3. Construction index Chroma (via LlamaIndex)
python builder.py

4. RAG (Terminal CLI)
python rag_terminal.py


Exemple :

❓ Votre question : Quelle est la différence entre SE31 et SE32 ?
🧠 Réponse : ...

5. RAG (Interface Web Gradio)
python rag_gradio.py


Interface disponible sur http://127.0.0.1:6060/rag_cwd.

6. Benchmark multi-modèles
python multi_model_benchmark.py


Teste la même question avec plusieurs LLM et affiche temps de réponse.

📄 Exemple de sortie OCR
**Selle : SE123**
**Matière : grainé veau**

## Arçon
Arçon en bois renforcé, apportant robustesse et souplesse.

## Siège
Siège semi-creux, conçu pour un équilibre optimal du cavalier.

⚠️ Limitations & Améliorations futures

La détection de titres OCR repose sur des heuristiques simples → améliorable avec NLP.

Les chemins sont actuellement codés en dur dans les scripts → à remplacer par des arguments CLI.

La correction du texte OCR s’appuie sur une liste fermée de patterns → à enrichir.

Ajout futur possible :

API REST pour poser des questions au RAG.

Intégration d’une base de connaissances continue.

👨‍💻 Auteur

Projet interne LIM — développé pour l’assistant technique & commercial CWD.

👉 Avec ça, tu as un README global qui couvre toutes les briques de ton projet.
