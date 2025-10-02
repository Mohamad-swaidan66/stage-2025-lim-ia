# builder.py (LlamaIndex -> Chroma, partagé avec LangChain)
"""
Ingestion des documents puis construction d'un index VectorStoreIndex
persisté dans Chroma, afin d'être réutilisé par LlamaIndex ET LangChain.
"""

# =========================================================
# Imports
# ---------------------------------------------------------
# os        : vérification d'existence de répertoires
# chromadb  : client Chroma (vecteur store persistant)
# llama_index.core : objets de base (Index, Reader, Settings, Storage)
# Ollama    : LLM & Embeddings via Ollama (serveur local)
# SentenceSplitter : découpage en chunks
# ChromaVectorStore: adaptation LlamaIndex <-> Chroma
# =========================================================
import os
import chromadb
from llama_index.core import (
    VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore

# =========================================================
# Configuration chemins & collection
# ---------------------------------------------------------
# DATA_DIR   : répertoire source des documents à indexer
# CHROMA_DIR : répertoire du stockage persistant Chroma (HNSW)
# COLLECTION : nom logique de la collection Chroma
# =========================================================
DATA_DIR = "/var/www/RAG/Data_parse/"
CHROMA_DIR = "/var/www/RAG/chroma_index"
COLLECTION = "cwd_knowledge"

# =========================================================
# 1) Modèles (paramétrage global via Settings)
# ---------------------------------------------------------
# - Embeddings : "nomic-embed-text" (Ollama)
# - LLM        : "llama3:latest" (Ollama)
# - Node parser: SentenceSplitter (chunk 500, overlap 50)
# NOTE: ces Settings seront utilisés lors de la construction de l'index.
# =========================================================
Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    base_url="http://localhost:11434",
)
Settings.llm = Ollama(
    model="llama3:latest",
    base_url="http://localhost:11434",
    temperature=0.1,
    request_timeout=60,
)
Settings.node_parser = SentenceSplitter(chunk_size=500, chunk_overlap=50)

# =========================================================
# 2) Ingestion des documents
# ---------------------------------------------------------
# - Vérifie l'existence du répertoire d'entrée
# - Charge récursivement tous les fichiers supportés
# - Transforme les documents en "nodes" (chunks) via le node_parser
# =========================================================
if not os.path.isdir(DATA_DIR):
    raise FileNotFoundError(f"Répertoire introuvable: {DATA_DIR}")

print("📥 Chargement des documents…")
documents = SimpleDirectoryReader(DATA_DIR, recursive=True).load_data()
print(f"📄 Fichiers détectés : {len(documents)}")

nodes = Settings.node_parser.get_nodes_from_documents(documents)
print(f"🧩 Chunks générés : {len(nodes)}")

# =========================================================
# 3) Initialisation Chroma
# ---------------------------------------------------------
# - PersistentClient : charge/crée le store sur disque (CHROMA_DIR)
# - get_or_create_collection : récupère ou crée la collection cible
# - hnsw:space=cosine : métrique de similarité (cosine) pour HNSW
# - ChromaVectorStore : adapter côté LlamaIndex
# - StorageContext    : contexte de stockage pour l'Index
# =========================================================
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection(
    name=COLLECTION,
    metadata={"hnsw:space": "cosine"}
)
vector_store = ChromaVectorStore(chroma_collection=collection)
storage_ctx = StorageContext.from_defaults(vector_store=vector_store)

# =========================================================
# 4) Construction de l'index (persistance côté Chroma)
# ---------------------------------------------------------
# - VectorStoreIndex construit à partir des nodes et du storage_ctx
# - show_progress=True : affichage des barres de progression
# - La persistance est gérée par Chroma (collection + path)
# =========================================================
print("🧠 Construction de l'index vectoriel dans Chroma…")
index = VectorStoreIndex(nodes, storage_context=storage_ctx, show_progress=True)

print(f"✅ Index persistant prêt dans: {CHROMA_DIR}  (collection: {COLLECTION})")
print(f"🧠 Total chunks indexés : {len(nodes)}")
