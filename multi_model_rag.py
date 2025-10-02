import os
import re
import time  # Pour mesurer le temps de réponse
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate

# =========================================================
# CONFIGURATION GLOBALE
# ---------------------------------------------------------
# DATA_DIR   : répertoire où se trouvent les documents
# CHROMA_DIR : répertoire où sera stocké l’index Chroma
# =========================================================
DATA_DIR = "/var/www/RAG/Data_parse"
CHROMA_DIR = "/var/www/RAG/chroma_index"

# =========================================================
# EMBEDDING MODEL (Ollama)
# ---------------------------------------------------------
# Modèle utilisé : "nomic-embed-text"
# Sert à transformer le texte en vecteurs numériques
# =========================================================
embed_model = OllamaEmbeddings(
    model="nomic-embed-text", 
    base_url="http://localhost:11434"
)

# =========================================================
# LISTE DE MODELES LLM A TESTER
# ---------------------------------------------------------
# Plusieurs LLM configurés avec Ollama
# - llama3
# - mistral
# - llama3.3
# - gpt-oss
# - gpt-oss 120b
# Chaque modèle sera testé avec la même requête
# =========================================================
models = [
    OllamaLLM(
        model="llama3:latest",
        base_url="http://localhost:11434",
        temperature=0.1,
        num_ctx=8192,
        request_timeout=3000
    ),
    OllamaLLM(
        model="mistral:instruct",  
        base_url="http://localhost:11434",
        temperature=0.1,
        num_ctx=8192,
        request_timeout=3000
    ),
    OllamaLLM(
        model="llama3.3:latest",  
        base_url="http://localhost:11434",
        temperature=0.1,
        num_ctx=8192,
        request_timeout=3000
    ),
    OllamaLLM(
        model="gpt-oss:latest",  
        base_url="http://localhost:11434",
        temperature=0.1,
        num_ctx=8192,
        request_timeout=3000
    ),
    OllamaLLM(
        model="gpt-oss:120b",  
        base_url="http://localhost:11434",
        temperature=0.1,
        num_ctx=8192,
        request_timeout=3000
    )
]

# =========================================================
# NETTOYAGE TEXTE
# ---------------------------------------------------------
# - supprime les espaces multiples
# - remplace les espaces insécables
# =========================================================
def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('\u00A0', ' ')
    return text

# =========================================================
# CHARGEMENT & NETTOYAGE DOCUMENTS
# ---------------------------------------------------------
# - charge récursivement les fichiers texte
# - nettoie le contenu avec clean_text
# =========================================================
def load_and_clean_docs():
    loader = DirectoryLoader(
        DATA_DIR,
        glob="**/*",
        loader_cls=TextLoader,
        show_progress=True,
        use_multithreading=True,
        loader_kwargs={"autodetect_encoding": True}
    )
    docs = loader.load()
    for doc in docs:
        doc.page_content = clean_text(doc.page_content)
    return docs

# =========================================================
# CONSTRUCTION OU RECHARGEMENT DE L’INDEX CHROMA
# ---------------------------------------------------------
# - si CHROMA_DIR existe : recharge l’index
# - sinon : crée un nouvel index à partir des documents
# =========================================================
if os.path.exists(CHROMA_DIR):
    db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embed_model)
else:
    print("📚 Création de l’index vectoriel...")
    docs = load_and_clean_docs()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    db = Chroma.from_documents(chunks, embed_model, persist_directory=CHROMA_DIR)
    db.persist()

# =========================================================
# RETRIEVER
# ---------------------------------------------------------
# - MMR (Maximal Marginal Relevance)
#   * k=5 résultats finaux
#   * fetch_k=20 candidats initiaux
#   * lambda=0.5 équilibre pertinence/diversité
# =========================================================
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5 , "fetch_k": 20 , "lambda": 0.5}
)

# =========================================================
# PROMPT RAG
# ---------------------------------------------------------
# Contrainte : 
# - réponse précise et factuelle
# - pas d'invention
# - citer la source
# =========================================================
question_prompt = ChatPromptTemplate.from_template(""" 
Vous êtes un assistant expert de la marque CWD. Répondez  en vous basant sur les documents fournis.

Contraintes :
- Répondez de manière **directe, précise**
- **N'inventez jamais** d'informations non présentes dans les documents
- **Ne reformulez pas la question**
- **N’ajoutez pas de conseils, génériques ou hors sujet**
- Si une information est utilisée, citez sa **source (nom du document)**

=== CONTEXTE ===
{context}

=== QUESTION ===
{question}

=== RÉPONSE COURTE ===
""")

# =========================================================
# FORMATAGE DES DOCUMENTS POUR LE CONTEXTE
# =========================================================
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# =========================================================
# CHAÎNE QA
# ---------------------------------------------------------
# - prend un modèle + une question
# - exécute la RAG chain
# - renvoie la réponse générée
# =========================================================
def get_response_for_model(model, query):
    qa_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | question_prompt
        | model
        | StrOutputParser()
    )
    return qa_chain.invoke(query)

# =========================================================
# INTERFACE TERMINAL (tests multi-modèles)
# ---------------------------------------------------------
# - L'utilisateur saisit une question
# - Pour chaque modèle :
#   * lance l'inférence
#   * mesure le temps de réponse
#   * affiche réponse + durée
# =========================================================
print("\n✅ Système prêt. Posez vos questions (ou tapez 'exit' pour quitter).\n")
while True:
    try:
        query = input("❓ Votre question : ")
        if query.strip().lower() in ["exit", "quit", "q"]:
            print("👋 Fin du programme.")
            break
        
        # Test pour chaque modèle configuré
        for model in models:
            print(f"🔄 Test du modèle: {model.model}")
            start_time = time.time()
            response = get_response_for_model(model, query)
            end_time = time.time()
            duration = end_time - start_time

            # Affichage
            print(f"\n🧠 Réponse du modèle {model.model} :\n", response)
            print(f"⏱ Durée de réflexion : {duration:.2f} secondes")
            print("-" * 60)

    except KeyboardInterrupt:
        print("\n👋 Interrompu.")
        break
    except Exception as e:
        print("❌ Erreur :", e)
