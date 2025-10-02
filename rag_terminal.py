import os
import re
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
# DATA_DIR   : répertoire contenant les documents prétraités
# CHROMA_DIR : répertoire où sera stocké l’index Chroma persistant
# =========================================================
DATA_DIR = "/var/www/RAG/Data_parse"
CHROMA_DIR = "/var/www/RAG/chroma_index"

# =========================================================
# EMBEDDING MODEL (Ollama)
# ---------------------------------------------------------
# - Utilise "nomic-embed-text" pour transformer les documents en vecteurs
# - base_url = Ollama local
# =========================================================
embed_model = OllamaEmbeddings(
    model="nomic-embed-text", 
    base_url="http://localhost:11434"
)

# =========================================================
# LLM (Ollama - Llama3)
# ---------------------------------------------------------
# - Modèle utilisé : "llama3:latest"
# - Contexte max = 8192 tokens
# - Température basse (0.1) pour réponses précises et déterministes
# =========================================================
llm = OllamaLLM(
    model="llama3:latest",
    base_url="http://localhost:11434",
    temperature=0.1,
    num_ctx=8192,
    request_timeout=3000
)

# =========================================================
# FONCTION DE NETTOYAGE TEXTE
# ---------------------------------------------------------
# - Supprime espaces multiples
# - Remplace espaces insécables par des espaces normaux
# - Retourne texte nettoyé
# =========================================================
def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('\u00A0', ' ')
    return text

# =========================================================
# CHARGEMENT + NETTOYAGE DES DOCUMENTS
# ---------------------------------------------------------
# - DirectoryLoader : charge tous les fichiers texte sous DATA_DIR
# - TextLoader      : détecte automatiquement l'encodage
# - Nettoyage du contenu avant indexation
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
# CONSTRUCTION OU RECHARGEMENT DE L'INDEX CHROMA
# ---------------------------------------------------------
# - Si index existe déjà : on le recharge
# - Sinon :
#   * On charge/nettoie les docs
#   * On les découpe en chunks (500 tokens, overlap 50)
#   * On construit un nouvel index Chroma et on le persiste
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
# - Méthode : MMR (Maximal Marginal Relevance)
#   * k=5     : nombre final de passages retournés
#   * fetch_k : 20 candidats initiaux
#   * lambda=0.5 : équilibre entre pertinence & diversité
# =========================================================
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5 , "fetch_k": 20 , "lambda": 0.5}
)

# =========================================================
# PROMPT RAG
# ---------------------------------------------------------
# - Ton : expert, précis, direct
# - Contraintes :
#   * Pas d'invention
#   * Pas de reformulation de la question
#   * Pas de conseils hors sujet
#   * Obligation de citer la source (nom du document)
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
# FORMATAGE DU CONTEXTE
# ---------------------------------------------------------
# - Concatène le contenu des documents récupérés
# - Séparés par deux sauts de ligne pour lisibilité
# =========================================================
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# =========================================================
# CHAÎNE QA (RAG)
# ---------------------------------------------------------
# - Étapes :
#   1) "context" : retriever + format_docs
#   2) "question" : passe telle quelle (RunnablePassthrough)
#   3) Injection dans le prompt
#   4) LLM génère réponse
#   5) StrOutputParser : formate la sortie texte
# =========================================================
qa_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | question_prompt
    | llm
    | StrOutputParser()
)

# =========================================================
# BOUCLE TERMINAL (Interface CLI)
# ---------------------------------------------------------
# - L'utilisateur saisit une question
# - Réponse générée via qa_chain
# - Quitter avec "exit" / "quit" / "q"
# =========================================================
print("\n✅ Système prêt. Posez vos questions (ou tapez 'exit' pour quitter).\n")
while True:
    try:
        query = input("❓ Votre question : ")
        if query.strip().lower() in ["exit", "quit", "q"]:
            print("👋 Fin du programme.")
            break
        response = qa_chain.invoke(query)
        print("\n🧠 Réponse :\n", response)
        print("-" * 60)
    except KeyboardInterrupt:
        print("\n👋 Interrompu.")
        break
    except Exception as e:
        print("❌ Erreur :", e)
