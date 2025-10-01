# rag_bot.py
# ---------------------------------------------------------
# RAG minimal (Ollama + Chroma) avec interface rag_bot()
# ---------------------------------------------------------
import os
from typing import Dict, List

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ---------- Config ----------
DATA_DIR = os.environ.get("RAG_DATA_DIR", "/var/www/RAG/Data_parse")     # .md / .txt
CHROMA_DIR = os.environ.get("RAG_CHROMA_DIR", "/var/www/RAG/chroma_index")
OLLAMA_BASE = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "nomic-embed-text")
GEN_MODEL = os.environ.get("RAG_GEN_MODEL", "llama3:latest")

CHUNK_SIZE = int(os.environ.get("RAG_CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.environ.get("RAG_CHUNK_OVERLAP", "50"))

RETRIEVER_K = int(os.environ.get("RAG_RETRIEVER_K", "5"))
RETRIEVER_FETCH_K = int(os.environ.get("RAG_RETRIEVER_FETCH_K", "20"))
RETRIEVER_LAMBDA = float(os.environ.get("RAG_RETRIEVER_LAMBDA", "0.5"))

# ---------- Embeddings ----------
embed_model = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE)

# ---------- Charger documents ----------
loader = DirectoryLoader(
    DATA_DIR,
    glob="**/*.md",                  # adapte si tu veux cibler un sous-ensemble (ex: "lilian_*.md")
    loader_cls=TextLoader,
    show_progress=True,
    use_multithreading=True,
    loader_kwargs={"autodetect_encoding": True},
)
docs = loader.load()

# ---------- Chunking ----------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)
chunks = splitter.split_documents(docs)

# ---------- Vector store (Chroma) ----------
db = Chroma.from_documents(chunks, embed_model, persist_directory=CHROMA_DIR)
#db.persist()

# ---------- Retriever ----------
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": RETRIEVER_K, "fetch_k": RETRIEVER_FETCH_K, "lambda": RETRIEVER_LAMBDA},
)

# ---------- Génération ----------
llm = OllamaLLM(
    model=GEN_MODEL,
    base_url=OLLAMA_BASE,
    temperature=0.0,
    num_ctx=8192,
    request_timeout=300,
)

# Prompt 
prompt = ChatPromptTemplate.from_template("""
Vous êtes un assistant utile. Répondez **uniquement** en utilisant le CONTEXTE.
- Répondez en **français**
- N'inventez pas de faits ; si l'information est absente, dites : "Je ne sais pas"
- **Quand le contexte contient des catégories ou termes, recopiez-les exactement**
- Terminez la réponse par le nom d’un fichier source si disponible, par ex. : (source : <fichier>)
- Utilisez exactement les termes trouvés dans le CONTEXTE.

=== CONTEXTE ===
{context}

=== QUESTION ===
{question}

=== RÉPONSE ===
""")



def format_docs(docs: List) -> str:
    return "\n\n".join(d.page_content for d in docs)

qa_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def rag_bot(question: str) -> Dict:
    """
    Interface standard pour l'évaluation locale.
    Retourne:
        {
          "answer": str,          # réponse générée
          "documents": List[Document]  # passages récupérés par le retriever
        }
    """
    docs = retriever.invoke(question)
    answer = qa_chain.invoke(question)
    return {"answer": answer, "documents": docs}

__all__ = ["rag_bot", "retriever", "qa_chain"]

# ---------- Test rapide en CLI ----------
if __name__ == "__main__":
    print("✅ RAG prêt. Tapez une question (ou 'exit').")
    try:
        while True:
            q = input("❓ Question: ").strip()
            if q.lower() in {"exit", "quit", "q"}:
                break
            out = rag_bot(q)
            print("\n🧠 Réponse:\n", out["answer"])
            print("\n📚 Docs utilisés:", len(out["documents"]))
            if out["documents"]:
                print("   Extrait doc[0]:", out["documents"][0].page_content[:300].replace("\n", " "))
            print("-" * 60)
    except KeyboardInterrupt:
        pass
