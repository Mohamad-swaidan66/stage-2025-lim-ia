import os
import re
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate

# === CONFIGURATION ===
DATA_DIR = "/var/www/RAG/Data_parse"
CHROMA_DIR = "/var/www/RAG/chroma_index"


# === EMBEDDING MODEL ===
embed_model = OllamaEmbeddings(
    model="nomic-embed-text", 
    base_url="http://localhost:11434"
)

# === GEMMA LLM ===
llm = OllamaLLM(
    model="llama3:latest",
    base_url="http://localhost:11434",
    temperature=0.1,
    num_ctx=8192,
    request_timeout=3000
)

# === TEXT CLEANING ===
def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('\u00A0', ' ')
    return text

# === LOADER + CLEAN ===
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

# === BUILD OR LOAD CHROMA ===
if os.path.exists(CHROMA_DIR):
    db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embed_model)
else:
    print("📚 Création de l’index vectoriel...")
    docs = load_and_clean_docs()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    db = Chroma.from_documents(chunks, embed_model, persist_directory=CHROMA_DIR)
    db.persist()

# === RETRIEVER ===
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5 , "fetch_k": 20 , "lambda": 0.5}
)

# === PROMPT ===
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


# === FORMAT CONTEXT ===
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# === QA CHAIN ===
qa_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | question_prompt
    | llm
    | StrOutputParser()
)

 #query = "Comment graisser sa selle juste après l'achat et dans le temps ? "

 #query = "comment choisir sa selle ?"

 #query = "Que veux dire la marque CWD ? "

 #query = "Quel est le meilleur cavalier mondial qui monte avec une selle CWD ?"

 #query = "Qui est Pauline Martin ?"

 #uery = "Combien de vache faut-il pour produire l'ensemble des selles CWD par an ?"

 #query = "Quelle est la différence entre une SE31 et une SE32 ?"

 #query = "Comment dire si le fitting des panneaux est correct sur le cheval ?"

 #query = "Quel cuire choisir pour un cavalier qui monte 4 fois par semaine et qui aime se sentir accroché dans sa selle ?"


# === INTERFACE TERMINAL ===
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
