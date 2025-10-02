# =========================================================
# Imports
# ---------------------------------------------------------
# - os: utilitaires système (existence de dossiers/fichiers)
# - re: nettoyage de texte via expressions régulières
# - gradio: interface web pour l'app RAG
# - langchain / ollama / chroma: stack RAG (LLM, embeddings, vecteur store, prompt, chain)
# - text splitter: découpage des documents en chunks
# =========================================================
import os
import re
import gradio as gr
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
# - DATA_DIR: répertoire contenant les fichiers déjà parsés (Markdown/TXT)
# - CHROMA_DIR: répertoire de persistance de l'index Chroma
# - embed_model: modèle d'embedding (Ollama)
# - llm: modèle de génération (Ollama)
# =========================================================
DATA_DIR = "/var/www/RAG/Data_parse"
CHROMA_DIR = "./chroma_index"

embed_model = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
llm = OllamaLLM(
    model="gpt-oss:latest",
    base_url="http://localhost:11434",
    temperature=0.1,
    num_ctx=8192,
    request_timeout=3000
)

# =========================================================
# UTILITAIRES TEXTE
# ---------------------------------------------------------
# clean_text:
# - compresse les espaces multiples
# - remplace l'espace insécable par un espace normal
# - retire espaces en début/fin
# =========================================================
def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('\u00A0', ' ')
    return text.strip()

# =========================================================
# CHARGEMENT & NETTOYAGE DES DOCUMENTS
# ---------------------------------------------------------
# load_and_clean_docs:
# - charge tous les fichiers texte sous DATA_DIR (récursif)
# - auto-détecte l'encodage
# - nettoie le contenu via clean_text
# - retourne la liste de documents LangChain
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
# INDEX VECTORIEL CHROMA
# ---------------------------------------------------------
# - Si CHROMA_DIR existe : on recharge l'index existant
# - Sinon : on crée l'index à partir des documents (split puis persist)
#   * chunk_size=500, overlap=50: compromis précision/rappel
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
# - MMR (Maximal Marginal Relevance): équilibre pertinence/diversité
# - k=5: nombre final de passages retournés
# - fetch_k=20: candidats initiaux avant MMR
# - lambda=0.5: équilibre MMR (0 = diversité, 1 = similarité)
# =========================================================
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20 , "lambda": 0.5}
)

# =========================================================
# PROMPT RAG
# ---------------------------------------------------------
# - Contexte injecté depuis les documents récupérés
# - Ton: direct, concis, factuel
# - Exigence: citer la source (ex: nom de fichier) quand info mentionnée
# - Si info absente: le dire clairement
# =========================================================
question_prompt = ChatPromptTemplate.from_template("""
Vous êtes un assistant technique de la marque CWD. Répondez de manière **directe, concise et strictement factuelle** à la question posée, en vous appuyant uniquement sur les documents fournis.

📌 Contraintes :
- Évitez toute reformulation de la question
- **Ne donnez aucune recommandation générale ou commerciale**
- Ne répétez pas d’information inutile ou hors sujet
- Si une information est mentionnée, citez **clairement sa source** (ex. : nom du fichier)
- Si l’information est absente, dites-le clairement, sans supposition

=== CONTEXTE DOCUMENTAIRE ===
{context}

=== QUESTION ===
{question}

=== RÉPONSE COURTE ===
""")

# =========================================================
# FORMATTEUR DE CONTEXTE
# ---------------------------------------------------------
# format_docs:
# - concatène les contenus des documents sélectionnés
# - séparés par deux sauts de ligne (lisibilité)
# =========================================================
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# =========================================================
# CHAÎNE RAG (retrieval -> prompt -> LLM -> parseur)
# ---------------------------------------------------------
# - "context": passe par le retriever puis formatage
# - "question": transite telle quelle (RunnablePassthrough)
# - StrOutputParser: standardise la sortie sous forme de texte
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
# FONCTIONS D'INTERFACE MÉTIER
# ---------------------------------------------------------
# rag_interface(query):
# - gère l'absence de saisie
# - invoque la chaîne RAG
# - renvoie la réponse et un bloc "sources" indicatif
#
# resolve_question(selected, typed, action):
# - choisit la question effective selon l'origine (sélecteur vs saisie)
# =========================================================
def rag_interface(query):
    if not query:
        return "⚠️ Veuillez entrer une question.", ""
    try:
        answer = qa_chain.invoke(query)
        return f"💡 **Réponse :**\n\n{answer}", f"📂 **Fichiers consultés :**\n(index Chroma local)"
    except Exception as e:
        return f"❌ Erreur : {str(e)}", ""

def resolve_question(selected, typed, action):
    if action == "typed":
        return typed
    elif action == "selected":
        return selected
    else:
        return typed or selected

# =========================================================
# UI GRADIO
# ---------------------------------------------------------
# - En-tête: logo + titre + sous-titre
# - Colonne gauche: dropdown de questions + zone de texte + bouton
# - Colonne droite: états ("thinking") + réponse + sources
# - Logique:
#   * change() met à jour last_action (selected/typed)
#   * click() affiche "thinking", exécute RAG, masque "thinking"
# - root_path="/rag_cwd": utile derrière un reverse proxy
# =========================================================
with gr.Blocks(title="🧠 Assistant CWD", theme=gr.themes.Soft(primary_hue="red", secondary_hue="gray")) as demo:
    gr.Markdown("""
    <div style="text-align:center">
        <img src="cwd_logo.png" width="150">
        <h2 style="color:#8B0000;">🧠 Assistant Technique & Commercial CWD</h2>
        <p>Posez votre question ci-dessous pour obtenir une réponse experte basée sur vos documents internes.</p>
    </div>
    """)

    with gr.Row():
        with gr.Column():
            # Sélecteur de questions prédéfinies (facilite les tests)
            question_selector = gr.Dropdown(
                label="📋 Choisissez une question (optionnel)",
                choices=[
                ],
                interactive=True
            )
            # Zone de saisie libre
            question_input = gr.Textbox(
                label="💬 Ou tapez votre propre question",
                placeholder="Ex : Quelle est la durée de vie moyenne d’une selle ?",
                lines=3
            )
            # Bouton de lancement
            run_button = gr.Button("🔍 Générer une réponse")
            # État interne pour savoir si l'utilisateur a tapé ou sélectionné
            last_action = gr.State()

        with gr.Column():
            # Zone "thinking" (affichée pendant l'inférence)
            thinking_output = gr.Markdown(visible=False)
            # Réponse finale
            answer_output = gr.Markdown()
            # Sources/infos complémentaires
            sources_output = gr.Markdown()

    # --- Détection de l'action utilisateur (met à jour last_action) ---
    question_selector.change(
        fn=lambda val: "selected",
        inputs=question_selector,
        outputs=last_action
    )

    question_input.change(
        fn=lambda val: "typed",
        inputs=question_input,
        outputs=last_action
    )

    # --- Affichage d'un état "réflexion" avant l'appel RAG ---
    def trigger_thinking(selector, typed, action):
        return gr.update(visible=True, value="⏳ L’assistant réfléchit..."), selector, typed, action

    # --- Pipeline clic:
    # 1) affiche "thinking"
    # 2) exécute la RAG sur la question résolue
    # 3) masque "thinking"
    run_button.click(
        fn=trigger_thinking,
        inputs=[question_selector, question_input, last_action],
        outputs=[thinking_output, question_selector, question_input, last_action]
    ).then(
        fn=lambda selector, typed, action: rag_interface(resolve_question(selector, typed, action)),
        inputs=[question_selector, question_input, last_action],
        outputs=[answer_output, sources_output]
    ).then(
        fn=lambda: gr.update(visible=False),
        outputs=[thinking_output]
    )

# =========================================================
# LANCEMENT DE L'APP
# ---------------------------------------------------------
# - server_name="127.0.0.1": écoute locale
# - server_port=6060: port HTTP
# - debug=True: logs détaillés (utile dev)
# - root_path="/rag_cwd": chemin racine si derrière un proxy
# =========================================================
if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=6050, debug=True, root_path="/rag_cwd")
