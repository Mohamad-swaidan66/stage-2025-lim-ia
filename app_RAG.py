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

# === CONFIGURATION ===
DATA_DIR = "/var/www/RAG/Data_parse"
CHROMA_DIR = "./chroma_index"

embed_model = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
llm = OllamaLLM(model="gpt-oss:latest", base_url="http://localhost:11434", temperature=0.1, num_ctx=8192, request_timeout=3000)

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('\u00A0', ' ')
    return text.strip()

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

if os.path.exists(CHROMA_DIR):
    db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embed_model)
else:
    print("📚 Création de l’index vectoriel...")
    docs = load_and_clean_docs()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    db = Chroma.from_documents(chunks, embed_model, persist_directory=CHROMA_DIR)
    db.persist()

retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 20 , "lambda": 0.5})

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



def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

qa_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | question_prompt
    | llm
    | StrOutputParser()
)

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
            question_selector = gr.Dropdown(
                label="📋 Choisissez une question (optionnel)",
                choices=[
                    "Comment graisser sa selle juste après l'achat et dans le temps ?",
                    "Comment choisir sa selle ?",
                    "Que veux dire la marque CWD ?",
                    "Quel est le meilleur cavalier mondial qui monte avec une selle CWD ?",
                    "Qui est Pauline Martin ?",
                    "Combien de vaches faut-il pour produire l'ensemble des selles CWD par an ?",
                    "Quelle est la différence entre une SE31 et une SE32 ?",
                    "Comment dire si le fitting des panneaux est correct sur le cheval ?",
                    "Quel cuir choisir pour un cavalier qui monte 4 fois par semaine et qui aime se sentir accroché dans sa selle ?",
                ],
                interactive=True
            )
            question_input = gr.Textbox(
                label="💬 Ou tapez votre propre question",
                placeholder="Ex : Quelle est la durée de vie moyenne d’une selle ?",
                lines=3
            )
            run_button = gr.Button("🔍 Générer une réponse")
            last_action = gr.State()

        with gr.Column():
            thinking_output = gr.Markdown(visible=False)
            answer_output = gr.Markdown()
            sources_output = gr.Markdown()

    # Détection des actions utilisateur = mise à jour de l'état
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

    def trigger_thinking(selector, typed, action):
        return gr.update(visible=True, value="⏳ L’assistant réfléchit..."), selector, typed, action

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

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=6060, debug=True, root_path="/rag_cwd")
