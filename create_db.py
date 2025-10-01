#%%
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document
from llama_index.core.agent.workflow import FunctionAgent
# from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# snowflake-arctic-embed
embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    base_url="http://localhost:11434",
    # ollama_additional_kwargs={"mirostat": 0},
    num_ctx=80000, 
)
Settings.embed_model = embed_model
# Configurez le modèle Ollama
Settings.llm = Ollama(model="gemma3:27b", num_ctx=80000, base_url="http://localhost:11434", temperature=0.1, request_timeout=3000)

# response = llm.complete("What is the capital of France?")
# print(response)

# Create a RAG tool using LlamaIndex
# documents = SimpleDirectoryReader(r'/var/www/RAG/Data_parse', recursive=True).load_data()
documents = SimpleDirectoryReader(r'/var/www/RAG/Data_parse/', recursive=True).load_data()
print( documents )
index = VectorStoreIndex.from_documents(documents,show_progress=True,embed_model=embed_model)
query_engine = index.as_query_engine()


# Exemple de requête
query = "Tu es un représentant pour la marque CWD. Comment graisser sa selle juste après l'achat et dans le temps ? Réponds en francais avec un langage soutenu et un style professionnel. Soit exhaustif dans ta réponse et ne réponds qu'à la question posée."

response = query_engine.query(query)
print(response)
# %%
