import os
from dotenv import load_dotenv

# === LangChain + Chroma Imports ===
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

# === ENV SETUP ===
current_folder = os.getcwd()
env_path = os.path.join(current_folder, ".env")
load_dotenv(dotenv_path=env_path, override=True)

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("❌ OPENAI_API_KEY not found in .env file")

# === EMBEDDINGS (Matches Your Notebook) ===
openai_embed = OpenAIEmbeddings(model="text-embedding-3-large")

# === LOAD EXISTING VECTOR STORE ===
vector_store = Chroma(
    embedding_function=openai_embed,
    persist_directory="./localCorpus",
    collection_name="PlantCorpus_langchain"
)

# === SYSTEM PROMPT (Matches Your Rules) ===
SYSTEM_PROMPT_TEMPLATE = """
You are an expert landscaping and agricultural assistant with deep, practical knowledge.

You MUST use the following retrieved Indo-Gangetic Plant Corpus as your ONLY factual source.

Rules:
1. Prioritize context from the retrieved corpus.
2. Synthesize information intelligently.
3. If information is missing, say so clearly.
4. Structure output as:
   - Common Name
   - Scientific Name
   - Local Name (if available)
   - Soil Needs
   - Water Needs
   - Climate
   - Growth Rate
   - Ecological Benefits
   - Traditional Uses (if applicable)

Retrieved Context:
{corpus_context}
"""

# === GPT-4.1 MODEL ===
model = ChatOpenAI(
    model="gpt-4.1",
    temperature=0
)

# === MAIN FUNCTION USED BY STREAMLIT ===
def ask_igp(user_query: str) -> str:
    # 1️⃣ Retrieve from Chroma
    corpus_context = vector_store.similarity_search(user_query, k=4)

    corpus_text = "\n\n".join([
        doc.page_content for doc in corpus_context
    ])

    # 2️⃣ Inject into system prompt
    system_message = SYSTEM_PROMPT_TEMPLATE.format(
        corpus_context=corpus_text
    )

    # 3️⃣ Run GPT-4.1
    response = model.invoke([
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_query}
    ])

    return response.content
