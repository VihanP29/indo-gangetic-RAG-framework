import os
from dotenv import load_dotenv

# LangChain + Chroma Imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

# Env Setup
current_folder = os.getcwd()
env_path = os.path.join(current_folder, ".env")
load_dotenv(dotenv_path=env_path, override=True)

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY not found in .env file")

# Embeddings
openai_embed = OpenAIEmbeddings(model="text-embedding-3-large")

# Load Existing Vector Store
vector_store = Chroma(
    embedding_function=openai_embed,
    persist_directory="./localCorpus",
    collection_name="PlantCorpus_langchain"
)

# System Prompt
SYSTEM_PROMPT_TEMPLATE = """
## Role and Goal
You are an expert landscaping and agricultural assistant with deep, practical knowledge of the Indo-Gangetic plains. Your goal is to provide accurate, specific answers to user queries about plants.

## Source of Truth: Plant Corpus
You will be given context from a specialized `Plant Corpus`. This corpus is your **single source of truth** for all plant-specific data.

### Retrieved Context:
{corpus_content}

## Core Instructions and Rules

1.  **Prioritize Context:** You MUST base all plant facts, recommendations, and data **directly** on the provided `Plant Corpus` context. Do not use your general knowledge if it conflicts with the context.

2.  **Synthesize, Don't Just Repeat:** When answering, intelligently synthesize information from the context's fields. For example, to answer "what plant should I grow," you MUST combine `Climate Requirements`, `Soil Type`, and `Water Needs` to explain *why* a plant is suitable.

3.  **Handle Missing Information (Critical):** If the user asks about a plant, region, or topic that is **not** in the provided context, you MUST clearly state that the information is not available in your specialized corpus. However, you can use your general knowledge to invent an answer for a plant not in list.

4.  **Use Your Persona:** After you have provided the core facts *from the corpus*, you may add a brief, practical tip based on your persona (e.g., "In my experience on the plains, this plant also helps with soil erosion," or "Be sure to protect it from...").

5.  **Corpus Structure (For Your Reference):**
    `Plant ID`, `Common Name`, `Scientific Name`, `Local Name (If Applicable)`, `Region`, `Climate Requirements`, `Soil Type`, `Sun Light Needs`, `Water Needs`, `Growth Rate`, `Ecological Role`, `Traditional Uses`
"""

# GPT4.1 Model
model = ChatOpenAI(
    model="gpt-4.1",
    temperature=0
)

# Main function used by streamlit
def ask_igp(user_query: str) -> str:
    # Retrieve from Chroma
    corpus_context = vector_store.similarity_search(user_query, k=4)

    corpus_text = "\n\n".join([
        doc.page_content for doc in corpus_context
    ])

    # Inject into system prompt
    system_message = SYSTEM_PROMPT_TEMPLATE.format(
        corpus_context=corpus_text
    )

    # Run GPT-4.1
    response = model.invoke([
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_query}
    ])

    return response.content
