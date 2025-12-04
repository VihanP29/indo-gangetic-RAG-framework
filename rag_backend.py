import os
import pandas as pd
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Env setup
current_folder = os.getcwd()
env_path = os.path.join(current_folder, ".env")
load_dotenv(dotenv_path=env_path, override=True)

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY not found in .env file")

# Load CSV into documents
csv_path = os.path.join(current_folder, "CleanCorpus.csv")

if not os.path.exists(csv_path):
    raise RuntimeError("CleanCorpus.csv NOT FOUND in project directory")

df = pd.read_csv(csv_path)

documents = []
for _, row in df.iterrows():
    text_blob = "\n".join([f"{col}: {row[col]}" for col in df.columns])
    documents.append(Document(page_content=text_blob))

# Embeddings
openai_embed = OpenAIEmbeddings(model="text-embedding-3-large")

# Vectort store
vector_store = Chroma.from_documents(
    documents=documents,
    embedding=openai_embed
)

# System prompt
SYSTEM_PROMPT_TEMPLATE = """
## Role and Goal
You are an expert landscaping and agricultural assistant with deep, practical knowledge of the Indo-Gangetic plains. Your goal is to provide accurate, specific answers to user queries about plants.

## Source of Truth: Plant Corpus
You will be given context from a specialized `Plant Corpus`. This corpus is your **single source of truth** for all plant-specific data.

### Retrieved Context:
{corpus_content}

## Core Instructions and Rules
1. You MUST base all plant facts on the provided corpus.
2. You MUST synthesize Climate + Soil + Water when recommending plants.
3. If a plant is NOT in the corpus, clearly say so.
4. After facts, add one short practical tip.
"""

# Model
model = ChatOpenAI(
    model="gpt-4.1",
    temperature=0
)

# Main function
def ask_igp(user_query: str) -> str:
    corpus_context = vector_store.similarity_search(user_query, k=4)

    if not corpus_context:
        return "No relevant plant data was found in the plant corpus for this query."

    corpus_text = "\n\n".join([doc.page_content for doc in corpus_context])

    system_message = SYSTEM_PROMPT_TEMPLATE.format(
        corpus_content=corpus_text
    )

    response = model.invoke([
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_query}
    ])

    return response.content

