import asyncio
import json
import os

# 1) For reading .env environment variables
from dotenv import load_dotenv
load_dotenv()

# 2) MongoDB (motor)
import motor.motor_asyncio

# 3) LightRAG & related
from lightrag import LightRAG, QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import setup_logger, EmbeddingFunc

# 4) Azure/OpenAI
import openai
import numpy as np

# ------------------------------------------------------------------------
# Initialize Logger
# ------------------------------------------------------------------------
setup_logger("lightrag", level="INFO")

# ------------------------------------------------------------------------
# Azure-based Embedding Function
# ------------------------------------------------------------------------
async def azure_embed_func(texts: list[str]) -> np.ndarray:
    """
    Uses a separate Azure deployment for embeddings.
    """
    # Grab environment variables
    embed_endpoint = os.getenv("AZURE_EMBED_ENDPOINT", "")
    embed_key      = os.getenv("AZURE_EMBED_KEY", "")
    embed_version  = os.getenv("AZURE_EMBED_VERSION", "")
    embed_deploy   = os.getenv("AZURE_EMBED_DEPLOYMENT", "")

    # Configure openai for embeddings
    openai.api_type    = "azure"
    openai.api_base    = embed_endpoint
    openai.api_key     = embed_key
    openai.api_version = embed_version

    # Call the Embedding API
    response = openai.Embedding.create(
        input=texts,
        engine=embed_deploy  # Must match your embedding deployment name
    )

    embeddings = [item["embedding"] for item in response["data"]]
    return np.array(embeddings, dtype=np.float32)

# ------------------------------------------------------------------------
# Azure-based Chat (LLM) Function
# ------------------------------------------------------------------------
async def azure_chat_func(
    prompt: str,
    system_prompt: str = None,
    history_messages = [],
    **kwargs
) -> str:
    """
    Uses a separate Azure deployment for chat completions.
    """
    # Grab environment variables
    chat_endpoint = os.getenv("AZURE_CHAT_ENDPOINT", "")
    chat_key      = os.getenv("AZURE_CHAT_KEY", "")
    chat_version  = os.getenv("AZURE_CHAT_VERSION", "")
    chat_deploy   = os.getenv("AZURE_CHAT_DEPLOYMENT", "")

    # Configure openai for chat
    openai.api_type    = "azure"
    openai.api_base    = chat_endpoint
    openai.api_key     = chat_key
    openai.api_version = chat_version

    # Build chat messages
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)  # list of {"role":"assistant",...} or {"role":"user",...}
    messages.append({"role": "user", "content": prompt})

    # Call ChatCompletion
    response = openai.ChatCompletion.create(
        engine=chat_deploy,  # Must match your GPT-4 deployment name
        messages=messages,
        max_tokens=500,
        temperature=0.7
    )
    return response["choices"][0]["message"]["content"]

# ------------------------------------------------------------------------
# 1) Initialize LightRAG
# ------------------------------------------------------------------------
async def initialize_rag():
    # We'll define an EmbeddingFunc wrapper for azure_embed_func
    embed_wrapper = EmbeddingFunc(
        embedding_dim=1536,  # text-embedding-ada-002 is 1536 dims
        max_token_size=8192,
        func=azure_embed_func
    )

    rag = LightRAG(
        working_dir="./mongo_cache",
        embedding_func=embed_wrapper,
        llm_model_func=azure_chat_func,
        chunk_token_size=1200,
        chunk_overlap_token_size=100,
        enable_llm_cache=False,
        enable_llm_cache_for_entity_extract=False,
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag

# ------------------------------------------------------------------------
# 2) Insert MongoDB Users
# ------------------------------------------------------------------------
async def insert_users_from_mongo(
    rag: LightRAG,
    mongo_uri: str,
    db_name: str,
    collection_name: str,
    batch_size=10
):
    """
    Connects to MongoDB, reads all documents from `collection_name`,
    and inserts them into LightRAG in batches.
    """
    client = motor.motor_asyncio.AsyncIOMotorClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]

    cursor = collection.find({})
    all_docs = await cursor.to_list(length=None)
    print(f"Fetched {len(all_docs)} documents from '{db_name}.{collection_name}'")

    texts = []
    ids_for_docs = []

    for doc in all_docs:
        doc_id = doc.pop("_id", None)
        doc_text = json.dumps(doc, indent=2, default=str)
        texts.append(doc_text)
        ids_for_docs.append(str(doc_id) if doc_id else None)

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_ids   = ids_for_docs[i : i + batch_size]
        await rag.ainsert(batch_texts, ids=batch_ids)
        print(f"Inserted batch {i // batch_size + 1} of {(len(texts) // batch_size) + 1} (Mongo docs)")

# ------------------------------------------------------------------------
# 3) Compare All Retrieval Modes
# ------------------------------------------------------------------------
async def compare_all_modes(rag: LightRAG, user_question: str):
    # modes = ["naive", "local", "global", "hybrid", "mix"]
    modes=["local"]
    print(f"\n=== Comparing Answers for Question: '{user_question}' ===\n")
    for mode in modes:
        response = await rag.aquery(query=user_question, param=QueryParam(mode=mode))
        print(f"--- Mode: {mode.upper()} ---\n{response}\n{'-'*60}")

# ------------------------------------------------------------------------
# 4) Main
# ------------------------------------------------------------------------
async def main():
    # 1) Initialize RAG with Azure-based embed + chat
    rag = await initialize_rag()

    # 2) Insert user data from MongoDB
    MONGO_URI = (
        "mongodb+srv://mongouser:Bgg4t84OEouVtRsW@personanet.pfrlthb.mongodb.net/"
        "superintro-dev-db?retryWrites=true&w=majority"
    )
    DB_NAME = "superintro-dev-db"
    COLLECTION = "users"

    await insert_users_from_mongo(rag, MONGO_URI, DB_NAME, COLLECTION, batch_size=10)

    # 3) Example question
    question = "You will receive a current user's persona and a list of other users' personas for now lets take user = Younes Essaadani"
    "Your task is to find the top 5 matches based on shared age range, goals, interests, hobbies."
    "For each match, explain the reason for the match based on persona similarities"
    await compare_all_modes(rag, question)

if __name__ == "__main__":
    asyncio.run(main())
