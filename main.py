import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

app = FastAPI(
    title="AI RAG Microservice",
    description="Production-ready FastAPI service combining RAG and Vector Search",
    version="1.0.0"
)

# Load lightweight open-source embedding model locally (Runs on CPU for free)
embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

class DocumentIngest(BaseModel):
    text: str
    metadata: dict = {}

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

@app.get("/")
def health_check():
    return {"status": "online", "system": "AI Microservice running"}

@app.post("/embed")
def generate_embeddings(payload: DocumentIngest):
    """Generates a 384-dimensional vector embedding for a given text payload."""
    try:
        vector = embeddings_model.embed_query(payload.text)
        return {
            "text": payload.text,
            "vector_dimensions": len(vector),
            "embedding_sample": vector[:5]  # Returns first 5 floats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))