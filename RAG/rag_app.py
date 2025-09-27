import os
import sys
import base64
import docx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import openai
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

load_dotenv()

OPENAI_KEY = os.getenv("OPENAI_KEY")
OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT")
GPT_DEPLOYMENT = os.getenv("GPT_DEPLOYMENT")

SEARCH_ENDPOINT = os.getenv("SEARCH_ENDPOINT")
SEARCH_KEY = os.getenv("SEARCH_KEY")
INDEX_NAME = os.getenv("INDEX_NAME")

LOCAL_FOLDER = os.getenv("LOCAL_FOLDER")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))

openai.api_type = "azure"
openai.api_version = "2025-01-01-preview"
openai.api_key = OPENAI_KEY
openai.azure_endpoint = OPENAI_ENDPOINT


search_client = SearchClient(
    endpoint=SEARCH_ENDPOINT,
    index_name=INDEX_NAME,
    credential=AzureKeyCredential(SEARCH_KEY)
)


def safe_key(filename, chunk_num=None):
    key = f"{filename}_{chunk_num}" if chunk_num is not None else filename
    key_bytes = base64.urlsafe_b64encode(key.encode("utf-8"))
    return key_bytes.decode("utf-8").rstrip("=")

def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks

def load_docx_files(folder=LOCAL_FOLDER):
    docs = []
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Folder '{folder}' not found!")
    for filename in os.listdir(folder):
        if filename.endswith(".docx") and not filename.startswith("~$"):
            path = os.path.join(folder, filename)
            doc = docx.Document(path)
            full_text = "\n".join([p.text for p in doc.paragraphs])
            for idx, chunk in enumerate(chunk_text(full_text)):
                docs.append({"id": safe_key(filename, idx), "content": chunk})
    return docs

def upload_docs_to_search(docs):
    if not docs:
        print("No documents to upload.")
        return
    clean_docs = [{"id": d["id"], "content": d["content"]} for d in docs]
    result = search_client.upload_documents(documents=clean_docs)
    print(f"Uploaded {len(result)} document chunks to Azure Search.")

def get_top_matches(query, top_k=3):
    results = search_client.search(search_text=query, top=top_k)
    return [(r.get("content"), r.get("id")) for r in results]

def generate_answer_with_context(question, top_matches):
    if not top_matches:
        return "I couldn't find relevant information in the documents, but according to my knowledge..."
    context_text = "\n\n".join([f"{i+1}. {para}" for i, (para, _) in enumerate(top_matches)])
    prompt = f"""
You are an HR assistant. Use the following policy documents to answer the question.
Do not make up answers that are not in the documents.

Policies:
{context_text}

Question: {question}

Answer:
"""
    response = openai.chat.completions.create(
        model=GPT_DEPLOYMENT,
        messages=[
            {"role": "system", "content": "You are a helpful HR assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=400,
        temperature=0
    )
    return response.choices[0].message.content.strip()

#FastAPI
app = FastAPI(title="RAG HR Assistant API")

class QuestionRequest(BaseModel):
    question: str
    top_k: int = 3

@app.post("/ask")
def ask_question(request: QuestionRequest):
    top_matches = get_top_matches(request.question, top_k=request.top_k)
    answer = generate_answer_with_context(request.question, top_matches)
    return {
        "question": request.question,
        "answer": answer,
        "sources": [source for _, source in top_matches]
    }

@app.post("/upload_docs")
def upload_documents():
    try:
        docs = load_docx_files()
        upload_docs_to_search(docs)
        return {"status": "success", "uploaded_chunks": len(docs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# CLI
def run_cli():
    print("Loading and chunking documents...")
    docs = load_docx_files()
    print(f"Loaded {len(docs)} chunks.")
    upload_docs_to_search(docs)
    print("\nRAG system ready! Ask questions (type 'exit' to quit).")
    while True:
        try:
            user_query = input("\nYour question: ")
        except EOFError:
            print("\nInput not supported in this environment. Exiting CLI.")
            break
        if user_query.lower() == "exit":
            break
        top_matches = get_top_matches(user_query)
        answer = generate_answer_with_context(user_query, top_matches)
        print("\nAnswer:\n", answer)
        print("\nSources:", [source for _, source in top_matches])
        print("="*50)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].lower() == "cli":
        run_cli()
    else:
        import uvicorn
        uvicorn.run("rag_app:app", host="127.0.0.1", port=8000, reload=True)
