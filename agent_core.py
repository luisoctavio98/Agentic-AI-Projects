import os
import time
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS

# Cargar variables de entorno
load_dotenv()

def get_text_splitter():
    """Parte el texto en chunks de manera recursiva."""
    return RecursiveCharacterTextSplitter(
        chunk_size=200, 
        chunk_overlap=20
    )

def get_embeddings_model():
    """Inicializa y devuelve el modelo de embeddings de Gemini."""
    gemini_key = os.getenv("GEMINI_KEY")
    if not gemini_key:
        raise ValueError("No se encontró GEMINI_KEY en el archivo .env")
        
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001", 
        google_api_key=gemini_key
    )

def get_llm():
    """Inicializa el LLM conectado a Groq usando la interfaz de OpenAI."""
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise ValueError("No se encontró GROQ_API_KEY en el archivo .env")
        
    return ChatOpenAI(
        model_name="openai/gpt-oss-120b", # Cualquier modelo de Groq
        temperature=0.5,
        openai_api_key=groq_key,
        base_url="https://api.groq.com/openai/v1"
    )

def build_vector_store(documents, index_path="faiss_index"):
    """
    Toma los documentos, los parte en chunks, calcula los embeddings 
    y crea la base de datos FAISS.
    """
    print("Partiendo los documentos en chunks...")
    splitter = get_text_splitter()
    chunks = splitter.split_documents(documents)
    
    print(f"Calculando embeddings para {len(chunks)} chunks...")
    embedding_model = get_embeddings_model()
    
    # Extraemos el texto para los embeddings
    text_contents = [chunk.page_content for chunk in chunks]
    
    # Langchain FAISS from_documents hace el batching automáticamente bajo el capó
    # pero como Gemini tiene límites estrictos, tu lógica manual era excelente.
    # Aquí la refactorizamos para que sea más limpia:
    
    embeddings = []
    batch_size = 100
    batches = [text_contents[i:i + batch_size] for i in range(0, len(text_contents), batch_size)]
    
    for i, batch in enumerate(batches):
        print(f"Procesando batch {i+1} de {len(batches)}")
        result = embedding_model.embed_documents(batch)
        embeddings.extend(result)
        # Respetar el límite de Gemini (evitar error 429 Too Many Requests)
        if i < len(batches) - 1:
            time.sleep(60) 
            
    # Unir textos con sus embeddings calculados
    text_embeddings = list(zip(text_contents, embeddings))
    
    print("Guardando base de datos FAISS localmente...")
    vector_store = FAISS.from_embeddings(
        text_embeddings=text_embeddings,
        embedding=embedding_model
    )
    vector_store.save_local(index_path)
    print(f"Base de datos guardada localmente en '\\{index_path}'.")
    
    return vector_store

def load_vector_store(index_path="faiss_index"):
    """Carga una base de datos FAISS existente desde el disco."""
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"No se encontró el índice en '\\{index_path}'.")
    
    print(f"Cargando base de datos existente desde '\\{index_path}'...")
    embedding_model = get_embeddings_model()
    
    # allow_dangerous_deserialization=True es un requerimiento de seguridad nuevo
    # en LangChain al cargar archivos locales tipo pickle (.pkl)
    return FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)
