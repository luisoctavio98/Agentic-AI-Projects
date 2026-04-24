import os
import tempfile
from pathlib import Path
import boto3
from dotenv import load_dotenv

# Langchain Loaders
from langchain_community.document_loaders import (
    TextLoader,
    UnstructuredMarkdownLoader,
    CSVLoader,
    BSHTMLLoader,
    PyPDFLoader
)
from langchain_unstructured import UnstructuredLoader

# Cargar variables de entorno (lee el archivo .env)
load_dotenv()

def pick_loader(path):
    """Devuelve el loader de LangChain adecuado según la extensión del archivo."""
    ext = Path(path).suffix.lower()
    if ext == ".txt":
        return TextLoader(path, encoding="utf-8")
    if ext == ".md":
        return UnstructuredMarkdownLoader(path, encoding="utf-8")
    if ext == ".pdf":
        return PyPDFLoader(path)
    if ext in {".html", ".htm"}:
        return BSHTMLLoader(path, open_encoding="utf-8")
    if ext == ".csv":
        return CSVLoader(path, encoding="utf-8")
    return UnstructuredLoader([path])

def download_and_load_from_s3(bucket_name):
    """
    Descarga todos los archivos de un bucket S3 a un directorio temporal,
    los procesa usando LangChain, y devuelve la lista de documentos.
    """
    aws_access_key = os.getenv("AWS_ACCESS_KEY")
    aws_secret_key = os.getenv("AWS_SECRET_KEY")
    
    # Boto3 detecta llaves en el .env.
    if aws_access_key and aws_secret_key:
        s3 = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
        )
    else:
        s3 = boto3.client("s3")

    data = []
    
    print(f"Conectando al bucket: {bucket_name}")
    response = s3.list_objects_v2(Bucket=bucket_name)
    contents = response.get("Contents", [])
    
    if not contents:
        print("El bucket está vacío o no se encontraron archivos.")
        return data

    with tempfile.TemporaryDirectory() as tmpdir:
        for obj in contents:
            key = obj["Key"]
            path = os.path.join(tmpdir, os.path.basename(key))
            
            print(f"Procesando: {key}")
            s3.download_file(bucket_name, key, path)
            
            loader = pick_loader(path)
            data.extend(loader.load())
            
    print("Contenido extraído exitosamente.\n")
    return data
