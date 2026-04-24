from s3_helper import download_and_load_from_s3
from agent_core import build_vector_store, load_vector_store, get_llm

BUCKET_NAME = "langchain-testing-660463065978-us-east-1-an"

# === Opción A: Construir desde cero (usa APIs de S3 y Gemini) ===
# Descomentar estas líneas para reconstruir la base de datos
# (por ejemplo, al meter archivos nuevos al bucket)

documents = download_and_load_from_s3(BUCKET_NAME)
vector_store = build_vector_store(documents)

# === Opción B: Cargar base existente (instantáneo, sin costo) ===
# Usar esta opción para no tener que generar los embeddings de nuevo.

# vector_store = load_vector_store()

# === Prueba rápida ===
query = "Qué información hay en tus documentos?"
resultados = vector_store.similarity_search_with_score(query)

for doc, score in resultados:
    print(f"[Score: {score:.4f}] {doc.page_content[:100]}...")