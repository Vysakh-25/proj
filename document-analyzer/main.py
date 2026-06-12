from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer,util
from sentence_transformers import CrossEncoder
import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pickle

model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
index = faiss.read_index("my_index.faiss")
chunks = pickle.load(open("chunks.pkl","rb"))
#print(index.ntotal)

query = input("Ask a question: ")

def retrieve(query,top_k = 3):
    query_emb =  model.encode(query)
    query_emb = query_emb.reshape(1,-1)
    distances,indices = index.search(query_emb,top_k)
    print(distances)
    return distances[0],indices[0]

distances,results = retrieve(query)
context = ""

for distance, idx in zip(distances, results):
    context += chunks[idx] + "\n\n"

prompt = f"""
Use only the information provided in the context below to answer the question.

If the answer is not present in the context, say:
"I could not find that information in the document."

Context:
{context}

Question:
{query}

Answer:
"""

print(prompt)