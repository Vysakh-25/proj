from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer,util
import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter

reader = PdfReader("sample.pdf")

text = ""
for page in reader.pages:
    text += page.extract_text()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)
chunks = text_splitter.split_text(text)

model = SentenceTransformer("all-MiniLM-L6-v2")
chunk_embeddings = model.encode(chunks)
index = faiss.IndexFlatL2(384)
index.add(chunk_embeddings)
#print(index.ntotal)

query = input("Ask a question: ")

top_k = 3

def retrieve(query):
    query_emb =  model.encode(query)
    query_emb = query_emb.reshape(1,-1)
    distances,indices = index.search(query_emb,top_k)
    return indices[0]

results = retrieve(query)

context = ""

for idx in results:
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