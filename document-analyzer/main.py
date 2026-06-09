from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer,util
import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter

reader = PdfReader("sample.pdf")

text = ""
for page in reader.pages:
    text += page.extract_text()

chunk_size = 300
overlap = 50
chunks = []
for i in range(0, len(text), chunk_size - overlap):
    chunk = text[i:i + chunk_size]
    chunks.append(chunk)

"""for j in range(len(chunks)):
    print("\n---para{}---".format(j))
    print(chunks[j])
print("\n\n length of chunks:",len(chunks))"""


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