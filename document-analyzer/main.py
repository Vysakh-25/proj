from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer,util

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
chunks_embedding = model.encode(chunks)

query = "what languages does he know?"

top_k = 3

def retrieve(query):
    query_emb =  model.encode(query)
    score = util.cos_sim(query_emb,chunks_embedding)
    top_indices = score[0].argsort(descending=True)[:top_k]
    print(top_indices)
    best_index = score.argmax()
    return top_indices

results = retrieve(query)

for idx in results:
    print("\n---Match---")
    print(chunks[idx])
    