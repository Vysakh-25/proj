from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

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

for j in range(len(chunks)):
    print("\n---para{}---".format(j))
    print(chunks[j])
print("\n\n length of chunks:",len(chunks))

model = SentenceTransformer("all-MiniLM-L6-v2")
embedding = model.encode(chunks[0])

query = "tell me about the handwritten digit recognition project"
query_emb =  model.encode(query)