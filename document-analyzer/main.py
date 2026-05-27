from PyPDF2 import PdfReader

reader = PdfReader("sample.pdf")

text = ""
for page in reader.pages:
    text += page.extract_text()

chunk_size = 300
chunks = []
for i in range(0, len(text), chunk_size):
    chunk = text[i:i + chunk_size]
    chunks.append(chunk)

for j in range(len(chunks)):
    print("\n---para{}---".format(j))
    print(chunks[j])
print("\n\n length of chunks:",len(chunks))