from PyPDF2 import PdfReader

reader = PdfReader("sample.pdf")
print(len(reader.pages))