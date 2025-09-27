!pip install langchain langchain-community
!pip install faiss-cpu
!pip install pypdf python-docx
!pip install sentence-transformers
!pip install transformers

"""Upload File in Colab"""

from google.colab import files
uploaded = files.upload()

file_path = list(uploaded.keys())[0]
print("Uploaded:", file_path)

"""Load and Split Document"""

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

if file_path.endswith('.pdf'):
  loader = PyPDFLoader(file_path)
elif file_path.endswith('.docx') or file_path.endswith('.doc'):
  loader = Docx2txtLoader(file_path)
else:
  loader = TextLoader(file_path)

docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

documents = splitter.split_documents(docs)
print(f"Total Chunks: {len(documents)}")

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = FAISS.from_documents(documents, embeddings)

from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline

flan_pipeline=pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_length=512
)
llm=HuggingFacePipeline(pipeline=flan_pipeline)

"""Retrieval Q and A"""

from langchain.chains import RetrievalQA

qa=RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k":3}),
    chain_type="stuff"
)
query="Give me a short summary of the document"
print(qa.run(query))

while True:
  q = input("Ask a question (or 'exit'): ")
  if q.lower() == 'exit':
    break
  print('Answer: ',qa.run(q))

