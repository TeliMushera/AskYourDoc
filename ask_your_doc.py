from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

file_path = "sample.pdf"  # <-- replace with your file path (pdf, docx, or txt)

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

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(documents, embeddings)

flan_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_length=512
)
llm = HuggingFacePipeline(pipeline=flan_pipeline)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    chain_type="stuff"
)

query = "Give me a short summary of the document"
print("\n📄 Summary:", qa.run(query))

print("\n💬 Ask questions about the document (type 'exit' to quit)\n")
while True:
    q = input("You: ")
    if q.lower() == 'exit':
        break
    print("AI:", qa.run(q))
