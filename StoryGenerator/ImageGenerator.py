import ollama
import bs4
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

website = 'https://www.marktechpost.com/2024/04/21/coconut-a-high-quality-large-scale-dataset-for-next-gen-segmentation-models/'

# Load the data
loader = WebBaseLoader(
    web_paths=(f"{website}",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("td-post-content tagdiv-type", "td-post-header", "td-post-title")
        )
    ),
)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
print(splits)
# Create Ollama embeddings and vector store
embeddings = OllamaEmbeddings(model="llama3")
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)


# Call Ollama Llama3 model
def ollama_llm(question, context):
    formatted_prompt = f"Question: {question}\n\nContext: {context}"
    response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': formatted_prompt}])
    return response['message']['content']

# RAG Setup
retriever = vectorstore.as_retriever()
def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
def rag_chain(question):
    retrieved_docs = retriever.invoke(question)
    formatted_context = combine_docs(retrieved_docs)
    return ollama_llm(question, formatted_context)

# Use the RAG App
Question = "What is COCONut?"
result = rag_chain(f"{Question}")
print(f"Question : {Question}")
print(f"Response : {result}")