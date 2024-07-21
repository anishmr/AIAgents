
import os
from dotenv import load_dotenv, find_dotenv

from yotube_utils import YoutubeUtilTool



load_dotenv(find_dotenv())
API_KEY = os.getenv('YOUTUBE_API_KEY')

yt_tool = YoutubeUtilTool(api_key= API_KEY)

search_query = "full stack javascript developer roadmap 2024"
search_results = yt_tool.run(search_query)
print(search_results)
transcripts = ''
for search_result in search_results:
    videoUrl = search_result['url']
    transcript = yt_tool.extractTranscript(videoUrl)
    #loder = YoutubeLoader.from_youtube_url(videoUrl)
    #transcript = loder.load()
    transcripts = transcript




#Load date from youtube/web


#Embed using 
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
#from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=512, chunk_overlap=0.2
)

docs = text_splitter.split_documents(transcripts)
print(docs[0])

#save to Chrome Vector db
vectorstore = Chroma.from_documents(
    documents=docs,
    collection_name="reg-chroma",
    embedding=OllamaEmbeddings(model='nomic-embed-text'),
    persist_directory="./chroma_db"
)


#retriver
retriever = vectorstore.as_retriever()
print(retriever)

from langchain_community.llms import Ollama

from langchain_core.runnables import RunnablePassthrough 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

llm = Ollama(model='llama3')

template = """Answer the question based only on the following context:
{context}
Question:{question}
"""

prompt = ChatPromptTemplate.from_template(template)
rag_chain = (
    {"context":retriever, "question": RunnablePassthrough()}
    | prompt
    | llm 
    | StrOutputParser()
)

print(rag_chain.invoke("Web developer roadmap in 10 bullet points"))











#---------------------------------

#Retriver Node








