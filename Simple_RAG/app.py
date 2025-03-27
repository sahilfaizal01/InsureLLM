import os 
import gradio as gr
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

embeddings = OpenAIEmbeddings()
load_dotenv(override=True)
vector_store = Chroma(persist_directory=os.environ['db_name'], embedding_function=embeddings)

def chat(message, history):
    result = conversation_chain.invoke({"question": message})
    return result["answer"]

# create a new Chat with OpenAI
llm = ChatOpenAI(temperature=0.7, model_name=os.environ['MODEL'], api_key=os.environ['OPENAI_API_KEY'])
# set up the conversation memory for the chat
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
# the retriever is an abstraction over the VectorStore that will be used during RAG
#retriever = vector_store.as_retriever()
retriever = vector_store.as_retriever(search_kwargs={"k": 25}) # after the search, we will rerank the results with the LLM
# set up the conversation chain with the GPT 4o-mini LLM, the vector store and memory
conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

#gradio ui
view = gr.ChatInterface(chat, type="messages").launch(inbrowser=True)
