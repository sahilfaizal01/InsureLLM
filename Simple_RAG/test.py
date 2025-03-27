from langchain_core.callbacks import StdOutCallbackHandler
import os 
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


embeddings = OpenAIEmbeddings()
load_dotenv(override=True)
llm = ChatOpenAI(temperature=0.7, model_name=os.environ['MODEL'], api_key=os.environ['OPENAI_API_KEY'])
vector_store = Chroma(persist_directory=os.environ['db_name'], embedding_function=embeddings)

memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

retriever = vector_store.as_retriever()

conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory, callbacks=[StdOutCallbackHandler()])

query = "Who received the prestigious IIOTY award in 2023?"
result = conversation_chain.invoke({"question": query})
answer = result["answer"]
print("\nAnswer:", answer)

# having difficulty in finding the answer to the query