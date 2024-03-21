import gradio as gr
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models.anyscale import ChatAnyscale

from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)    
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize vector store
vdb = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)

#setting up llm
ANYSCALE_MODEL_NAME = "meta-llama/Llama-2-70b-chat-hf"
#defining LLm
llm = ChatAnyscale(model_name=ANYSCALE_MODEL_NAME)

#chain
from langchain.chains import RetrievalQA
#memory
from langchain.memory import ConversationBufferMemory
#prompt templatte
from langchain.prompts import PromptTemplate

#memory
from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

#prompt
template ="""
your name is 'catbot'\
you are expected to answer to user questions\
anways answer in polite and friendly way\
you are a helpful assistant, always help customers, do not hallucinate\
if you dont know this answer to customer querry or customer asking out of context question then always remaind him about what you are and ask questions back to customer to guide him to right track\
if customer is asking irrevevent question then tell him about your purpose\
do not answer to question too lenghty\
your purpose as follows\
purpose:
A friendly chatbot to help users over their questions related to the ccontent.\
    CONTEXT: {context}
    QUESTION: {question}
    """
PROMPT = PromptTemplate(
        template=template, input_variables=["context", "chat_history", "question"]
    )
#retriever
retriever = vdb.as_retriever()
#chain+
chain = RetrievalQA.from_chain_type(llm=llm,
                                    chain_type="stuff",
                                    retriever=retriever,
                                    verbose=True,
                                    input_key="query",
                                    return_source_documents=True,
                                    chain_type_kwargs={"prompt": PROMPT})

def generate_answer(question, history=[]):
    result = chain({"query": question})
    return result["result"]

gr.ChatInterface(generate_answer).launch()





