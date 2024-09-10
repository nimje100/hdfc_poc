import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
import os

# Load EMI information
def load_emi_info(file_path):
    with open(file_path, 'r') as file:
        return file.read()

# Split text into chunks
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n\n", "\n\n", "\n", " ", ""]
    )
    return splitter.split_text(text)

# Create vector store
def create_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

# Create conversation chain
def create_conversation_chain(vectorstore):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

# Handle user input
def handle_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

# Main function
def main():
    st.set_page_config(page_title="EMI Sales Agent Assistant", page_icon=":bank:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        emi_info = load_emi_info("emi_info.txt")
        text_chunks = split_text(emi_info)
        vectorstore = create_vectorstore(text_chunks)
        st.session_state.conversation = create_conversation_chain(vectorstore)

    st.header("EMI Sales Agent Assistant :bank:")
    user_question = st.text_input("Ask a question about EMIs:")

    if user_question:
        handle_user_input(user_question)

if __name__ == '__main__':
    main()