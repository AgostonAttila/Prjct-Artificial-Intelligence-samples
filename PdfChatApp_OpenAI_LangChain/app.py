import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os 

with st.sidebar:
    st.title('LLM Chat App')
    st.markdown('Welcome to the LLM Chat App! Please enter your name and start chatting with the bot.') 
    st.write('Made with love')


def main():
    st.header('Chat with PDF:')

    load_dotenv()

    #upload PDF file
    pdf = st.file_uploader('Upload your PDF', type=['pdf'])
    st.write(pdf.name)
    if pdf is not None:
        st.write('PDF uploaded successfully!')
        pdf_reader = PdfReader(pdf)

        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )
        chunks = text_splitter.split_text(text=text)
      
        #embeddings
    
        store_name = pdf.name[:-4]

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
            st.write("Loaded existing vector store")
        else:
            with open(f"{store_name}.pkl", "wb") as f:
                embeddings = OpenAIEmbeddings()
                VectorStore = FAISS.from_texts(chunks,embedding=embeddings)
                pickle.dump(VectorStore, f)

        #User query
        query = st.text_input('Enter your query:')
        st.write(query)

        if query:
            docs = VectorStore.similarity_search(query = query,k=3)

            llm = OpenAI(model_name="gpt-3.5-turbo")
            chain = load_qa_chain(llm = llm, chain_type="stuff")
            with get_openai_callback() as cb:
                  response = chain.run(input_documents=docs, question= query)
                  print(cb)
            st.write(response)


if __name__ == '__main__':
    main()