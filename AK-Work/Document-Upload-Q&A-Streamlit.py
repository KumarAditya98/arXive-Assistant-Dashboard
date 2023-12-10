import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import  RecursiveCharacterTextSplitter
#from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.embeddings import LlamaCppEmbeddings
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import pickle
import os
from langchain.llms import HuggingFacePipeline
from langchain.chains.question_answering import load_qa_chain
from unstructured.partition.auto import partition

#with st.sidebar:
    #st.title("Document Upload - Q&A Chat App")

def main():
    st.title("Upload Your Document in PDF format.")
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        store_name = pdf.name[:-4]
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl","rb") as f:
                VectorStore = pickle.load(f)
                st.write("Embedding loaded from disk since similar file exxists.")
        else:
            embedding = HuggingFaceEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embedding)
            with open(f"{store_name}.pkl","wb") as f:
                pickle.dump(VectorStore,f)

        query = st.text_input("Ask question about your PDF Document:")
        st.write(query)
        if query:
            docs = VectorStore.similarity_search(query, k=5)
            llm = HuggingFacePipeline.from_model_id(model_id="google/flan-t5-large",task = "text2text-generation",model_kwargs={"temperature": 0, "max_length": 500}, device=0)
            chain = load_qa_chain(llm=llm,chain_type="stuff")
            response = chain.run(input_documents = docs,question=query)
            st.write(response)


if __name__ == '__main__':
    main()