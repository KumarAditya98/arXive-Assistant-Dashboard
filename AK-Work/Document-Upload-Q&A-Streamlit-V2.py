import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import  RecursiveCharacterTextSplitter
#from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.embeddings import LlamaCppEmbeddings
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import HuggingFaceHub
import pickle
import os
from langchain.llms import HuggingFacePipeline
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory
#from unstructured.partition.auto import partition
from langchain.chains import ConversationalRetrievalChain
from transformers import AutoTokenizer
# Load model directly
from transformers import AutoTokenizer, AutoModelForDocumentQuestionAnswering, pipeline, AutoModel
from langchain.chains import RetrievalQA
from langchain import HuggingFaceHub
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory
import langchain
import chromadb

import os
import getpass

from langchain.document_loaders import PyPDFLoader  #document loader: https://python.langchain.com/docs/modules/data_connection/document_loaders
from langchain.text_splitter import RecursiveCharacterTextSplitter  #document transformer: text splitter for chunking
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import PromptTemplate
from langchain.vectorstores import Chroma #vector store
from langchain import HuggingFaceHub  #model hub
from langchain.chains import RetrievalQA

from langchain.memory import ConversationBufferMemory

#loading the API key
import getpass
import os


#loading the API key
os.environ['HUGGING_FACE_HUB_API_KEY'] = getpass.getpass('Hugging face api key:')


#with st.sidebar:
    #st.title("Document Upload - Q&A Chat App")
template = """
Use the following context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) to answer the question:
------
<ctx>
{context}
</ctx>
------
<hs>
{history}
</hs>
------
{question}
Answer:
"""
def main():
    st.title("Upload Your Document in PDF format.")
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    if pdf is not None:
        #pdf_reader = PdfReader(pdf)
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        embeddings = HuggingFaceEmbeddings()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len)
        docs = splitter.split_text(text)

        store_name = pdf.name[:-4]
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl","rb") as f:
                doc_search = pickle.load(f)
        else:
            doc_search = Chroma.from_documents(docs, embeddings)
            with open(f"{store_name}.pkl","wb") as f:
                pickle.dump(doc_search,f)


        query = st.text_input("Ask question about your PDF Document:")
        if query:
            st.write(f"You: ", query)
            docs = doc_search.similarity_search(query, k=3)
            repo_id = 'lmsys/fastchat-t5-3b-v1.0'  # has 3B parameters: https://huggingface.co/lmsys/fastchat-t5-3b-v1.0
            llm = HuggingFaceHub(huggingfacehub_api_token=os.environ['HUGGING_FACE_HUB_API_KEY'],
                                 repo_id=repo_id,
                                 model_kwargs={'temperature': 1e-10, 'max_length': 1000})
            prompt = PromptTemplate(
                input_variables=["history", "context", "question"],
                template=template,
            )
            memory = ConversationBufferMemory(
                memory_key="history",
                input_key="question"
            )

            retrieval_chain = RetrievalQA.from_chain_type(llm,
                                                          chain_type='stuff',
                                                          retriever=doc_search.as_retriever(),
                                                          chain_type_kwargs={
                                                              "prompt": prompt,
                                                              "memory": memory
                                                          })
            retrieval_chain.run(query)
            st.write("Chatbot:",memory.load_memory_variables({'history'}) )



if __name__ == '__main__':
    main()