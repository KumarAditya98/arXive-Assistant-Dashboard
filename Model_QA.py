#%%

import streamlit as st
from langchain.vectorstores.deeplake import DeepLake
from langchain.embeddings import HuggingFaceEmbeddings
import torch
from transformers import pipeline
from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFacePipeline

db = DeepLake(dataset_path="./deeplake/",
              read_only=True)

embeddings = HuggingFaceEmbeddings()

generate_text = pipeline(model="databricks/dolly-v2-7b", 
                         torch_dtype=torch.bfloat16,
                         device_map='auto',
                         trust_remote_code=True, 
                         return_full_text=True)

llm = HuggingFacePipeline(pipeline=generate_text)

prompt = PromptTemplate(
    input_variables=["instruction"],
    template="{instruction}")

prompt_with_context = PromptTemplate(
    input_variables=["instruction", "context"],
    template="{instruction}\n\nInput:\n{context}")

def generate(question = None):
    
    context = db.similarity_search(query=question,
                                k=2,
                                embedding_function = embeddings)

    llm_context_chain = LLMChain(llm = llm,
                                prompt=prompt_with_context)

    answer = llm_context_chain.predict(instruction=f"{question}", 
                                    context=context).lstrip()
    
    return answer 

def side_bar():
    with st.sidebar:
        if st.button('Clear Chat'):
            # Clear the chat history
            st.session_state.messages = []
            
# %%
