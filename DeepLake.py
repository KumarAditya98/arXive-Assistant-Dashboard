#%%

import os
import json
from tqdm import tqdm
from langchain.document_loaders import ArxivLoader
from multiprocessing import Pool
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.deeplake import DeepLake
from langchain.text_splitter import CharacterTextSplitter

#----------------------------------------------
# dependencies

# pip install langchain
# pip install pypdf
# pip install gpt4all
# pip install deeplake[enterprise]
# pip install -U pyopenssl cryptography

os.environ['ACTIVELOOP_TOKEN'] = "eyJhbGciOiJIUzUxMiIsImlhdCI6MTcwMDQyMTE5MSwiZXhwIjoxNzMyMDQzNTY4fQ.eyJpZCI6InR3YWxsZXR0In0.fmyqEm0BgY6F_4ax0aJjJWiT0_CMrixfLTE-T-xGvKwX_q8O4VtlemxpJD7J0wO6Siqr0TtDForm_jo1jLf-Zg"

file = open("arxiv-metadata.json")

data = json.load(file)

alist = [data[i]['id'] for i in range(len(data)) if data[i]['categories'] == 'cs.AI']

docs = []

def f(id):
    return ArxivLoader(query=id, doc_content_chars_max=2).load()

if __name__ == "__main__":
    with Pool(4) as p:
        for result in tqdm(p.imap(f, alist), total=len(alist)):
            try:
                docs.append(result[0])
            except:
                pass

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings()

db = DeepLake(dataset_path="./deeplake/",
              embedding_function=embeddings, 
              token=os.environ['ACTIVELOOP_TOKEN'])

db.add_documents(texts)
# %%
