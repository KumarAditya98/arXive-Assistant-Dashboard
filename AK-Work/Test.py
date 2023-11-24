from llama_index import VectorStoreIndex, SimpleDirectoryReader
import os
from llama_index import Document, SimpleDirectoryReader


os.environ['OPENAI_API_KEY'] = ''

documents = SimpleDirectoryReader('C:/Users/Aditya Kumar/PycharmProjects/Project-NLP/AK-Work/SampleData/').load_data()
index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()
response = query_engine.query("Summarize this document for me.")
print(response)
