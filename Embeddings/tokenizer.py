import json
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter


text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)


client = chromadb.PersistentClient(path="vector_store")

collection = client.create_collection(name= "hf_blogs_vectors")

with open("Dataset/hf_blogs_data.json") as file:
    data = json.load(file)

id = 1
for i, d in enumerate(data):
    print(f"Document {i} in progress")
    chunks = text_splitter.split_text(d['Text'])
    for c in chunks:
        collection.add(ids=f"id{id}", documents=c, metadatas={"link" : d['link'], "Title" : d['Title'], "Publish Date" : d['Publish Date']})
        id += 1
