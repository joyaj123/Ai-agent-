from langchain_ollama import OllamaEmbeddings #vector search , take text and hawelo la vector
from langchain_chroma import Chroma
from langchain_core.documents import Document 
import os
import pandas as pd #to easily read in our csv file

df = pd.read_csv("ai_search.csv")#read the csv file  (dataframe)
embddings = OllamaEmbeddings(model = "mxbai-embed-large")  #this model convert text into vector 


db_location = "./chrome_langchain_db" #folder were we store our database
add_documents = not os.path.exists(db_location)#check if this database exists

#if the location doesn't exists prepare all our data by converting it into documents
if add_documents :
    documents =[]
    ids = []
     
     #iterates through our rows in our csv file to access various entries
    for i,row in df.iterrows() : 
        document = Document( #ollama document
            page_content=row["Title"]+ " "+row["explanation"], #what we will be actually vectorizing and what will be looking up
            metadata={"rating": row["Rating"],"date": row["Date"]},#additional info we will grab along the doc w won't be quering based on this 
            id=str(i)
        )
        ids.append(str(i))
        documents.append(document)
#initialize the vector store
vector_store=Chroma(
    collection_name="ai_seach",
    persist_directory=db_location,
    embedding_function=embddings
)
#add data into the vector store by adding the documents
if add_documents:
    vector_store.add_documents(documents=documents,ids=ids)
#Converts the vector store into a retriever object.
retriever= vector_store.as_retriever(
    search_kwargs={"k":5}
)