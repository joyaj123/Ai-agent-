from langchain_ollama import OllamaEmbeddings #vector search , take text and hawelo la vector
from langchain_chroma import Chroma
from langchain_core.documents import Document 
import os
import pandas as pd 

df = pd.read_csv("realistic_restaurant_reviews.csv")#dataframe
embddings = OllamaEmbeddings(model = "mxbai-embed-large")  


db_location = "./chrome_langchain_db" #folder were we store our database
add_documents = not os.path.exists(db_location)

if add_documents :
    documents =[]
    ids = []

    """for i,row in df.iterrows() : 
        documents = Document(
            page_content= 
        )

        """