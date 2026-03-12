from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate #run locallt
from vector import retriever
model = OllamaLLM(model="llama3.2") #precised the model 

template = """ 
You are an expert in different AI search algorithm :

Here are some relevant reviews : {reviews}

Here is the question to answer : {question}

""" 
prompt = ChatPromptTemplate.from_template(template) 
chain = prompt | model  # we link the prompt(template) to the model

while True : 
    print("\n\n -------------------------------------------")
    question = input("Ask your question : (q to quit ) :")
    print("\n\n")
    if question == "q" :
        break 
    reviews=retriever.invoke(question) #for vector search in the vector database (chroma)
    result = chain.invoke({"reviews" : reviews , "question": question })
    print(result)

