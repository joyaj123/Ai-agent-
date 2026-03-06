from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate #run locallt

model = OllamaLLM(model="llama3.2") #precised the model 

template = """ 
You are an expert in answering questions about pizza restaurant :

Here are some relevant reviews : {reviews}

Here is the question to answer : {question}


""" 
prompt = ChatPromptTemplate.from_template(template) 
chain = prompt | model  # we fice the prompt to the model

while True : 
    print("\n\n -------------------------------------------")
    question = input("Ask your question : (q to quit ) :")
    print("\n\n")
    if question == "q" :
        break 
    result = chain.invoke({"reviews" : [] , "question": question })
    print(result)

