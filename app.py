from flask import Flask, render_template, request
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

app = Flask(__name__)

model = OllamaLLM(model="llama3.2")

template = """
You are an expert in different AI search algorithms.

Here are some relevant reviews:
{reviews}

Here is the question to answer:
{question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model


@app.route("/", methods=["GET", "POST"])
def home():
    answer = None

    if request.method == "POST":
        question = request.form.get("question")

        if question:
            reviews = retriever.invoke(question)
            answer = chain.invoke({
                "reviews": reviews,
                "question": question
            })

    return render_template("index.html", answer=answer)


if __name__ == "__main__":
    app.run(debug=False)