from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="llama3.2")
template = """
You are an expert in answering questions about rail freight operations, infrastructure, safety, efficiency, and validation procedures.

Here are some titles: {railcontext}

Here is the question to answer: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model


while True:
    print("\n\n------------------------------------------")
    print("Welcome to the Rail Freight Q&A Test Bot! This is just a test bot, so it may not be the best.")
    print("-------------------------------------------")
    question = input("Ask away! (press q to quit) ")
    if question == "q":
        break
    
    reviews = retriever.invoke(question)    
    result = chain.invoke({"railcontext": reviews, "question": question})
    print(result)

