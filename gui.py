import gradio as gr
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

# LLM and prompt setup
model = OllamaLLM(model="llama3.2")
template = """
You are an expert in answering questions about rail freight operations, infrastructure, safety, efficiency, and validation procedures.

Here are some titles: {railcontext}

Here is the question to answer: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def answer_question(question):
    if question.strip().lower() == 'q':
        return "Goodbye!"
    
    context = retriever.invoke(question)
    response = chain.invoke({"railcontext": context, "question": question})
    return str(response)



with gr.Blocks() as demo:
    gr.Markdown("""# Rail Freight Q&A Test Bot
Welcome! Ask any question about rail freight safety, operations, infrastructure, and more.
This is just a test bot - answers may be weird in quality, and my Dataset is still small. Type your question below:
""")

    with gr.Row():
        output_box = gr.Textbox(label="Answer", lines=10)
        question_box = gr.Textbox(placeholder="Ask your question here...", label="Your Question", lines=1)
        question_box.submit(fn=answer_question, inputs=question_box, outputs=output_box)

    
    with gr.Row():
        submit_btn = gr.Button("Submit my Inquiry")
        clear_btn = gr.Button("Clear the Chat!")

    

    submit_btn.click(fn=answer_question, inputs=question_box, outputs=output_box)
    clear_btn.click(fn=lambda: ("", ""), inputs=[], outputs=[question_box, output_box])

demo.launch()
