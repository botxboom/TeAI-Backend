from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.memory import ConversationSummaryMemory
from flask import Flask, request, jsonify, render_template,Response
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'document/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'pdf'}

memory = ConversationSummaryMemory(llm=Ollama(model="gemma2:2b"), memory_key='history',return_messages=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload-pdf')
def upload_file_form():
    return render_template('upload.html')

@app.route('/')
def chat():
    return render_template('chat.html')

__retriever = None
__prompt_template = """You are an AI designed to teach data structures and algorithms using the Socratic method. Your goal is to guide learners through questioning, helping them arrive at solutions on their own rather than providing direct answers.

When a learner presents a problem or question, follow these guidelines:

    Ask Clarifying Questions: Start by encouraging the learner to articulate their understanding of the problem. Ask them to explain what they know about the data structures or algorithms involved.

    Encourage Exploration: Prompt the learner to think about different approaches. Questions like “What have you tried so far?” or “What other methods do you think might work?” can help them explore various angles.

    Guide Problem-Solving: When the learner gets stuck, ask them to break the problem down into smaller parts. Questions like “What is the first step you need to take?” or “How can you simplify this problem?” will help them think critically.

    Challenge Assumptions: If the learner presents an idea, ask them to justify it. For example, “Why do you think that approach would work?” or “What would happen if you changed that part?”

    Encourage Reflection: After guiding them toward a solution, ask them what they learned during the process. Questions like “What did you find most challenging?” or “How can you apply this thinking to future problems?” help reinforce their learning.

Remember to keep the tone supportive and encouraging, fostering a sense of curiosity and self-discovery."""


@app.route('/document', methods=['POST'])
def document():

    global __retriever
    file = request.files['file']
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        loader = PyPDFLoader(file_path)
        doc = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
        documents = splitter.split_documents(doc)
        embedding = OllamaEmbeddings(model = 'gemma2:2b')
        db = FAISS.from_documents(documents,embedding)
        __retriever = db.as_retriever()
        
    return Response(status=204)
    

@app.route('/chatbot', methods=['POST'])
def chatbot():

    global memory
    user_input = request.get_json()
    input = user_input['text']
    llm = Ollama(model = "gemma2:2b")
    output_parser = StrOutputParser()
    history = memory.load_memory_variables({}).get('history', '')


    if __retriever:
        prompt = ChatPromptTemplate.from_template("Prompt: {text} <context> {context}</context> Input: {input} History Summary: {history}")
        
        doc_chain = create_stuff_documents_chain(llm,prompt)
        retrieval_chain =  create_retrieval_chain(__retriever, doc_chain)
        result = retrieval_chain.invoke({"input":input,"text":__prompt_template, "history": history})
        response = result['answer']
        
    else:
        prompt = ChatPromptTemplate.from_template("Prompt: {text} Input: {input} History Summary: {history}.")

        chain = prompt | llm | output_parser
        response = chain.invoke({"input":input,"text":__prompt_template, "history": history})

    memory.save_context({'input': input}, {'output': response})


    return jsonify({"response":response})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
 


