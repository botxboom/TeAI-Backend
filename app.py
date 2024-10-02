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


@app.route('/')
def upload_file_form():
    return render_template('upload.html')

@app.route('/chat')
def chat():
    return render_template('chat.html')

__retriever = None
__text = None


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

        print(documents[:5])


        embedding = OllamaEmbeddings(model = 'gemma2:2b')
        db = FAISS.from_documents(documents,embedding)
        __retriever = db.as_retriever()
        print('Embeddings generated')
        
    return Response(status=204)

@app.route('/prompt',methods=['POST'])
def prompt():

    global __text
    user_prompt = request.get_json()
    __text = user_prompt['text']
    print(__text)

    return Response(status=204)
    

@app.route('/chatbot', methods=['POST'])
def chatbot():

    global memory
    user_input = request.get_json()
    input = user_input['text']
    llm = Ollama(model = "gemma2:2b")
    output_parser = StrOutputParser()


    if __retriever:
        prompt = ChatPromptTemplate.from_template("Prompt: {text} <context> {context}</context> Input: {input} History Summary: {history}")
        
        doc_chain = create_stuff_documents_chain(llm,prompt)
        retrieval_chain =  create_retrieval_chain(__retriever, doc_chain)
        result = retrieval_chain.invoke({"input":input,"text":__text, "history": memory.load_memory_variables({})['history']})
        response = result['answer']
        memory.save_context({'input':input} , {'output': response})
    else:
        prompt = ChatPromptTemplate.from_template("Prompt: {text} Input: {input} History Summary: {history}.")

        chain = prompt | llm | output_parser
        response = chain.invoke({"input":input,"text":__text, "history": memory.load_memory_variables({})['history']})
        memory.save_context({'input':input} , {'output': response})


    return jsonify({"response":response})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
 


