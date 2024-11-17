from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from langchain.prompts import PromptTemplate
from docx import Document
import tiktoken
import faiss
import os
import re
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
load_dotenv()

# Token pricing constants
UNCACHED_INPUT_COST_PER_MILLION = 1.50
OUTPUT_COST_PER_MILLION = 2.00

# Initialize tokenizer
encoding = tiktoken.get_encoding("cl100k_base")

# Directory containing pre-uploaded documents
DOCUMENTS_DIR = os.path.dirname(os.path.abspath(__file__))  # Ensure this directory exists and contains .docx files

# Initialize conversation chain (global variable for this example)
conversation_chain = None
# Track user conversation count
user_question_count = {}

CONVERSATION_LIMIT = 10
# System prompt tailored for blog assistant


def clean_text(text):
    text = re.sub(r'(\w)\s(?=\w)', r'\1', text)  # Joins letters that were spaced out
    text = re.sub(r'\s+', ' ', text)  # Reduces multiple spaces to a single space
    return text.strip()

def estimate_tokens(text):
    return len(encoding.encode(text))

def get_text_docx(file_path):
    doc = Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs if para.text])
    return clean_text(text)

def get_text_and_filenames(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".docx"):
            file_path = os.path.join(directory, filename)
            text = get_text_docx(file_path)
            blog_title = os.path.splitext(filename)[0]
            documents.append((f"BLOG TITLE: {blog_title}\n\n{text}", blog_title))
    return documents

def get_chunks(documents):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    all_chunks = []
    for text, blog_title in documents:
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            processed_chunk = {
                "text": f"BLOG: {blog_title}\n\n{chunk}",
                "blog_title": blog_title
            }
            all_chunks.append(processed_chunk)
    return all_chunks

def get_vector(chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [{"blog_title": chunk["blog_title"]} for chunk in chunks]
    vectorstore = FAISS.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas
    )
    return vectorstore

def create_conversation_chain(vector_store):
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)  # Use a more cost-effective model like GPT-3.5
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')
    
    system_template ="""
You are a friendly and engaging intelligent assistant. Your responsibility is to provide helpful positive and engaging information and answer questions based on the context you already know(Imagine you are Oleon).

CURRENT BLOG CONTENT:
{context}

Feel free to respond in a natural and conversational tone. If the user's question is unrelated to the blog content, kindly redirect them back to the blog topic by saying:
<p>"I'm here to assist with questions related to the blog. Please ask me about the content of the blog."</p>

Previous conversation context:
{chat_history}

User's question: {question}

Please respond in a warm, engaging, and user-friendly HTML format. If you need to say that information is not available, be empathetic and friendly. For example, you could say:
<p>"It looks like there's no information about that topic in the content I have. But I'm here to help with anything else you'd like to know about! Feel free to ask."</p>
"""
    
    
    PROMPT = PromptTemplate(input_variables=["context", "chat_history", "question"], template=system_template)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 10, "lambda_mult": 0.5}),  # Reduce retrieval complexity
        memory=memory,
        combine_docs_chain_kwargs={"prompt": PROMPT},
        chain_type="stuff",
        return_source_documents=False,  # Disable source document return to reduce token usage
        verbose=False  # Disable verbosity
    )

def initialize_chain():
    global conversation_chain
    documents = get_text_and_filenames(DOCUMENTS_DIR)
    chunks = get_chunks(documents)
    vector_store = get_vector(chunks)
    conversation_chain = create_conversation_chain(vector_store)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_id = request.remote_addr  # Use user's IP address as an identifier
    user_input = request.json.get("question")

    if not user_input:
        return jsonify({"error": "No question provided"}), 400

    if conversation_chain is None:
        return jsonify({"error": "Conversation chain not initialized. Please provide a blog URL first."}), 400

    # Initialize user_question_count dictionary if not already defined
    global user_question_count
    if 'user_question_count' not in globals():
        user_question_count = {}

    # Check and update user question count
    if user_id not in user_question_count:
        user_question_count[user_id] = 0

    if user_question_count[user_id] >= CONVERSATION_LIMIT:
        return jsonify({"error": "Your limit of 3 questions has been reached. Thank you for using our service."}), 400

    user_question_count[user_id] += 1

    try:
        # Generate response from conversation chain
        response = conversation_chain({"question": user_input})

        # Token Estimation and Cost Calculation
        question_tokens = estimate_tokens(user_input)
        answer_text = response.get("answer", "")
        answer_tokens = estimate_tokens(answer_text)

        total_tokens = question_tokens + answer_tokens

        # Calculate costs
        input_cost = (question_tokens / 1_000_000) * UNCACHED_INPUT_COST_PER_MILLION
        output_cost = (answer_tokens / 1_000_000) * OUTPUT_COST_PER_MILLION
        total_cost = input_cost + output_cost
        

    

        # Return response, tokens, and cost
        return jsonify({
            "response": answer_text,
            "question_tokens": question_tokens,
            "answer_tokens": answer_tokens,
            "total_tokens": total_tokens,
            "estimated_cost": f"${total_cost:.4f}",
            "remaining_questions": CONVERSATION_LIMIT - user_question_count[user_id]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    initialize_chain() 
    app.run(host='0.0.0.0', port=5005, debug=False)
