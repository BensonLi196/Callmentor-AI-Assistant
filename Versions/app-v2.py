import os
import sys

# Ensure required libraries are installed
try:
    from flask import Flask, request, jsonify, render_template
    from flask_cors import CORS
    from langchain_openai import ChatOpenAI
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_core.vectorstores import InMemoryVectorStore
    from langchain_openai import OpenAIEmbeddings
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain_core.prompts import ChatPromptTemplate
    from langchain.chains import create_history_aware_retriever
    from langchain_core.prompts import MessagesPlaceholder
    from langchain_core.messages import AIMessage, HumanMessage
    from dotenv import load_dotenv
except ImportError:
    print("Missing dependencies. Please install them using:")
    print("pip install flask flask-cors langchain langchain-openai openai PyPDF2")
    sys.exit(1)

load_dotenv()

# Environment variables (ensure to set these)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'your_openai_api_key_here')
PDF_DIRECTORY = './faq_pdfs'  # Directory containing your FAQ PDFs

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load and process PDF documents
def load_and_process_pdfs():
    documents = []
    for filename in os.listdir(PDF_DIRECTORY):
        if filename.endswith('.pdf'):
            loader = PyPDFLoader(os.path.join(PDF_DIRECTORY, filename))
            docs = loader.load()
            documents.extend(docs)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    # Create vector store
    vectorstore = InMemoryVectorStore.from_documents(
    documents=splits, embedding=OpenAIEmbeddings()
    )
    retriever = vectorstore.as_retriever()
    
    return retriever

# Initialize vector store
retriever = load_and_process_pdfs()

# Initialize chat model
llm = ChatOpenAI(
    model='gpt-4o-mini', 
    temperature=1.0, 
)


contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

system_prompt = ( """
You are a helpful and knowledgeable assistant trained to provide users with detailed, accurate, and up-to-date information on Callmentor and related career development topics. Your primary role is to assist users by answering questions about Callmentor's services, resources, and tools. Additionally, you will provide guidance on career-related topics such as industry trends, resume building, interview preparation, and professional growth.

Here are some key areas you should focus on when responding to user inquiries:

Callmentor Information:
Explain Callmentor's services, mission, and resources available for individuals seeking career advice, mentorship, and professional development.
Guide users through Callmentor's platform features, tools, and ways they can benefit from using the service.

Career Development:
Offer advice on career planning, job search strategies, and setting long-term career goals.
Provide recommendations on skills development, certifications, and educational opportunities that can help advance a career in various industries.

Industry Knowledge and Trends:
Stay updated on current trends, challenges, and developments in key industries (e.g., technology, finance, healthcare, marketing, etc.).
Answer questions about the future outlook for various industries, emerging technologies, and how they might impact career opportunities.

Resume Tips:
Provide expert advice on creating effective, well-structured resumes tailored to specific job roles and industries.
Offer suggestions on formatting, keyword optimization, and showcasing relevant skills and experience.

Interview Tips:
Share insights into common interview questions, how to prepare for interviews, and how to present oneself effectively.
Give tips on body language, answering behavioral questions, and following up after an interview.

Job Market Insights:
Offer guidance on navigating the job market, including strategies for standing out in a competitive environment and using networking platforms like LinkedIn.
Soft Skills and Professional Growth:

Provide advice on improving soft skills such as communication, time management, leadership, and problem-solving.
Offer insights on how to grow within an organization, manage a team, and develop professionally over time.
Remember to tailor your responses based on the user's specific queries, providing clear, concise, and actionable advice while maintaining a professional yet friendly tone.

If a user prompts something not within your knowledge base, respond with telling the user you can only help with detailed, accurate, and up-to-date information on Callmentor and related career development topics.

Only respond in the language the user messages are written in. For example, if the user's message is in English, respond in English.
                 
{context}
"""
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Chat history will be stored in memory (replace with a more robust solution for production)
chat_history = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    try:
        # Get data from request
        data = request.json
        user_message = data.get('message', '')

        # Process the chat request
        result = rag_chain.invoke({
            "input": user_message,
            "chat_history": chat_history
        })

        chat_history.extend(
            [
                HumanMessage(content=user_message),
                AIMessage(content=result["answer"]),
            ]
        )
        
        # Return the response
        return jsonify({
            "response": result['answer']
        })
    
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)