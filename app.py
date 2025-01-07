import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Configure the generative AI client using the API key from environment variables
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to extract text from uploaded PDF files
def get_pdf_text(pdf_docs):
    """
    Extracts text from the pages of the provided PDF documents.

    Args:
    pdf_docs (list): List of uploaded PDF file objects.

    Returns:
    str: Combined text content of all pages from all PDFs.
    """
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into manageable chunks
def get_text_chunks(text):
    """
    Splits the given text into smaller chunks for processing.

    Args:
    text (str): The complete text extracted from PDFs.

    Returns:
    list: List of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create and store vector embeddings for text chunks
def get_vector_store(text_chunks):
    """
    Creates a vector store from text chunks using Google Generative AI embeddings 
    and saves it locally.

    Args:
    text_chunks (list): List of text chunks to be embedded.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to initialize a conversational chain with a custom prompt
def get_conversational_chain():
    """
    Creates a conversational chain for question-answering based on a custom prompt.

    Returns:
    chain: A LangChain QA chain instance using Google Generative AI.
    """
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to handle user input and generate a response
def user_input(user_question):
    """
    Processes the user's question and generates a response based on context from the vector store.

    Args:
    user_question (str): The question provided by the user.

    Outputs:
    Writes the generated response to the Streamlit app.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Load the vector store and search for similar documents
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    # Generate a response using the conversational chain
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    
    # Display the response in the Streamlit app
    print(response)
    st.write("Reply: ", response["output_text"])

# Main function to run the Streamlit app
def main():
    """
    The main function to initialize and run the Streamlit app interface.
    Handles user interactions for uploading PDFs and asking questions.
    """
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini")

    # Input field for user questions
    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    # Sidebar for uploading and processing PDF files
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button", 
            accept_multiple_files=True
        )
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                # Extract, split, and embed text from the uploaded PDFs
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

# Entry point for the script
if __name__ == "__main__":
    main()
