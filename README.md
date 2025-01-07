
# Chat with PDF
## Chat with PDF using Streamlit and Google Generative AI

This project is a Streamlit-based application that allows users to upload multiple PDF files, process their content, and ask questions based on the extracted text. It uses Google Generative AI for embeddings and conversational responses.

---

## Features

- **PDF Text Extraction**: Extracts text content from uploaded PDF files.
- **Text Chunking**: Splits the extracted text into smaller chunks for efficient processing.
- **Vector Store Creation**: Uses Google Generative AI embeddings to create and store vector representations of text chunks.
- **Question Answering**: Leverages a conversational chain with custom prompts to answer user queries based on the uploaded PDF content.
- **Streamlit Interface**: Provides a user-friendly interface for uploading files and interacting with the chatbot.

---

## Prerequisites

1. **Python**: Ensure Python 3.8 or above is installed.
2. **API Key**: Obtain an API key from Google Generative AI.
3. **Dependencies**: Install the required Python libraries listed below.
---





## Deployment

To deploy this project run

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repository/chat-with-pdf.git
   cd chat-with-pdf
2. **Set Up a Virtual Environment:**

```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows
```
3. **Install Dependencies:**
```bash
pip install -r requirements.txt
```
4. **Add New API Key:**

You need to get new API Key to run the code on your local computer.
Here is the link https://aistudio.google.com/app/apikey.

Change the GOOGLE_API_KEY in .env file to designated apikey you get from the link.

5. **Run the Application:**
```bash
streamlit run app.py
```






## Usage

To run the application, do following steps:
1. **Upload PDF Files:**

- Use the sidebar to upload one or more PDF files.
- Click on the "Submit & Process" button to process the uploaded PDFs.

2. **Ask Questions:**
- Enter your question in the text input field on the main page.
- The application will display a detailed response based on the content of the uploaded PDFs.

## Demo

Here is the video demo link
```bash
https://drive.google.com/file/d/1KtoQ_yVlETZIUcx3KwzYq2mqLAfcRcNg/view?usp=sharing
```

