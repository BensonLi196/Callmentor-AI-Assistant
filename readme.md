
# Callmentor AI Assistant

  

## Installation and Setup


### 1. Requirements

- Python 3

### 2. Dependencies

Install the necessary libraries by running:

```bash
pip  install  -r  requirements.txt
```

### 3. Configure Environment Variable

Create a `.env` file in the project root and add the following:

`OPENAI_API_KEY=your_openai_api_key_here`

### 4. Add PDF Files
Place the FAQ PDFs that the chatbot will use in a `./faq_pdfs` directory. These PDFs should contain the knowledge base for Callmentor and other relevant topics.

## Running the Application

### 1. Start the Server

Run the Flask application by executing:

```bash
python app.py 
```

This will start the server at `http://localhost:5000/`.

### 2. Access the Web Interface

Open your web browser and navigate to:

`http://localhost:5000/`