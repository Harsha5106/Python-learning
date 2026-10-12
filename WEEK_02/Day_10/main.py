import os
import uuid
import chromadb
import gradio as gr
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI


# 1. Load environment variables
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY was not found. Check your root .env file."
    )


# 2. Configure Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    google_api_key=api_key,
    temperature=0.2
)


# 3. Create the ChromaDB client
client = chromadb.Client()

# Keep track of the currently uploaded document
active_collection = None


# 4. Process an uploaded PDF
def upload_document(file):
    global active_collection

    if file is None:
        return "Please upload a PDF file first."

    try:
        # Extract text from every PDF page
        reader = PdfReader(file)

        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                text_parts.append(page_text)

        text = "\n".join(text_parts)

        if not text.strip():
            return (
                "No readable text found in this PDF. "
                "Scanned PDFs may require OCR."
            )

        # Split the extracted text into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = splitter.split_text(text)

        if not chunks:
            return "No text chunks could be created."

        # Give each upload a unique collection name
        collection_name = "study_" + uuid.uuid4().hex[:12]

        collection = client.create_collection(
            name=collection_name
        )

        # Store chunks in ChromaDB.
        # ChromaDB automatically creates embeddings for documents
        # and uses them for semantic search.
        collection.add(
            ids=[str(i) for i in range(len(chunks))],
            documents=chunks
        )

        # Activate the new collection for question answering
        active_collection = collection

        return (
            f"PDF uploaded successfully!\n"
            f"Pages: {len(reader.pages)}\n"
            f"Chunks stored: {len(chunks)}"
        )

    except Exception as e:
        return f"Upload failed: {type(e).__name__}: {e}"


# 5. Answer a question using retrieved context
def ask_question(question):
    if active_collection is None:
        return "Please upload a study document first."

    if not question or not question.strip():
        return "Please enter a question."

    try:
        # Retrieve the two most relevant chunks
        results = active_collection.query(
            query_texts=[question.strip()],
            n_results=2
        )

        documents = results.get("documents", [[]])[0]

        if not documents:
            return "No relevant information was found."

        context = "\n\n".join(documents)

        # Ground the answer in the retrieved context
        prompt = f"""
You are an AI study assistant.

Answer the question using only the study material in the context.
If the context does not contain the answer, say:
"I don't know based on the uploaded document."

Keep the answer clear and beginner-friendly.

Study material:
{context}

Question:
{question.strip()}

Answer:
"""

        response = llm.invoke(prompt)

        # Handle LangChain's response content
        if isinstance(response.content, str):
            return response.content

        content = response.content

        if isinstance(content, list):
            return "\n".join(
            block["text"]
            for block in content
            if isinstance(block, dict) and "text" in block
            )

        return str(content)

    except Exception as e:
        return f"Question failed: {type(e).__name__}: {e}"


# 6. Build the Gradio interface
with gr.Blocks(title="AI Study Assistant") as demo:
    gr.Markdown("# AI Study Assistant")
    gr.Markdown(
        "Upload a PDF, then ask questions about its content."
    )

    with gr.Group():
        file_input = gr.File(
            label="Upload study material (PDF)",
            file_types=[".pdf"],
            type="filepath"
        )

        upload_button = gr.Button("Upload Document")
        upload_status = gr.Textbox(
            label="Upload Status",
            interactive=False
        )

    with gr.Group():
        question_input = gr.Textbox(
            label="Ask a question",
            placeholder="What is the difference between AI and ML?"
        )

        ask_button = gr.Button("Ask Question")
        answer_output = gr.Textbox(
            label="Answer",
            lines=6,
            interactive=False
        )

    upload_button.click(
        fn=upload_document,
        inputs=file_input,
        outputs=upload_status
    )

    ask_button.click(
        fn=ask_question,
        inputs=question_input,
        outputs=answer_output
    )


# 7. Launch the application
if __name__ == "__main__":
    demo.launch()
