import streamlit as st
from streamlit_chat import message
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from sentence_transformers import SentenceTransformer
import pdfplumber
import docx
from storage import store_document_chunks, get_relevant_chunks
import io
import hashlib

# Page configuration
st.set_page_config(
    page_title="Document Q&A VKU Assistant",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "language" not in st.session_state:
    st.session_state.language = "English"
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()
if "current_file_hash" not in st.session_state:
    st.session_state.current_file_hash = None

# Function to compute file hash
def compute_file_hash(file_content):
    """Compute SHA-256 hash of file content"""
    return hashlib.sha256(file_content).hexdigest()

@st.cache_resource
def load_embedding_models():
    """Load and cache embedding models"""
    return {
        "English": SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'),
        "Vietnamese": SentenceTransformer('keepitreal/vietnamese-sbert')
    }

@st.cache_resource
def initialize_ollama():
    """Initialize and cache Ollama client"""
    return Ollama(
        base_url="http://localhost:11434",
        model="llama3.2Q5KM:latest",
    )

# Load models
embedding_models = load_embedding_models()
ollama = initialize_ollama()

def split_document(document):
    """Split document into chunks with caching"""
    try:
        file_extension = document.name.split('.')[-1].lower()
        text = ""
        
        if file_extension == 'txt':
            for encoding in ['utf-8', 'latin-1', 'ascii']:
                try:
                    text = document.getvalue().decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
        
        elif file_extension == 'pdf':
            with pdfplumber.open(io.BytesIO(document.getvalue())) as pdf:
                text = " ".join([page.extract_text() or "" for page in pdf.pages])
        
        elif file_extension == 'docx':
            doc = docx.Document(io.BytesIO(document.getvalue()))
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        
        if not text:
            raise ValueError("No text could be extracted from the document")
            
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        return text_splitter.split_text(text)
    
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        return []

# UI Components
with st.sidebar:
    st.title("Configuration")
    
    # Language selection
    new_language = st.selectbox(
        "Choose Language",
        ["English", "Vietnamese"]
    )
    
    # Handle language change
    if new_language != st.session_state.language:
        st.session_state.language = new_language
        st.session_state.embedding_model = embedding_models[new_language]
        # Clear processed files if language changes
        st.session_state.processed_files = set()
        st.rerun()
    else:
        st.session_state.embedding_model = embedding_models[st.session_state.language]
    
    st.markdown("---")
    st.subheader("Model Information")
    st.info("""
    - English: MiniLM-L6-v2
    - Vietnamese: vietnamese-sbert
    - LLM: llama3.2Q80, llama3.2Q5KM
    """)

st.title("📚 Document Q&A Assistant")

# File upload section
col1, col2 = st.columns([2, 1])
with col1:
    uploaded_file = st.file_uploader(
        "Upload your document",
        type=["txt", "pdf", "docx"],
        help="Supported formats: TXT, PDF, DOCX"
    )

# Process uploaded file
if uploaded_file is not None:
    # Compute file hash
    file_content = uploaded_file.getvalue()
    current_hash = compute_file_hash(file_content)
    
    # Check if file needs processing
    if current_hash != st.session_state.current_file_hash:
        with st.spinner("Processing new document..."):
            chunks = split_document(uploaded_file)
            if chunks:
                try:
                    store_document_chunks(chunks)
                    st.session_state.current_file_hash = current_hash   
                    st.session_state.processed_files.add(current_hash)
                    st.success(f"✨ Document processed and ready for questions using {st.session_state.language} models")
                except Exception as e:
                    st.error(f"Error storing embeddings: {str(e)}")
    else:
        st.success("✨ Document already processed and ready for questions!")

    with col2:
        st.success("✅ File uploaded successfully!")
        st.info(f"📄 Filename: {uploaded_file.name}")
else:
    with col2:
        st.warning("⚠️ Please upload a document to begin")

# Chat interface
st.markdown("---")
st.subheader("💬 Chat Interface")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input and response generation
if prompt := st.chat_input("Ask a question about your document..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    try:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Get relevant chunks
                relevant_chunks = get_relevant_chunks(prompt)
                context = "\n".join(relevant_chunks)
                
                # Prepare prompt based on language
                if st.session_state.language == "English":
                    enhanced_prompt = f"""Please use the following context: {context}
                    to answer the following question: {prompt}
                    as detailed as possible."""
                else:
                    enhanced_prompt = f"""Hãy dựa trên ngữ cảnh sau: {context}
                     để trả lời câu hỏi: {prompt}
                     chi tiết nhất có thể."""

                # Generate response
                print("PROMT IS: "+enhanced_prompt)
                response = ollama.generate(
                    prompts=[enhanced_prompt],
                    generation_config={
                        'max_tokens': 4096,
                        'temperature': 0.7,
                        'top_p': 0.9,
                        'num_predict': 1024,
                        'stop': ['\n\n\n'],
                        'repeat_penalty': 1.1,
                    }
                )
                
                assistant_response = response.generations[0][0].text
                st.markdown(assistant_response)
                
                # Add to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": assistant_response
                })
    
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <small>Built with Streamlit • Powered by Nam-Giang</small>
    </div>
    """,
    unsafe_allow_html=True
)