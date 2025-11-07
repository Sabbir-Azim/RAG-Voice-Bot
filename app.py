import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from gtts import gTTS
from io import BytesIO
from streamlit_mic_recorder import speech_to_text
import base64
import tempfile
import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional, Dict, Any

# Load environment variables
load_dotenv()

# ===============================
# Configuration
# ===============================
class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    COLLECTION_NAME = "xeven_voicebot"
    EMBEDDING_MODEL = "text-embedding-3-small"
    CHAT_MODEL = "gpt-4o-mini"
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 100
    LANGUAGE = "en"
    CHROMA_PERSIST_DIR = "./chroma_db"  # Local folder to store embeddings


# ===============================
# Session State Initialization
# ===============================
def initialize_session_state() -> None:
    defaults = {
        "chat_history": [],
        "data_stored": False,
        "file_meta": {},
        "recording_key": 0,
        "uploaded_file": None,
        "pdf_processed": False,
        "chroma_store": None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ===============================
# PDF Upload + Processing
# ===============================
def upload_pdf() -> Optional[Any]:
    uploaded_file = st.file_uploader("📄 Upload your PDF", type=["pdf"])
    if uploaded_file is not None:
        st.session_state["file_meta"] = {
            "file_name": uploaded_file.name,
            "file_size": f"{uploaded_file.size / 1024:.2f} KB"
        }
        st.session_state["uploaded_file"] = uploaded_file
    return uploaded_file


def process_pdf(uploaded_file: Any) -> None:
    """Extract, split, and store document embeddings in Chroma."""
    if uploaded_file is None:
        st.error("Please upload a valid PDF file.")
        return

    try:
        with st.spinner("Processing PDF... ⏳"):
            # Save temporary PDF
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_path = temp_file.name

            # Load and split
            loader = PyPDFLoader(temp_path)
            pages = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=Config.CHUNK_SIZE,
                chunk_overlap=Config.CHUNK_OVERLAP,
                separators=["\n\n", "\n", " ", ""]
            )
            chunks = splitter.split_documents(pages)

            embeddings = OpenAIEmbeddings(
                model=Config.EMBEDDING_MODEL,
                openai_api_key=Config.OPENAI_API_KEY
            )

            # Create Chroma vector store
            chroma_store = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=Config.CHROMA_PERSIST_DIR,
                collection_name=Config.COLLECTION_NAME
            )

            chroma_store.persist()

            st.session_state["chroma_store"] = chroma_store
            st.session_state["data_stored"] = True
            st.session_state["pdf_processed"] = True

            Path(temp_path).unlink(missing_ok=True)
            st.success("✅ PDF successfully processed and stored in Chroma!")

    except Exception as e:
        st.error(f"❌ Error processing PDF: {str(e)}")


def get_chroma_store() -> Optional[Chroma]:
    """Reload Chroma store if not in session."""
    if st.session_state.get("chroma_store") is not None:
        return st.session_state["chroma_store"]

    try:
        embeddings = OpenAIEmbeddings(
            model=Config.EMBEDDING_MODEL,
            openai_api_key=Config.OPENAI_API_KEY
        )

        chroma_store = Chroma(
            collection_name=Config.COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=Config.CHROMA_PERSIST_DIR
        )

        st.session_state["chroma_store"] = chroma_store
        return chroma_store

    except Exception as e:
        st.error(f"Failed to initialize Chroma store: {str(e)}")
        return None


# ===============================
# Voice Input and Output
# ===============================
def process_audio_input() -> Optional[str]:
    """Capture and transcribe audio input."""
    unique_key = f"STT-{st.session_state['recording_key']}"
    transcribed_text = speech_to_text(
        language=Config.LANGUAGE,
        use_container_width=True,
        just_once=True,
        key=unique_key
    )
    return transcribed_text


def text_to_speech(text: str, lang: str = "en") -> BytesIO:
    """Convert bot text to audio."""
    tts = gTTS(text=text, lang=lang, slow=False)
    audio_data = BytesIO()
    tts.write_to_fp(audio_data)
    audio_data.seek(0)
    return audio_data


# ===============================
# RAG + Response Generation
# ===============================
def generate_response(chroma_store: Chroma, query: str) -> str:
    """RAG pipeline: retrieve context + answer."""
    try:
        template = """
        You are a helpful AI assistant. Use the provided context to answer accurately.
        Always respond in English.

        Context: {context}
        Question: {question}
        Answer:
        """

        prompt = ChatPromptTemplate.from_template(template)
        retriever = chroma_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        setup_and_retrieval = RunnableParallel({"context": retriever, "question": RunnablePassthrough()})

        model = ChatOpenAI(
            model=Config.CHAT_MODEL,
            temperature=0.3,
            openai_api_key=Config.OPENAI_API_KEY
        )

        output_parser = StrOutputParser()
        rag_chain = setup_and_retrieval | prompt | model | output_parser
        response = rag_chain.invoke(query)
        return response

    except Exception as e:
        return f"Error generating response: {str(e)}"


# ===============================
# Display Utilities
# ===============================
def display_chat_history() -> None:
    for chat in st.session_state["chat_history"]:
        with st.chat_message("user"):
            st.write(chat["user_input"])
        with st.chat_message("assistant"):
            st.write(chat["bot_response"])
            st.audio(chat["bot_audio"], format="audio/mp3")


def display_pdf_preview() -> None:
    if not st.session_state["pdf_processed"]:
        return
    uploaded_file = st.session_state["uploaded_file"]
    if not uploaded_file:
        return

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            path = tmp.name
        with open(path, "rb") as f:
            b64_pdf = base64.b64encode(f.read()).decode("utf-8")
        st.markdown(f'<iframe src="data:application/pdf;base64,{b64_pdf}" width="100%" height="600px"></iframe>', unsafe_allow_html=True)
        Path(path).unlink(missing_ok=True)
    except Exception as e:
        st.error(f"Error displaying PDF: {str(e)}")


# ===============================
# Streamlit App
# ===============================
def main():
    st.set_page_config(page_title="Voicebot - Chat with Documents", layout="wide")
    initialize_session_state()

    st.title("🎙️ Voicebot - Chat with Your Documents")
    st.markdown("Upload a PDF, ask with voice, and get answers in text + speech!")

    # Sidebar
    with st.sidebar:
        st.header("📖 Instructions")
        st.markdown("""
        1️⃣ Upload a PDF document  
        2️⃣ Click **Process Document**  
        3️⃣ Ask using your **voice** (English only)  
        4️⃣ Get both **text & audio** answers  
        """)
        if st.session_state["file_meta"]:
            st.info(f"📎 **{st.session_state['file_meta']['file_name']}** — {st.session_state['file_meta']['file_size']}")

    # Layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("💬 Upload & Ask")
        uploaded_file = upload_pdf()

        if uploaded_file and not st.session_state["pdf_processed"]:
            if st.button("🔄 Process Document", use_container_width=True):
                process_pdf(uploaded_file)

        if st.session_state["data_stored"]:
            st.subheader("🎤 Ask your question")
            audio_input = process_audio_input()
            if audio_input:
                st.session_state["recording_key"] += 1
                with st.spinner("Generating response..."):
                    chroma_store = get_chroma_store()
                    if chroma_store:
                        bot_response = generate_response(chroma_store, audio_input)
                        bot_audio = text_to_speech(bot_response, lang=Config.LANGUAGE)
                        st.session_state["chat_history"].append({
                            "user_input": audio_input,
                            "bot_response": bot_response,
                            "bot_audio": bot_audio
                        })
                        st.rerun()

            if st.session_state["chat_history"]:
                st.divider()
                st.subheader("💭 Conversation History")
                display_chat_history()

    with col2:
        st.header("📄 PDF Preview")
        if st.session_state["pdf_processed"]:
            display_pdf_preview()
        else:
            st.info("Upload and process a document to preview it here.")


if __name__ == "__main__":
    main()
