🎙️ RAG VoiceBot — Chat with Your PDF using OpenAI + Chroma + Streamlit

🧠 A Retrieval-Augmented Generation (RAG) based voice assistant that lets you ask questions from any PDF using your voice — it converts your speech to text, searches your document using embeddings, and replies back in both text and speech.

🚀 Features

✅ Voice Input (Speech-to-Text) — Ask your questions verbally.
✅ RAG Retrieval — Answers grounded in your uploaded PDF data.
✅ Chroma Vector Store — Local and persistent vector database (no API timeout).
✅ Text-to-Speech Output — Bot responds with a natural voice.
✅ Streamlit Interface — Simple, clean, and interactive web UI.
✅ OpenAI-Powered Intelligence — Uses GPT and embedding models from OpenAI.

🏗️ Tech Stack
Component	Library / Tool
LLM	OpenAI GPT (via langchain-openai)
Vector DB	Chroma
App Framework	Streamlit
Speech-to-Text	streamlit-mic-recorder
Text-to-Speech	gTTS
PDF Loader	PyPDF2, langchain_community
Environment	python-dotenv
📦 Installation
1️⃣ Clone the Repository
git clone https://github.com/your-username/rag-voicebot.git
cd rag-voicebot

2️⃣ Create a Virtual Environment
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Add Environment Variables

Create a .env file in your project root:

OPENAI_API_KEY=your_openai_api_key_here


⚠️ You must have a valid OpenAI API key
.

▶️ Usage

Run the Streamlit app:

streamlit run app.py


Then:

1️⃣ Upload a PDF file 📄
2️⃣ Click Process Document to embed and store in Chroma
3️⃣ Click the 🎤 Mic button to ask a question aloud
4️⃣ The bot will reply with text + audio answer 🔊

🧩 Project Structure
rag-voicebot/
├── app.py                # Main Streamlit application
├── .env                  # Environment variables (API key)
├── requirements.txt      # Python dependencies
├── chroma_db/            # Local Chroma vector store
└── README.md             # Documentation

🧠 How It Works

PDF Upload: You upload your document.

Chunking & Embedding: The text is split into chunks and converted into embeddings using OpenAI.

Storage in Chroma: These embeddings are stored in a local vector database (Chroma).

Voice Query: Your voice input is transcribed into text using streamlit-mic-recorder.

RAG Pipeline: The bot retrieves relevant document chunks and sends them to the GPT model for an answer.

Voice Output: The response is converted back to speech with gTTS and played automatically.

🛠️ Configuration

Modify settings in the Config class inside app.py:

class Config:
    EMBEDDING_MODEL = "text-embedding-3-small"
    CHAT_MODEL = "gpt-4o-mini"
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 100
    LANGUAGE = "en"
    CHROMA_PERSIST_DIR = "./chroma_db"

🗣️ Future Improvements

 Add multilingual support (input in any language, output in English).

 Add memory-based chat context.

 Deploy on Streamlit Cloud or Hugging Face Spaces.

 Replace gTTS with real-time voice using OpenAI TTS.

🤝 Contributing

Pull requests and feature suggestions are always welcome!
To contribute:

Fork the repo

Create your feature branch (git checkout -b feature-name)

Commit your changes (git commit -m "Add new feature")

Push and open a pull request

📜 License

This project is licensed under the MIT License — see the LICENSE
 file for details.

💡 Acknowledgements

OpenAI
 for LLMs & embeddings

LangChain
 for RAG pipeline

Chroma
 for local vector DB

Streamlit
 for UI

gTTS
 for text-to-speech