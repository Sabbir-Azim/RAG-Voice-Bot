# 🎙️ RAG VoiceBot — Chat with Your PDF using OpenAI + Chroma + Streamlit

> 🧠 A Retrieval-Augmented Generation (RAG) powered voice assistant that lets you **ask questions from any PDF using your voice** — it converts your speech to text, searches your document intelligently, and replies in both **text and speech** using OpenAI models.

---

## 🚀 Features

- 🎤 **Voice Input (Speech-to-Text)** — Ask questions using your microphone  
- 🧠 **RAG-Powered Retrieval** — Contextual answers based on your uploaded PDF  
- 🗃️ **Chroma Vector Store** — Local, persistent, and fast vector database  
- 🔊 **Text-to-Speech Output** — The assistant responds with spoken answers  
- 💬 **Streamlit Interface** — Simple and interactive chat-style UI  
- 🤖 **OpenAI-Powered Intelligence** — Embeddings and responses from GPT models  

---

## 🏗️ Tech Stack

| Component | Library / Tool |
|------------|----------------|
| LLM | OpenAI GPT (via `langchain-openai`) |
| Vector Store | Chroma |
| App Framework | Streamlit |
| Speech-to-Text | `streamlit-mic-recorder` |
| Text-to-Speech | `gTTS` |
| PDF Loader | `langchain_community` |
| Env Management | `python-dotenv` |

---

## 📦 Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Sabbir-Azim/RAG-Voice-Bot.git
cd RAG-Voice-Bot
```
### 2️⃣ Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Set Environment Variables

Create a file named .env in your project root directory and add your OpenAI API key:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### ▶️ Usage

Run the Streamlit app:

```bash
streamlit run app.py
```