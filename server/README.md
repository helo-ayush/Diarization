# 🗣️ Voice Diarization & Analytics API

Welcome to the backend engine for the **Voice Diarization & Analytics** platform! This Python (FastAPI) application is designed to ingest raw call-center audio, isolate different speakers, identify known support agents using Voice Fingerprinting, transcribe the conversation (specifically optimized for Hinglish/Indian languages), and analyze the transcript's technical sentiment using Large Language Models.

---

## 🧠 How the "Brain" Works (Architecture)

This application uses a highly-optimized, multi-stage hybrid AI pipeline to achieve its results. Here is exactly what happens when an audio file hits the server:

### 1. Audio Pre-Processing (Noise Reduction & VAD)
Before any AI model sees the audio, it is processed through **Silero VAD** (Voice Activity Detection) and **noisereduce**. This strips out long stretches of silence, background hums, and static, significantly reducing the file size and ensuring the downstream models only process actual human speech.

### 2. Transcription & Base Diarization (Sarvam AI)
The cleaned audio is sent to **Sarvam AI**, a model specifically tuned for Indian accents and mixed languages (like Hindi + English). Sarvam returns the raw text and groups it by generic speaker IDs (e.g., `Speaker 0`, `Speaker 1`).

### 3. Voice Fingerprinting & Agent Identification (Pyannote)
This is where the magic happens. We don't want to blindly guess who the Agent is and who the Customer is. 
* We extract the longest continuous audio segment for each detected speaker.
* We pass that snippet through **Pyannote's WeSpeaker ResNet** model to generate a 256-dimensional mathematical "Voice Print" (embedding) for that speaker.
* We calculate the **Cosine Similarity** of this voice print against all registered Agents in our MongoDB database. 
* If a strong match is found (> 0.45 similarity), we explicitly rename that generic `Speaker 0` to the actual agent's name (e.g., `"Ayush"`).

### 4. Technical Refinement & Sentiment Analysis (LangChain + Gemini)
The newly mapped, diarized transcript is passed to **Google Gemini**. 
* **If Pyannote successfully identified an Agent**, we instruct Gemini: *"Do not guess roles; the person named Ayush is the Agent. Clean the transcript formatting but preserve his name."*
* **If no Agent was identified**, Gemini uses the conversational context (who is asking questions vs. who is solving problems) to label "Agent" and "Client".
* Gemini then generates a dense technical summary, extracts tags (e.g., "Hardware Issue", "Refund"), and scores the customer's satisfaction from 1-10.

### 5. Vector Embeddings (RAG)
Finally, the generated summary is converted into a Vector Embedding. This allows users to perform semantic searches later (e.g., *"Find calls where the customer had a blue screen laptop issue"*).

---

## 📡 API Routes & Documentation

Below are the primary routes exposed by the FastAPI server, along with exactly how to format your requests and what to expect in return.

### 1. `POST /api/v1/agents/enroll`
Registers a new Agent into the system by extracting and saving their unique Voice Fingerprint.

**Request Format (Multipart Form Data):**
- `name` (text): The name of the agent (e.g., "Ayush").
- `voice_sample` (file): A clear, 10-30 second audio clip of the agent speaking.

**Example (cURL):**
```bash
curl -X POST "http://localhost:8000/api/v1/agents/enroll" \
  -F "name=Ayush" \
  -F "voice_sample=@/path/to/ayush_voice.wav"
```

**Success Response:**
```json
{
  "success": true,
  "message": "Successfully enrolled agent: Ayush",
  "agent_id": "ayush"
}
```

---

### 2. `GET /api/v1/agents`
Fetches a list of all enrolled agents currently in the MongoDB database (excluding their heavy vector embeddings to save bandwidth).

**Example (cURL):**
```bash
curl -X GET "http://localhost:8000/api/v1/agents"
```

**Success Response:**
```json
{
  "agents": [
    {
      "_id": "ayush",
      "name": "Ayush",
      "updatedAt": "2026-03-15T10:00:00Z"
    }
  ]
}
```

---

### 3. `POST /api/v1/transcriptions`
The core engine endpoint. Upload a call recording to be cleaned, transcribed, identified, analyzed, and saved to the database.

**Request Format (Multipart Form Data):**
- `call_recording` (file): The audio recording of the conversation.

**Example (JavaScript / Fetch):**
```javascript
const formData = new FormData();
formData.append("call_recording", fileInput.files[0]);

const response = await fetch("http://localhost:8000/api/v1/transcriptions", {
  method: "POST",
  body: formData
});
const data = await response.json();
```

**Success Response:**
```json
{
  "success": true,
  "_id": "64fe1b...",
  "pynoteDiarized": true,
  "analysis": {
    "summary": "Customer called regarding a green line on their laptop screen. Agent Ayush determined it was a hardware issue...",
    "satisfactionScore": 7,
    "tags": ["Hardware", "Screen Issue", "Lenovo"],
    "detectedRoles": {
      "Ayush": "Agent",
      "Speaker 1": "Client"
    }
  },
  "transcript": "Ayush: Mere PC mein abhi ek minor issue find hua hai...\nClient: I guess mere PC mein ek blue screen aa raha hai...",
  "speakerCount": 2,
  "metrics": {
    "totalWords": 450,
    "rawSegments": 12,
    "geminiRefined": true,
    "diarizationConfidence": 0.945,
    "pynoteDiarized": true,
    "processingMs": {
      "stt": 4500,
      "gemini": 2100
    }
  }
}
```

---

### 4. `POST /api/v1/search`
Performs a semantic Vector Search across all saved transcriptions using natural language.

**Request Format (JSON):**
- `query` (string): The natural language search query.
- `limit` (integer, optional): Max results to return (default 5).
- `all_entries` (boolean, optional): Set to true to bypass vector search and just return the latest records.

**Example (cURL):**
```bash
curl -X POST "http://localhost:8000/api/v1/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "customer complained about a blue screen and green line", "limit": 3}'
```

**Success Response:**
```json
[
  {
    "_id": "64fe1b...",
    "filename": "call_001.m4a",
    "transcript": "Ayush: ...",
    "summary": "Customer called regarding a green line...",
    "satisfactionScore": 7,
    "similarity": 0.89
  }
]
```

---

## 🏗️ Building & Installation

We use **Docker** to completely automate the setup of this project. You do not need to worry about manually installing Python, FFmpeg, Node.js, or patching broken AI libraries—Docker handles it all automatically!

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed on your machine.
- [Git](https://git-scm.com/) installed.

### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd Diarization
```

### 2. Set Up Environment Variables
Inside the `server/` folder, you will find a file called `.env.example`.
Create a copy of it and name it `.env`:
```bash
cp .env.example .env
```
Open `.env` and insert your actual API keys (Gemini, Sarvam, HuggingFace) and your MongoDB URI.

### 3. Run with Docker Compose
Navigate into the `server/` directory (where `docker-compose.yml` is located) and run:
```bash
cd server
docker-compose up --build
```

**That's it!** Docker will:
1. Spin up a Linux container.
2. Install system-level FFmpeg binaries.
3. Install all Python dependencies.
4. Auto-patch the `pyannote` library via `patch_pyannote.py`.
5. Start the backend API on `http://localhost:8000`.

You can now send requests to `http://localhost:8000` via Postman or your browser!
