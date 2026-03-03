import os
import json
import time
import asyncio
import traceback
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage


# Initialize LangChain Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.1,
)

PROMPT_TEMPLATE = """You are a Senior Technical Dialogue Editor and Data Analyst. Your job is to lightly clean a diarized transcript and generate metadata for a RAG search system.

═══════════════════════════════════
TRANSCRIPT (pre-cleaned, mostly accurate):
{raw_transcript}
═══════════════════════════════════

IMPORTANT CONTEXT: This transcript has already been cleaned with ML-based audio processing. The words are mostly accurate. Your job is NOT to heavily rephrase or rewrite — it is to make light corrections.

YOUR WORKFLOW & RULES:

1. **Role Analysis & Labeling:**
   - Determine who is the 'Agent' (Provider) and who is the 'Client' (User). 
   - Replace "Speaker 0/1" with these labels consistently.

2. **Vector-Optimized Summary:**
   - Write a dense summary mentioning all technical entities (app names, features, error messages).
   - Include what the issue was and how it was resolved.

3. **Light Transcript Cleanup (NOT a full rewrite):**
   - **PRESERVE original wording** as much as possible. The transcription is mostly correct.
   - **Only fix speaker assignments** where the conversation flow clearly shows the wrong person is speaking (e.g., a question and answer both assigned to the same speaker).
   - **Merge fragments** only when the same speaker's thought was split across consecutive lines.
   - **Remove obvious filler** like repeated "ok ok ok" or "hmm hmm", but keep natural acknowledgements.
   - **Do NOT rephrase or paraphrase** sentences that are already understandable.

4. **Hinglish & Formatting:**
   - Convert Devanagari (Hindi) to Romanized Hinglish.
   - Keep technical English words exactly as they are.

5. **Satisfaction Score:**
   - Rate the Client's satisfaction from 1-10.

STRICT JSON OUTPUT FORMAT (return ONLY this JSON, nothing else):
{{
  "summary": "Dense technical summary for embeddings...",
  "satisfactionScore": 10,
  "detectedRoles": {{ "speaker0": "Agent", "speaker1": "Client" }},
  "tags": ["ERP", "Technical Support"], 
  "refinedTranscript": [
    {{ "role": "Agent", "text": "Lightly cleaned Hinglish text..." }},
    {{ "role": "Client", "text": "Lightly cleaned Hinglish text..." }}
  ]
}}

Speaker count is {speaker_count}. Return ONLY the JSON object."""


async def refine_transcript(raw_transcript: str, speaker_count: int) -> dict:
    """
    Refine a raw diarized transcript using LangChain + Gemini.
    """
    print("   📝 Building Gemini prompt...")
    prompt = PROMPT_TEMPLATE.format(
        raw_transcript=raw_transcript,
        speaker_count=speaker_count
    )
    print(f"   📝 Prompt length: {len(prompt)} chars")

    try:
        start_time = time.time()

        # Step 1: Call Gemini via LangChain (use thread + timeout to prevent hanging)
        print("   🔄 Calling Gemini via LangChain...")
        message = HumanMessage(content=prompt)
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(llm.invoke, [message]),
                timeout=60
            )
        except asyncio.TimeoutError:
            print("   ❌ Gemini timed out after 60s")
            return _error_result()
        
        elapsed = int((time.time() - start_time) * 1000)
        print(f"   ✅ Gemini responded in {elapsed}ms")

        # Step 2: Extract text
        text = response.content.strip()
        print(f"   📄 Response length: {len(text)} chars")
        print(f"   📄 First 200 chars: {text[:200]}")

        # Step 3: Clean markdown fences
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first line (```json) and last line (```)
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            text = text.strip()
            print("   🧹 Stripped markdown code fences")

        # Step 4: Parse JSON
        print("   🔍 Parsing JSON...")
        data = json.loads(text)
        print(f"   ✅ JSON parsed successfully — keys: {list(data.keys())}")

        # Step 5: Build transcript string
        transcript_parts = data.get("refinedTranscript", [])
        transcript_string = "\n".join(
            f"{s['role']}: {s['text']}" for s in transcript_parts
        )

        print(f"   🔧 Gemini produced {len(transcript_parts)} refined segments")

        return {
            "summary": data.get("summary", ""),
            "satisfaction_score": data.get("satisfactionScore", 0),
            "tags": data.get("tags", []),
            "detected_roles": data.get("detectedRoles", {}),
            "transcript": transcript_string,
            "processing_time": elapsed,
        }

    except json.JSONDecodeError as e:
        print(f"   ❌ JSON parse error: {e}")
        print(f"   ❌ Raw text was: {text[:500] if 'text' in dir() else 'N/A'}")
        return _error_result()
    except Exception as err:
        print(f"   ❌ Gemini processing failed: {err}")
        traceback.print_exc()
        return _error_result()


def _error_result():
    return {
        "transcript": None,
        "summary": "Could not generate summary",
        "satisfaction_score": 0,
        "tags": [],
        "detected_roles": {},
        "processing_time": 0,
    }
