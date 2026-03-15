# ==============================================================================
# TEXT EMBEDDING PROCESSOR
# Uses Google GenAI to map text summaries into a 768-dimensional Math Vector 
# to be saved into MongoDB Atlas Vector Search.
# ==============================================================================
import os
import asyncio
from google import genai


# Use Google GenAI SDK directly for embeddings
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


async def generate_embedding(text: str) -> list[float]:
    """
    Generate a 768-dimensional embedding vector from text using Gemini.
    Used for MongoDB Atlas Vector Search.
    """
    print("   🔢 Generating embedding vector...")
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(
                client.models.embed_content,
                model="gemini-embedding-001",
                contents=text,
            ),
            timeout=30
        )
        vector = result.embeddings[0].values
        print(f"   ✅ Embedding generated: {len(vector)} dimensions")
        return vector
    except asyncio.TimeoutError:
        print("   ❌ Embedding generation timed out after 30s")
        return []
    except Exception as e:
        print(f"   ❌ Embedding generation failed: {e}")
        return []
