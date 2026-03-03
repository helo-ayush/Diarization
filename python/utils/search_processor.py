import os
import asyncio
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from database import db
from utils.embedding_processor import generate_embedding


# Gemini for query optimization
llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.1,
)

# MongoDB collection
transcriptions = db["transcriptions"]


async def optimize_query(user_query: str) -> str:
    """
    Use Gemini to convert a natural language query into a
    search-optimized text that will produce a better embedding
    for similarity matching against our stored summaries.
    """
    prompt = f"""You are a search query optimizer for a call transcription database. 
Each record has a summary describing: the customer issue, the resolution, technical entities, and satisfaction level.

Convert this user's natural language query into a dense, keyword-rich search phrase that will match well against those summaries. 

Rules:
- Output ONLY the optimized query text, nothing else
- Include relevant synonyms and related terms
- Keep it under 50 words
- Focus on the core intent

User query: "{user_query}"

Optimized search query:"""

    try:
        message = HumanMessage(content=prompt)
        response = await asyncio.wait_for(
            asyncio.to_thread(llm.invoke, [message]),
            timeout=15
        )
        optimized = response.content.strip().strip('"')
        print(f"   🔍 Optimized query: {optimized}")
        return optimized
    except Exception as e:
        print(f"   ⚠️ Query optimization failed: {e}, using original")
        return user_query


async def vector_search(query: str, limit: int = 5) -> dict:
    """
    Full search pipeline:
    1. Optimize query with Gemini
    2. Generate embedding
    3. Run MongoDB Atlas Vector Search
    4. Return results
    """
    start_time = time.time()

    # Step 1: Optimize query
    print(f"🔎 Search query: \"{query}\"")
    print("   🧠 Optimizing query with Gemini...")
    optimized_query = await optimize_query(query)

    # Step 2: Generate embedding from optimized query
    query_embedding = await generate_embedding(optimized_query)

    if not query_embedding:
        return {
            "query": query,
            "optimizedQuery": optimized_query,
            "results": [],
            "totalResults": 0,
            "error": "Failed to generate query embedding"
        }

    # Step 3: MongoDB Atlas Vector Search
    print(f"   📊 Running vector search (limit={limit})...")
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": limit * 10,  # Search wider pool
                "limit": limit,
            }
        },
        {
            "$project": {
                "_id": {"$toString": "$_id"},
                "filename": 1,
                "summary": 1,
                "transcript": 1,
                "satisfactionScore": 1,
                "tags": 1,
                "detectedRoles": 1,
                "speakerCount": 1,
                "createdAt": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        }
    ]

    results = []
    async for doc in transcriptions.aggregate(pipeline):
        print(f"      📎 {doc.get('filename', '?')} → score: {doc.get('score', 0):.4f}")
        results.append(doc)

    # Filter out low-relevance results
    MIN_SCORE = 0.80
    filtered = [r for r in results if r.get("score", 0) >= MIN_SCORE]
    dropped = len(results) - len(filtered)
    if dropped:
        print(f"   🚫 Filtered out {dropped} low-relevance results (score < {MIN_SCORE})")

    elapsed = int((time.time() - start_time) * 1000)
    print(f"   ✅ Found {len(filtered)} relevant results in {elapsed}ms")

    return {
        "query": query,
        "optimizedQuery": optimized_query,
        "results": filtered,
        "totalResults": len(filtered),
        "processingMs": elapsed,
    }
