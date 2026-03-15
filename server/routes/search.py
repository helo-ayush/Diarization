# ==============================================================================
# SEARCH ROUTE
# This file handles Natural Language / Vector Search against stored transcripts.
# ==============================================================================
from fastapi import APIRouter
from pydantic import BaseModel, Field
from utils.search_processor import vector_search

router = APIRouter(redirect_slashes=False)

# Defines the expected JSON schema for search requests
class SearchRequest(BaseModel):
    query: str
    limit: int = Field(default=5, ge=1, le=50) # Strict validation: between 1 and 50 results
    all_entries: bool = Field(default=False)


@router.post("")
@router.post("/")
async def search_transcriptions(req: SearchRequest):
    """
    Search transcriptions using natural language (RAG system).
    
    Workflow:
    1. Gemini parses the user's natural language request to optimize it.
    2. Gemini generates an embedding vector from the optimized query.
    3. The vector is passed to MongoDB Atlas Vector Search.
    4. Atlas performs a Cosine Similarity match against all saved transcript summaries.
    5. Returns the top N most relevant results.
    """
    try:
        result = await vector_search(req.query, req.limit, req.all_entries)
        return result
    except Exception as e:
        print(f"Search Error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"Search failed: {str(e)}"}
