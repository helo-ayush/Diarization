from fastapi import APIRouter
from pydantic import BaseModel, Field
from utils.search_processor import vector_search

router = APIRouter(redirect_slashes=False)


class SearchRequest(BaseModel):
    query: str
    limit: int = Field(default=5, ge=1, le=50)
    all_entries: bool = Field(default=False)


@router.post("")
@router.post("/")
async def search_transcriptions(req: SearchRequest):
    """
    Search transcriptions using natural language.
    
    - Gemini optimizes the query for better embedding match
    - Generates embedding from optimized query
    - Runs MongoDB Atlas Vector Search
    - Returns top N results sorted by similarity
    """
    try:
        result = await vector_search(req.query, req.limit, req.all_entries)
        return result
    except Exception as e:
        print(f"Search Error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"Search failed: {str(e)}"}
