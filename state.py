
from typing import TypedDict, Annotated, List, Dict, Optional
import operator

class GraphState(TypedDict, total=False):
    query: str
    final_response: str
    needs_assistance: bool
    revision_count: int
    rag_docs:Optional [str]
    error_log: List[str]
    web_results:Optional [List]
    validation_score: Optional[float]