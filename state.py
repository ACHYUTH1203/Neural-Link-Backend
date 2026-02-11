from typing import List, Optional, TypedDict
class GraphState(TypedDict):
    query: str
    intent: str  
    collections: List[str]
    documents: List[dict]
    first_principles_analysis: Optional[str]
    final_response: Optional[str]
    rag_answer: Optional[str] 
    needs_assistance: bool
    error_log: List[str]
    revision_count: int