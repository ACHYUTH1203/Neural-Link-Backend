
from typing import TypedDict, Annotated, List, Dict, Optional
import operator

# class GraphState(TypedDict, total=False):
#     user_id: str
#     query: str
#     original_query: str       
#     final_response: str
#     conversation_mode: Optional[str]
#     clarification_question: Optional[str]
#     chat_history: List[Dict]
#     needs_assistance: bool
#     revision_count: int
#     rag_docs:Optional [str]
#     error_log: List[str]
#     web_results:Optional [List]
#     validation_score: Optional[float]
    
class GraphState(TypedDict, total=False):
    user_id: str
    query: str
    original_query: str
    final_response: str
    chat_history: List[Dict]

    clarification_count: int
    force_expand: Optional[bool]
    # NEW FIELDS
    conversation_mode: Optional[str]  # "clarify" | "answer"
    clarification_question: Optional[str]
    awaiting_clarification: bool

    needs_assistance: bool
    revision_count: int
    rag_docs: Optional[str]
    error_log: List[str]
    web_results: Optional[List]
    validation_score: Optional[float]
