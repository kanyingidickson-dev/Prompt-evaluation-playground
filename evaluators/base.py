from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from pydantic import BaseModel

class EvaluationResult(BaseModel):
    score: float  # Normalized 0.0 to 1.0 or scale 0-10
    reasoning: str
    evaluator_name: str
    metadata: Dict[str, Any] = {}

class BaseEvaluator(ABC):
    """
    Abstract Interface for Prompt Evaluators.
    Evaluators can be heuristic (regex, length) or Model-based (LLM-as-judge).
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    @abstractmethod
    def evaluate(self, 
                 query: str, 
                 response_text: str, 
                 reference_answer: Optional[str] = None) -> EvaluationResult:
        """
        Evaluate the response quality against a specific dimension.
        
        Args:
            query: The original input/question.
            response_text: The LLM output.
            reference_answer: (Optional) Ground truth data.
            
        Returns:
            EvaluationResult with score and explanation.
        """
        pass
