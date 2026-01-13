from typing import Optional
from .base import BaseEvaluator, EvaluationResult
from difflib import SequenceMatcher

class AccuracyEvaluator(BaseEvaluator):
    """
    Heuristic accuracy checker.
    Compares response to a reference answer using sequence matching (Levenshtein distance proxy).
    """
    
    def evaluate(self, query: str, response_text: str, reference_answer: Optional[str] = None) -> EvaluationResult:
        if not reference_answer:
            return EvaluationResult(
                score=0.0,
                reasoning="No reference answer provided for accuracy check.",
                evaluator_name="AccuracyHeuristic",
                metadata={"status": "skipped"}
            )
            
        similarity = SequenceMatcher(None, response_text.lower(), reference_answer.lower()).ratio()
        
        return EvaluationResult(
            score=round(similarity * 10, 2),
            reasoning=f"Similarity to reference: {similarity:.2f}",
            evaluator_name="AccuracyHeuristic"
        )
