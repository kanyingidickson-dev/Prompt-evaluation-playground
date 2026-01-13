from typing import Optional
from .base import BaseEvaluator, EvaluationResult

class RelevanceEvaluator(BaseEvaluator):
    """
    Heuristic-based relevance evaluator.
    Checks if keywords from the query appear in the response.
    """
    
    def evaluate(self, query: str, response_text: str, reference_answer: Optional[str] = None) -> EvaluationResult:
        # Simple stop-word filtering
        stop_words = {'what', 'is', 'how', 'to', 'the', 'a', 'an', 'in', 'of', 'for'}
        query_words = set(word.lower() for word in query.split() if word.lower() not in stop_words and len(word) > 2)
        
        if not query_words:
            return EvaluationResult(
                score=0.5,
                reasoning="Query contained only stop words or was empty.",
                evaluator_name="RelevanceHeuristic"
            )

        response_lower = response_text.lower()
        matched_words = [w for w in query_words if w in response_lower]
        score = len(matched_words) / len(query_words)
        
        # Scale to 0-10
        final_score = round(score * 10, 2)
        
        reasoning = (
            f"Found {len(matched_words)}/{len(query_words)} meaningful query terms in response. "
            f"Terms found: {matched_words}"
        )

        return EvaluationResult(
            score=final_score,
            reasoning=reasoning,
            evaluator_name="RelevanceHeuristic"
        )
