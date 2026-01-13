from typing import Optional
from .base import BaseEvaluator, EvaluationResult

class ClarityEvaluator(BaseEvaluator):
    """
    Checks for structure and readability.
    """
    
    def evaluate(self, query: str, response_text: str, reference_answer: Optional[str] = None) -> EvaluationResult:
        # Heuristics: Bullet points, moderate length, no all-caps
        lines = response_text.strip().split('\n')
        has_bullets = any(line.strip().startswith(('-', '*', '1.')) for line in lines)
        length_ok = 50 < len(response_text) < 2000
        
        score = 5.0
        details = []
        
        if has_bullets:
            score += 2.0
            details.append("Has bullet points")
        if length_ok:
            score += 3.0
            details.append("Good length")
        
        return EvaluationResult(
            score=min(score, 10.0),
            reasoning=f"Clarity checks: {', '.join(details)}",
            evaluator_name="ClarityHeuristic"
        )
