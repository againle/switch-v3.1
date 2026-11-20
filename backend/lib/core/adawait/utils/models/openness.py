import re

class OpennessScore:
     
    def __init__(self):
        self.patterns = [ 
            (r"medium-high|moderate-high|mid-high|medium\s*-\s*high", 0.75),
            (r"medium-low|mid-low|moderate-low|low\s*-\s*medium|low\s*-\s*moderate", 0.25),
            (r"low", 0.10),
            (r"medium|mid|average|moderate", 0.50),
            (r"high", 0.90)
        ]
    
    def score(self, openness_text: str) -> float:
        """
        Parse a text description of openness into a score.

        Args:
            openness_text (str): A text description of openness.
        Returns:
            float: A score representing the level of openness, normalized between 0 and 1.
        """
        # Simple keyword-based scoring for demonstration purposes
        # Order: medium-high, medium-low, low, medium, high
        

        score = 0.5  # Default neutral score
        for pattern, value in self.patterns:
            if re.search(pattern, openness_text.lower()):
                score = value
                break

        return score