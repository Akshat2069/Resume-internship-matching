import numpy as np
from sentence_transformers import CrossEncoder

# Break text into overlapping chunks so the AI reads the entire resume
_CHUNK_WORDS = 300
_OVERLAP_WORDS = 50

def _chunk_text(text: str) -> list[str]:
    words = text.split()
    if len(words) <= _CHUNK_WORDS:
        return [text]
    
    chunks = []
    for i in range(0, len(words), _CHUNK_WORDS - _OVERLAP_WORDS):
        chunk = " ".join(words[i:i + _CHUNK_WORDS])
        chunks.append(chunk)
    return chunks

def _truncate_jd(text: str, max_words: int = 300) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])

class ResumeReranker:
    def __init__(self):
        self.model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

    def rerank(self, jd_text, resume_texts):
        """
        Compare one JD against multiple retrieved resumes.
        Instead of truncating the resume, it chunks it and 
        returns the highest cross-encoder score across all chunks.
        """
        jd_short = _truncate_jd(jd_text)

        final_scores = []
        for resume_text in resume_texts:
            chunks = _chunk_text(resume_text)
            pairs = [[jd_short, chunk] for chunk in chunks]
            
            # Predict scores for all chunks of this single resume
            chunk_scores = self.model.predict(pairs)
            
            # The resume's score is the score of its best-matching chunk
            final_scores.append(float(np.max(chunk_scores)))

        return final_scores