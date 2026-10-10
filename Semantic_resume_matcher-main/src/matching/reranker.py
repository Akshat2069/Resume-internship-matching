from sentence_transformers import CrossEncoder

# Max words passed to the cross-encoder per document.
# The model tokenizer caps at 512 tokens anyway, so
# truncating here avoids slow tokenisation of huge texts.
_MAX_WORDS = 400


def _truncate(text: str, max_words: int = _MAX_WORDS) -> str:
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

        Returns a cross-encoder score for each resume.
        """

        jd_short = _truncate(jd_text)

        pairs = [
            [jd_short, _truncate(resume_text)]
            for resume_text in resume_texts
        ]

        scores = self.model.predict(pairs)

        return scores