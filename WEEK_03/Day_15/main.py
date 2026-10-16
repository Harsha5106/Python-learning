import re
from collections.abc import Iterable

from nltk.stem import PorterStemmer
from nltk.tokenize import TreebankWordTokenizer


STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "as", "at", "be",     "because", "been", "before", "can",
    "being", "below", "between", "both", "but", "by", "could", "did", "do",
    "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers",
    "herself", "him", "himself", "his", "how", "i", "if", "in", "into",
    "is", "it", "its", "itself", "just", "me", "more", "most", "my",
    "myself", "no", "nor", "not", "now", "of", "off", "on", "once", "only",
    "or", "other", "our", "ours", "ourselves", "out", "over", "own", "same",
    "she", "should", "so", "some", "such", "than", "that", "the", "their",
    "theirs", "them", "themselves", "then", "there", "these", "they", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was",
    "we", "were", "what", "when", "where", "which", "while", "who", "whom",
    "why", "with", "would", "you", "your", "yours", "yourself", "yourselves",
}

TOKENIZER = TreebankWordTokenizer()
STEMMER = PorterStemmer()


def preprocess_corpus(
    corpus: Iterable[str],
) -> tuple[list[str], list[str]]:
    """Return lowercased raw tokens and cleaned, stemmed tokens."""
    raw_tokens = []
    cleaned_tokens = []

    for text in corpus:
        tokens = TOKENIZER.tokenize(text.lower())
        raw_tokens.extend(tokens)
        cleaned_tokens.extend(
            STEMMER.stem(token)
            for token in tokens
            if re.fullmatch(r"[a-z]+", token) and token not in STOP_WORDS
        )

    return raw_tokens, cleaned_tokens


def main() -> None:
    corpus = [
        "Natural Language Processing helps computers understand human language.",
        "The researchers are studying language models and processing text.",
        "Large language models can generate useful answers from text.",
    ]

    raw_tokens, cleaned_tokens = preprocess_corpus(corpus)

    print("NLP TEXT PREPROCESSING")
    print("=" * 40)
    print("\nSample corpus:")
    for sentence in corpus:
        print(f"- {sentence}")

    print("\nRaw tokens (lowercased):")
    print(raw_tokens)
    print(f"Raw token count: {len(raw_tokens)}")

    print("\nCleaned tokens (stopwords/punctuation removed, words stemmed):")
    print(cleaned_tokens)
    print(f"Cleaned token count: {len(cleaned_tokens)}")


if __name__ == "__main__":
    main()
