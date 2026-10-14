sentences = [
    "FastAPI is a Python framework for building APIs.",
    "Flask is a lightweight Python web framework.",
    "Django is a Python framework for web development.",
    "Python is a popular programming language.",
    "JavaScript is widely used for web development.",
    "TypeScript adds static typing to JavaScript.",
    "Machine learning allows computers to learn from data.",
    "Deep learning uses neural networks with multiple layers.",
    "Artificial intelligence enables machines to perform intelligent tasks.",
    "Generative AI can create text, images, and code.",
    "Embeddings represent text as numerical vectors.",
    "Vector databases store and search embeddings.",
    "Semantic search finds information based on meaning.",
    "RAG combines retrieval with language generation.",
    "APIs allow different software systems to communicate.",
    "PostgreSQL is a relational database.",
    "ChromaDB is a vector database for AI applications.",
    "Git is a distributed version control system.",
    "Docker is used to package applications into containers.",
    "Cloud computing provides computing resources over the internet."
]

# sentence transformer from text to vectors 
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(sentences)

print(embeddings.shape)

#chromadb importing 
import chromadb

client = chromadb.Client()

collection = client.create_collection("day8_documents")

collection.add(
    ids=[str(i) for i in range(20)],
    documents=sentences,
    embeddings=embeddings.tolist()
)

query = "Which Python technology can I use to build APIs?"

query_embedding = model.encode(query)

results = collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=3
)

print("Documents:")
print(results["documents"])

print("\nDistances:")
print(results["distances"])