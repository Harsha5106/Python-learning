from sentence_transformers import SentenceTransformer
import chromadb


# 1. Read the document
with open("sample.txt", "r") as file:
    text = file.read()


# 2. Recursive chunking
def recursive_chunk(text, chunk_size):
    paragraphs = text.split("\n\n")

    chunks = []

    for paragraph in paragraphs:

        if len(paragraph) <= chunk_size:
            chunks.append(paragraph)

        else:
            sentences = paragraph.split(". ")

            current_chunk = ""

            for sentence in sentences:
                if len(current_chunk) + len(sentence) <= chunk_size:
                    current_chunk += sentence + ". "
                else:
                    if current_chunk:
                        chunks.append(current_chunk)

                    current_chunk = sentence + ". "

            if current_chunk:
                chunks.append(current_chunk)

    return chunks


# 3. Create chunks
chunks = recursive_chunk(text, 100)

print("\nRecursive Chunks:")

for i, chunk in enumerate(chunks):
    print(f"\nChunk {i + 1}:")
    print(chunk)


# 4. Create embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(chunks)

print("\nNumber of chunks:", len(chunks))
print("Embedding shape:", embeddings.shape)


# 5. Create ChromaDB collection
client = chromadb.Client()

collection = client.create_collection("day9_rag")


# 6. Store chunks and embeddings
collection.add(
    ids=[str(i) for i in range(len(chunks))],
    documents=chunks,
    embeddings=embeddings.tolist()
)

print("\nChunks successfully stored in ChromaDB.")