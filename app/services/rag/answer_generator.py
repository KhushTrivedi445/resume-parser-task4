def generate_answer(retrieved_chunks: list[str]) -> str:
    if not retrieved_chunks:
        return "No relevant job-description content was found."
    return "Relevant job-description content: " + " ".join(retrieved_chunks)
