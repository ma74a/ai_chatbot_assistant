from retriever import retrieve_chunks


query = "What is cross entropy loss?"

result = retrieve_chunks(query)

print(f"\nQuery: {query}\n")


for doc, score in result:

    print("=" * 50)


    print("=" * 50)
    print(f"score: {score}")

    print(doc.page_content)

    print("\nMetadata:")
    print(doc.metadata)

    print("\n")