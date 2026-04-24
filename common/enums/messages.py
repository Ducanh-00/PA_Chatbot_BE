class CLIMessages:
    ADVANCED_RAG_WELCOME = (
        "\n"
        + "=" * 60
        + "\nADVANCED RAG SYSTEM (Hybrid + Rerank + Rewrite)\nEnter 'exit' or 'q' to quit.\n"
        + "=" * 60
        + "\n"
    )
    BASIC_RAG_WELCOME = (
        "\n"
        + "=" * 60
        + "\nPA Chatbot System (Using Gemini API)\nType 'exit' or 'q' to quit.\n"
        + "=" * 60
        + "\n"
    )
    RETRIEVAL_WELCOME = "\nRETRIEVAL TEST SYSTEM\nType 'exit' or 'q' to quit.\n"

    PROMPT_USER = "You: "
    PROMPT_QUESTION = "Enter your question: "

    ANALYZING = "Analyzing question..."
    REWRITTEN_TO = "   -> Rewritten question: {}"
    SEARCHING_HYBRID = "Searching (Hybrid Search) & Re-ranking..."
    SEARCHING_BASIC = "Searching in documents...\n"
    SEARCHING_RETRIEVAL = "Searching in 149 documents...\n"

    AI_THINKING = "AI is thinking...\n"
    AI_GENERATING = "AI is generating answer...\n"

    SOURCES_HEADER = "Sources:"
    CITATIONS_HEADER = "Citations from:"
    NO_RESULTS = "No relevant information found."
    ERROR = "Error: {}"
