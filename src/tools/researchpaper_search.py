from langchain_community.retrievers import ArxivRetriever
from langchain.tools import tool

retriever = ArxivRetriever(
    load_max_docs=3,
    get_full_documents=True,
)

_seen_paper_ids: set = set()

# NOT USED : Keep in case of future use in a long-running scenario
def reset_arxiv_cache() -> None:
    """Call this at the start of a new research task to clear the dedup cache."""
    _seen_paper_ids.clear()


@tool("arxiv_search")
def arxiv_search(query: str) -> str:
    """
    Search arXiv for academic papers and research articles.
    Best for scientific topics, technical research, and cutting-edge studies.
    Returns a formatted string of results (title, authors, abstract, content).
    Already-seen papers are automatically skipped across multiple queries.
    """
    docs = retriever.invoke(query)
    results = []
    skipped = 0

    for doc in docs:
        entry_id = doc.metadata.get("Entry ID", "")

        if entry_id in _seen_paper_ids:
            skipped += 1
            continue

        _seen_paper_ids.add(entry_id)

        title   = doc.metadata.get("Title", "No title")
        authors = doc.metadata.get("Authors", "Unknown authors")
        results.append(
            f"Title: {title}\n"
            f"Authors: {authors}\n"
            f"URL: {entry_id}\n"
            f"Content:\n{doc.page_content}"
        )

    if skipped:
        print(f"  [arxiv_search] Skipped {skipped} duplicate paper(s) for query: {query!r}")

    return "\n\n---\n\n".join(results) if results else "No new arXiv results found."