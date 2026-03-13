import requests
from bs4 import BeautifulSoup
from functools import lru_cache

from langchain_community.retrievers import WikipediaRetriever
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_tavily import TavilySearch
from langchain_community.tools.semanticscholar.tool import SemanticScholarQueryRun
from langchain_community.retrievers import PubMedRetriever
from langchain.tools import tool

# Lazy singletons
# Instantiated on first use, not at import time, so load_dotenv() in main.py
# runs first and all API keys are available.

@lru_cache(maxsize=1)
def _get_wiki():
    return WikipediaRetriever(load_max_docs=3)

@lru_cache(maxsize=1)
def _get_tavily():
    return TavilySearch(
        max_results=5,
        topic="general",
        include_answer=True,        # returns a synthesized direct answer on top of links
        include_raw_content=False,  # keep False for search agent; set True selectively in writer/critique
        include_images=False,
    )

@lru_cache(maxsize=1)
def _get_ddg():
    return DuckDuckGoSearchResults(output_format="list")

@lru_cache(maxsize=1)
def _get_semantic_scholar():
    return SemanticScholarQueryRun()

@lru_cache(maxsize=1)
def _get_pubmed():
    return PubMedRetriever(load_max_docs=3)

# Tools

@tool("wikipedia_search")
def wikipedia_search(query: str) -> str:
    """
    Search Wikipedia for general background knowledge, definitions,
    and historical information. Best for foundational or academic topics.
    Returns content from up to 3 Wikipedia pages.
    """
    try:
        docs = _get_wiki().invoke(query)
        if not docs:
            return f"No Wikipedia results found for: {query}"
        sections = []
        for i, doc in enumerate(docs, 1):
            title = doc.metadata.get("title", f"Result {i}")
            sections.append(f"[Wikipedia Page {i}: {title}]\n{doc.page_content}")
        return "\n\n---\n\n".join(sections)
    except Exception as e:
        return f"Wikipedia search failed for '{query}': {str(e)}"


@tool("tavily_search")
def tavily_search(query: str) -> str:
    """
    Search the web using Tavily — the primary web search tool.
    Returns high-quality, ranked results with a direct synthesized answer.
    Use for recent events, statistics, news, and general web research.
    Falls back to DuckDuckGo automatically if Tavily fails.
    """
    try:
        results = _get_tavily().invoke({"query": query})
        if not results:
            raise ValueError("No results returned from Tavily.")

        lines = []
        # Tavily returns a list of dicts with keys: url, content, (optionally) answer
        if isinstance(results, list):
            for item in results:
                url     = item.get("url", "")
                content = item.get("content", "")
                lines.append(f"URL: {url}\nSummary: {content}")
        elif isinstance(results, str):
            # Some versions return a pre-formatted string
            return results

        return "\n\n".join(lines)

    except Exception as tavily_err:
        # DDG fallback
        print(f"  [tavily_search] Tavily failed ({tavily_err}), falling back to DuckDuckGo...")
        try:
            ddg_results: list = _get_ddg().invoke(query)
            if not ddg_results:
                return f"No results found for: {query} (both Tavily and DuckDuckGo failed or returned nothing)"
            lines = []
            for item in ddg_results:
                title   = item.get("title", "No title")
                link    = item.get("link", item.get("href", ""))
                snippet = item.get("snippet", item.get("body", ""))
                lines.append(f"Title: {title}\nURL: {link}\nSnippet: {snippet}")
            return "[Source: DuckDuckGo fallback]\n\n" + "\n\n".join(lines)
        except Exception as ddg_err:
            return f"Web search failed for '{query}': Tavily error: {tavily_err} | DDG error: {ddg_err}"


@tool("scrape_url")
def scrape_url(url: str) -> str:
    """
    Scrape and extract the main text content from a given URL.
    Use when you have a specific URL and need its full page content
    for deeper analysis beyond search snippets.
    Returns up to 3000 characters of cleaned page text.
    """
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/91.0.4472.124 Safari/537.36"
            )
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.extract()
        text = soup.get_text(separator=" ")
        lines  = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text   = "\n".join(chunk for chunk in chunks if chunk)
        return text[:3000]
    except Exception as e:
        return f"Failed to scrape {url}: {str(e)}"


@tool("semantic_scholar_search")
def semantic_scholar_search(query: str) -> str:
    """
    Search Semantic Scholar for peer-reviewed academic papers.
    Covers computer science, social sciences, economics, and more —
    complementing arXiv which skews toward physics and ML preprints.
    Use for literature reviews, citation counts, and finding established research.
    """
    try:
        result = _get_semantic_scholar().invoke(query)
        if not result:
            return f"No Semantic Scholar results found for: {query}"
        return result
    except Exception as e:
        return f"Semantic Scholar search failed for '{query}': {str(e)}"


@tool("pubmed_search")
def pubmed_search(query: str) -> str:
    """
    Search PubMed for biomedical and life science literature.
    Use for any topic touching medicine, biology, public health,
    clinical trials, pharmacology, or neuroscience.
    Returns full abstracts and metadata from up to 3 papers.
    """
    try:
        docs = _get_pubmed().invoke(query)
        if not docs:
            return f"No PubMed results found for: {query}"

        sections = []
        for doc in docs:
            uid        = doc.metadata.get("uid", "N/A")
            title      = doc.metadata.get("Title", "No title")
            published  = doc.metadata.get("Published", "N/A")
            copyright_ = doc.metadata.get("Copyright Information", "")
            abstract   = doc.page_content.strip() or "No abstract available."

            entry = (
                f"Title: {title}\n"
                f"PubMed ID: {uid}\n"
                f"Published: {published}\n"
                f"Abstract: {abstract}"
            )
            if copyright_:
                entry += f"\nCopyright: {copyright_}"
            sections.append(entry)

        return "\n\n---\n\n".join(sections)
    except Exception as e:
        return f"PubMed search failed for '{query}': {str(e)}"


@tool("comprehensive_search")
def comprehensive_search(query: str) -> str:
    """
    Combines Wikipedia, Tavily web search, and URL scraping to find
    comprehensive information about a topic from multiple angles.
    Use as a last resort when targeted tools have failed or when
    broad multi-source coverage is explicitly needed.
    """
    result_text = f"--- Comprehensive Search: {query} ---\n\n"

    # 1. Wikipedia
    result_text += "[Wikipedia]\n"
    result_text += wikipedia_search.invoke(query) + "\n\n"

    # 2. Tavily (with DDG fallback built into tavily_search)
    result_text += "[Web Search (Tavily / DDG fallback)]\n"
    try:
        raw_results = _get_tavily().invoke({"query": query})
        if isinstance(raw_results, list) and raw_results:
            result_text += "[Web Search]\n"
            for item in raw_results:
                url     = item.get("url", "")
                content = item.get("content", "")
                result_text += f"{url}\nSummary: {content}\n"
                # Scrape each URL for deeper content
                scraped = scrape_url.invoke(url)
                result_text += f"Scraped Content: {scraped}\n\n"
        else:
            # Tavily returned a string or nothing — fall back to DDG
            raise ValueError("Tavily returned no list results.")
    except Exception:
        try:
            ddg_results: list = _get_ddg().invoke(query)
            if ddg_results:
                result_text += "[Web Search - DuckDuckGo fallback]\n"
                for item in ddg_results:
                    title   = item.get("title", "No title")
                    link    = item.get("link", item.get("href", ""))
                    snippet = item.get("snippet", item.get("body", ""))
                    result_text += f"{title} - {link}\nSnippet: {snippet}\n"
                    scraped = scrape_url.invoke(link)
                    result_text += f"Scraped Content: {scraped}\n\n"
        except Exception as e:
            result_text += f"Web search failed: {str(e)}\n\n"

    return result_text