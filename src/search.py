"""
Live web search integration. Uses DuckDuckGo (no API key required) to pull
recent sports news so the quiz generator isn't limited to static offline facts.
"""

from ddgs import DDGS
from ddgs.exceptions import DDGSException


def get_live_news_context(sport_name: str, max_results: int = 3) -> str:
    """
    Searches the live web for recent news/results for a given sport.
    Returns a joined block of text summarizing the top results, or a
    graceful fallback message if the search fails (rate limit, no
    connection, etc.) so the rest of the pipeline can keep running.
    """
    search_query = f"{sport_name} latest tournament results championship winners news"
    retrieved_texts = []

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(search_query, max_results=max_results))

        if not results:
            return "No recent live search results were found for this sport."

        for index, r in enumerate(results, start=1):
            title = r.get("title", "No Title")
            snippet = r.get("body", "No snippet available")
            retrieved_texts.append(f"Web Source {index}: {title}\nSnippet: {snippet}")

    except DDGSException as e:
        return f"Live web search unavailable right now ({e}). Falling back to offline facts only."
    except Exception as e:
        return f"Live web search failed unexpectedly ({e}). Falling back to offline facts only."

    return "\n\n".join(retrieved_texts)
