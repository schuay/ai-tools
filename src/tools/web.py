"""Web search and fetch tools for use by LangChain/LangGraph agents."""

import os

import httpx
import trafilatura
from tavily import TavilyClient


def web_search(query: str, max_results: int = 5) -> str:
    """Search the web for real-time information, latest news, or technical documentation.

    Use this when you need information that isn't in the local codebase, such as
    external library documentation, blog posts, bug reports, or spec proposals.

    query: the search query string
    max_results: maximum number of results to return (default 5)
    """
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return "Error: TAVILY_API_KEY environment variable is not set."
    try:
        results = (
            TavilyClient(api_key=api_key)
            .search(query=query, search_depth="basic", max_results=max_results)
            .get("results", [])
        )
        return (
            "\n".join(
                f"Title: {r['title']}\nURL: {r['url']}\nSnippet: {r['content']}\n---"
                for r in results
            )
            or "No results found."
        )
    except Exception as e:
        return f"Error performing web search: {e}"


def web_fetch(url: str) -> str:
    """Fetch and extract the main text content from a specific URL.

    Use this to read the full content of a webpage (blog post, documentation page,
    bug report) after finding a promising URL via web_search.
    Strips navigation, ads, and boilerplate automatically.

    url: the full URL of the page to fetch
    """
    try:
        with httpx.Client(follow_redirects=True, timeout=15.0) as client:
            content = trafilatura.extract(
                client.get(url).raise_for_status().text,
                include_links=True,
                include_comments=False,
            )
        if not content:
            return "Error: could not extract meaningful content from the page."
        return (
            content[:30000] + "\n… (content truncated for length)"
            if len(content) > 30000
            else content
        )
    except httpx.HTTPStatusError as e:
        return f"HTTP {e.response.status_code} fetching {url}"
    except Exception as e:
        return f"Error fetching {url}: {e}"
