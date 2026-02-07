"""
GPT Researcher MCP Server

This script implements an MCP server for GPT Researcher, allowing AI assistants
to conduct web research and generate reports via the MCP protocol.
"""

import os
import sys
import uuid
import logging
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
from fastapi.responses import JSONResponse
from fastmcp import FastMCP
from gpt_researcher.retrievers.tavily import tavily_search
from gpt_researcher import GPTResearcher
from gpt_researcher.utils.enum import ReportType

# Load environment variables
load_dotenv()

from utils import (
    research_store,
    create_success_response, 
    handle_exception,
    get_researcher_by_id, 
    format_sources_for_response,
    format_context_with_sources, 
    store_research_results,
    create_research_prompt
)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s] - %(message)s',
)

logger = logging.getLogger(__name__)

TAVILY_MAX_QUERY_CHARS = int(os.getenv("TAVILY_MAX_QUERY_CHARS", "800"))


def _truncate_tavily_query(query: str) -> str:
    if not query:
        return query
    if len(query) <= TAVILY_MAX_QUERY_CHARS:
        return query
    trimmed = query[:TAVILY_MAX_QUERY_CHARS].rsplit(" ", 1)[0]
    return trimmed if trimmed else query[:TAVILY_MAX_QUERY_CHARS]


def _patch_tavily_truncation():
    original_search = tavily_search.TavilySearch._search

    def _search_with_truncation(
        self,
        query: str,
        search_depth: str = "basic",
        topic: str = "general",
        days: int = 2,
        max_results: int = 10,
        include_domains=None,
        exclude_domains=None,
        include_answer: bool = False,
        include_raw_content: bool = False,
        include_images: bool = False,
        use_cache: bool = True,
    ):
        safe_query = _truncate_tavily_query(query)
        return original_search(
            self,
            safe_query,
            search_depth=search_depth,
            topic=topic,
            days=days,
            max_results=max_results,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
            include_answer=include_answer,
            include_raw_content=include_raw_content,
            include_images=include_images,
            use_cache=use_cache,
        )

    tavily_search.TavilySearch._search = _search_with_truncation


_patch_tavily_truncation()

# Initialize FastMCP server
mcp = FastMCP(
    name="GPT Researcher"
)

# Initialize researchers dictionary
if not hasattr(mcp, "researchers"):
    mcp.researchers = {}

# Simple in-memory progress tracking keyed by research_id
if not hasattr(mcp, "research_status"):
    mcp.research_status = {}


@mcp.resource("research://{topic}")
async def research_resource(topic: str) -> str:
    """
    Provide research context for a given topic directly as a resource.
    
    This allows LLMs to access web-sourced information without explicit function calls.
    
    Args:
        topic: The research topic or query
        
    Returns:
        String containing the research context with source information
    """
    # Check if we've already researched this topic
    if topic in research_store:
        logger.info(f"Returning cached research for topic: {topic}")
        return research_store[topic]["context"]
    
    # If not, conduct the research
    logger.info(f"Conducting new research for resource on topic: {topic}")
    
    # Initialize GPT Researcher
    researcher = GPTResearcher(topic)
    
    try:
        # Conduct the research
        await researcher.conduct_research()
        
        # Get the context and sources
        context = researcher.get_research_context()
        sources = researcher.get_research_sources()
        source_urls = researcher.get_source_urls()
        
        # Format with sources included
        formatted_context = format_context_with_sources(topic, context, sources)
        
        # Store for future use
        store_research_results(topic, context, sources, source_urls, formatted_context)
        
        return formatted_context
    except Exception as e:
        return f"Error conducting research on '{topic}': {str(e)}"


@mcp.tool()
async def deep_research(query: str) -> Dict[str, Any]:
    """
    Conduct a web deep research on a given query using GPT Researcher. 
    Use this tool when you need time-sensitive, real-time information like stock prices, news, people, specific knowledge, etc.
    
    Args:
        query: The research query or topic
        
    Returns:
        Dict containing research status, ID, and the actual research context and sources
        that can be used directly by LLMs for context enrichment
    """
    logger.info(f"Conducting research on query: {query}...")
    
    # Generate a unique ID for this research session
    research_id = str(uuid.uuid4())
    mcp.research_status[research_id] = {
        "status": "running",
        "query": query,
        "progress": 0.0,
        "message": "Starting research"
    }
    
    # Initialize GPT Researcher
    safe_query = _truncate_tavily_query(query)
    researcher = GPTResearcher(safe_query)
    
    # Start research
    try:
        await researcher.conduct_research()
        mcp.research_status[research_id].update({
            "status": "completed",
            "progress": 100.0,
            "message": "Research completed"
        })
        mcp.researchers[research_id] = researcher
        logger.info(f"Research completed for ID: {research_id}")
        
        # Get the research context and sources
        context = researcher.get_research_context()
        sources = researcher.get_research_sources()
        source_urls = researcher.get_source_urls()
        
        # Store in the research store for the resource API
        store_research_results(query, context, sources, source_urls)
        
        return create_success_response({
            "research_id": research_id,
            "query": query,
            "source_count": len(sources),
            "context": context,
            "sources": format_sources_for_response(sources),
            "source_urls": source_urls
        })
    except Exception as e:
        mcp.research_status[research_id].update({
            "status": "error",
            "progress": 0.0,
            "message": str(e)
        })
        return handle_exception(e, "Research")


@mcp.tool()
async def source_research(
    query: str,
    deep: bool = False,
    deep_breadth: Optional[int] = None,
    deep_depth: Optional[int] = None,
    deep_concurrency: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Conduct research focused on identifying sources only (resource report).
    Returns sources and URLs without generating a narrative report.
    """
    safe_query = _truncate_tavily_query(query)
    logger.info(f"Conducting source-only research on query: {safe_query}...")

    research_id = str(uuid.uuid4())
    mcp.research_status[research_id] = {
        "status": "running",
        "query": query,
        "progress": 0.0,
        "message": "Starting source research"
    }
    if deep:
        if deep_breadth is not None:
            os.environ["DEEP_RESEARCH_BREADTH"] = str(deep_breadth)
        if deep_depth is not None:
            os.environ["DEEP_RESEARCH_DEPTH"] = str(deep_depth)
        if deep_concurrency is not None:
            os.environ["DEEP_RESEARCH_CONCURRENCY"] = str(deep_concurrency)
        researcher = GPTResearcher(safe_query, report_type=ReportType.DeepResearch.value)
    else:
        researcher = GPTResearcher(safe_query, report_type=ReportType.ResourceReport.value)

    try:
        if deep:
            def _progress(p):
                total = max(1, getattr(p, "total_queries", 0))
                completed = getattr(p, "completed_queries", 0)
                current = getattr(p, "current_query", "")
                progress = min(99.0, (completed / total) * 100.0)
                mcp.research_status[research_id].update({
                    "progress": progress,
                    "message": current or "Deep research in progress"
                })

            await researcher.conduct_research(on_progress=_progress)
        else:
            await researcher.conduct_research()
        mcp.research_status[research_id].update({
            "status": "completed",
            "progress": 100.0,
            "message": "Source research completed"
        })
        mcp.researchers[research_id] = researcher
        logger.info(f"Source research completed for ID: {research_id}")

        sources = researcher.get_research_sources()
        source_urls = researcher.get_source_urls()
        context = researcher.get_research_context()

        store_research_results(query, context, sources, source_urls)

        return create_success_response({
            "research_id": research_id,
            "query": query,
            "source_count": len(sources),
            "sources": format_sources_for_response(sources),
            "source_urls": source_urls
        })
    except Exception as e:
        mcp.research_status[research_id].update({
            "status": "error",
            "progress": 0.0,
            "message": str(e)
        })
        return handle_exception(e, "Source research")


@mcp.tool()
async def get_research_status(research_id: str) -> Dict[str, Any]:
    """
    Return status/progress for a research_id.
    """
    status = mcp.research_status.get(research_id)
    if not status:
        return handle_exception(ValueError("Unknown research_id"), "Status")
    return create_success_response(status)


@mcp.tool()
async def quick_search(query: str) -> Dict[str, Any]:
    """
    Perform a quick web search on a given query and return search results with snippets.
    This optimizes for speed over quality and is useful when an LLM doesn't need in-depth
    information on a topic.
    
    Args:
        query: The search query
        
    Returns:
        Dict containing search results and snippets
    """
    logger.info(f"Performing quick search on query: {query}...")
    
    # Generate a unique ID for this search session
    search_id = str(uuid.uuid4())
    
    # Initialize GPT Researcher
    researcher = GPTResearcher(query)
    
    try:
        # Perform quick search
        search_results = await researcher.quick_search(query=query)
        mcp.researchers[search_id] = researcher
        logger.info(f"Quick search completed for ID: {search_id}")
        
        return create_success_response({
            "search_id": search_id,
            "query": query,
            "result_count": len(search_results) if search_results else 0,
            "search_results": search_results
        })
    except Exception as e:
        return handle_exception(e, "Quick search")


@mcp.tool()
async def write_report(research_id: str, custom_prompt: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate a report based on previously conducted research.
    
    Args:
        research_id: The ID of the research session from deep_research
        custom_prompt: Optional custom prompt for report generation
        
    Returns:
        Dict containing the report content and metadata
    """
    success, researcher, error = get_researcher_by_id(mcp.researchers, research_id)
    if not success:
        return error
    
    logger.info(f"Generating report for research ID: {research_id}")
    
    try:
        # Generate report
        report = await researcher.write_report(custom_prompt=custom_prompt)
        
        # Get additional information
        sources = researcher.get_research_sources()
        costs = researcher.get_costs()
        
        return create_success_response({
            "report": report,
            "source_count": len(sources),
            "costs": costs
        })
    except Exception as e:
        return handle_exception(e, "Report generation")


@mcp.tool()
async def get_research_sources(research_id: str) -> Dict[str, Any]:
    """
    Get the sources used in the research.
    
    Args:
        research_id: The ID of the research session
        
    Returns:
        Dict containing the research sources
    """
    success, researcher, error = get_researcher_by_id(mcp.researchers, research_id)
    if not success:
        return error
    
    sources = researcher.get_research_sources()
    source_urls = researcher.get_source_urls()
    
    return create_success_response({
        "sources": format_sources_for_response(sources),
        "source_urls": source_urls
    })


@mcp.tool()
async def get_research_context(research_id: str) -> Dict[str, Any]:
    """
    Get the full context of the research.
    
    Args:
        research_id: The ID of the research session
        
    Returns:
        Dict containing the research context
    """
    success, researcher, error = get_researcher_by_id(mcp.researchers, research_id)
    if not success:
        return error
    
    context = researcher.get_research_context()
    
    return create_success_response({
        "context": context
    })


@mcp.prompt()
def research_query(topic: str, goal: str, report_format: str = "research_report") -> str:
    """
    Create a research query prompt for GPT Researcher.
    
    Args:
        topic: The topic to research
        goal: The goal or specific question to answer
        report_format: The format of the report to generate
        
    Returns:
        A formatted prompt for research
    """
    return create_research_prompt(topic, goal, report_format)

@mcp.custom_route("/health", methods=["GET"])
async def health_check(request):
    return JSONResponse({"status": "healthy", "service": "mcp-server"})

def run_server():
    """Run the MCP server using FastMCP's built-in event loop handling."""
    # Check if API keys are set
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not found. Please set it in your .env file.")
        return

    # Determine transport based on environment
    transport = os.getenv("MCP_TRANSPORT", "stdio").lower()
    
    # Auto-detect Docker environment (only if transport not explicitly set)
    if (not os.getenv("MCP_TRANSPORT")) and (os.path.exists("/.dockerenv") or os.getenv("DOCKER_CONTAINER")):
        transport = "sse"
        logger.info("Docker environment detected, using SSE transport")
    
    # Add startup message
    logger.info(f"Starting GPT Researcher MCP Server with {transport} transport...")
    print(f"🚀 GPT Researcher MCP Server starting with {transport} transport...")
    print("   Check researcher_mcp_server.log for details")

    # Let FastMCP handle the event loop
    try:
        if transport == "stdio":
            logger.info("Using STDIO transport (Claude Desktop compatible)")
            mcp.run(transport="stdio")
        elif transport == "sse":
            mcp.run(transport="sse", host="0.0.0.0", port=8000)
        elif transport == "streamable-http":
            mcp.run(transport="streamable-http", host="0.0.0.0", port=8000)
        else:
            raise ValueError(f"Unsupported transport: {transport}")
            
        # Note: If we reach here, the server has stopped
        logger.info("MCP Server is running...")
        while True:
            pass  # Keep the process alive
    except Exception as e:
        logger.error(f"Error running MCP server: {str(e)}")
        print(f"❌ MCP Server error: {str(e)}")
        return
        
    print("✅ MCP Server stopped")


if __name__ == "__main__":
    # Use the non-async approach to avoid asyncio nesting issues
    run_server()
