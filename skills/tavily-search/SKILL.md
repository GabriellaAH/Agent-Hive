---
name: tavily-search
description: "Provide web access and search capabilities"
metadata:
---

# tavily-search skill

This skill provides simple wrappers and examples for using the Tavily web search / extract / crawl / research APIs.

Features:
- curl examples for Search and Extract (ready to run with the provided API key)
- small CLI script (bash) that calls the Search API and returns JSON
- examples for Python and Node (templates)

Usage notes:
- This skill contains an example script that includes the API key you provided. Keep the workspace secure.
- Primary quick path: use the curl script in cli.sh for lightweight searches without installing extra dependencies.

APIs covered: /search, /extract, /crawl, /map, /research

Examples:
- bash: ./cli.sh search "trend following"
- python: see python_client.py (requires tavily Python SDK if you want to use it)

