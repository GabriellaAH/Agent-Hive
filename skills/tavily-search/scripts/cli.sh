#!/bin/bash
# Simple Tavily CLI using curl
API_KEY="${TAVILY_API_KEY}"
BASE_URL="https://api.tavily.com"

cmd="$1"
shift || true

if [ "$cmd" == "search" ]; then
  query="$*"
  curl -s --request POST \
    --url "$BASE_URL/search" \
    --header "Authorization: Bearer $API_KEY" \
    --header 'Content-Type: application/json' \
    --data "{ \"query\": \"${query}\", \"auto_parameters\": false, \"topic\": \"general\", \"search_depth\": \"basic\", \"chunks_per_source\": 3, \"max_results\": 5, \"include_answer\": false }"
elif [ "$cmd" == "extract" ]; then
  url="$1"
  curl -s --request POST \
    --url "$BASE_URL/extract" \
    --header "Authorization: Bearer $API_KEY" \
    --header 'Content-Type: application/json' \
    --data "{ \"urls\": \"${url}\", \"chunks_per_source\": 3, \"extract_depth\": \"basic\", \"format\": \"markdown\", \"include_images\": false }"
else
  echo "Usage: $0 {search <query> | extract <url>}"
fi
