"""Universe knowledgebase gathering agent.

This script provides a small tool-calling loop with two local tools:
- read_file(path): read text files from disk
- search_hard_drive(query, root_path, max_results): search file names/content
"""

import json
import os
import argparse
from pathlib import Path

from groq import Groq

MODEL = "openai/gpt-oss-120b"
MAX_FILE_READ_CHARS = 20_000
MAX_FILE_SCAN_BYTES = 512_000

client = Groq()


def read_file(path: str) -> str:
    """Read a UTF-8 text file and return content or an error."""
    try:
        file_path = Path(path).expanduser().resolve()
        if not file_path.exists():
            return f"File not found at {file_path}"
        if not file_path.is_file():
            return f"Path is not a file: {file_path}"

        text = file_path.read_text(encoding="utf-8", errors="replace")
        if len(text) > MAX_FILE_READ_CHARS:
            text = text[:MAX_FILE_READ_CHARS] + "\n\n[truncated]"
        return text
    except Exception as exc:
        return f"Error reading file: {exc}"


def search_hard_drive(query: str, root_path: str = ".", max_results: int = 20) -> str:
    """Search for files whose name or text content matches query."""
    try:
        root = Path(root_path).expanduser().resolve()
        if not root.exists() or not root.is_dir():
            return json.dumps({"error": f"Invalid directory: {root_path}"})

        q = query.lower().strip()
        if not q:
            return json.dumps({"error": "Query cannot be empty."})

        results = []
        for file_path in root.rglob("*"):
            if len(results) >= max_results:
                break
            if not file_path.is_file():
                continue

            rel = str(file_path.relative_to(root))
            if q in rel.lower():
                results.append({"path": rel, "match_type": "filename"})
                continue

            try:
                if file_path.stat().st_size > MAX_FILE_SCAN_BYTES:
                    continue
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                if q in content.lower():
                    results.append({"path": rel, "match_type": "content"})
            except Exception:
                continue

        return json.dumps({"query": query, "count": len(results), "results": results}, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


available_functions = {
    "read_file": read_file,
    "search_hard_drive": search_hard_drive,
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file at a given path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute or relative file path."}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_hard_drive",
            "description": "Search files by name and content under a root directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Text to search for."},
                    "root_path": {
                        "type": "string",
                        "description": "Directory to search from. Defaults to current directory.",
                        "default": ".",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of matches to return.",
                        "default": 20,
                    },
                },
                "required": ["query"],
            },
        },
    },
]


def execute_tool_call(tool_call) -> str:
    """Execute one model-requested tool call."""
    function_name = tool_call.function.name
    function_args = json.loads(tool_call.function.arguments or "{}")
    function_to_call = available_functions.get(function_name)
    if not function_to_call:
        return json.dumps({"error": f"Unknown tool: {function_name}"})
    return function_to_call(**function_args)


def run_agent(user_query: str, max_iterations: int = 8) -> str:
    """Run a tool-calling loop and return the final model answer."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a universe knowledgebase gathering assistant. "
                "Use tools to inspect local files and summarize findings."
            ),
        },
        {"role": "user", "content": user_query},
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    iterations = 0
    while response.choices[0].message.tool_calls and iterations < max_iterations:
        iterations += 1
        assistant_message = response.choices[0].message
        messages.append(assistant_message)

        for tool_call in assistant_message.tool_calls:
            tool_result = execute_tool_call(tool_call)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.function.name,
                    "content": tool_result,
                }
            )

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )

    final_text = response.choices[0].message.content or ""
    return final_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Universe knowledgebase manager")
    parser.add_argument(
        "--read",
        dest="read_path",
        help="Read and print a file directly (bypasses model call).",
    )
    parser.add_argument(
        "--query",
        dest="query",
        help="Prompt for the model-driven tool-calling agent.",
    )
    args = parser.parse_args()

    if args.read_path:
        print(read_file(args.read_path))
        raise SystemExit(0)

    default_query = (
        "Search this project for universe-related files and produce a concise "
        "worldbuilding knowledge summary."
    )
    query = args.query or os.getenv("UNIVERSE_QUERY", default_query)
    answer = run_agent(query)
    print(answer)