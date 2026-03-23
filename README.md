# Agentic Playground

This repository is my learning sandbox for building agents as part of the [Intro to Agents workshop](https://themultiverse.school/classes/188) at The Multiverse School.

## Agents in this repo

- `agent.py` - basic chat-completion example using Groq-compatible OpenAI client.
- `calculator_agent.py` - tool-calling financial calculator agent (expressions, percentages, compound interest).
- `universe_manager.py` - universe knowledgebase agent with local tools for reading files and searching project content.

## Demo content

- `universe_demo.md` - sample worldbuilding file used to demo `universe_manager.py`.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install openai groq
```

### Run each agent

```bash
python agent.py
python calculator_agent.py
python universe_manager.py --read universe_demo.md
python universe_manager.py --query "Summarize the demo universe."
```
