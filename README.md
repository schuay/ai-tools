# ai-exp-langgraph

A multi-model AI assistant and developer toolkit built on LangGraph.

## Tools

- **`agent <prompt>`** — interactive Textual TUI; multi-turn conversation with a choice of models
- **`qq <question>`** — quick one-shot query from the terminal, optionally with stdin context
- **`commitmsg`** — generate a commit message for the current `git diff HEAD`

## Installation

```sh
uv tool install .  # locally
uv tool install https://github.com/schuay/ai-tools.git  # from github
```

This installs all three commands into your PATH via `~/.local/bin/`.

To update after pulling changes:

```sh
uv tool upgrade ai-exp-langgraph
```

## Environment variables

The tools require API keys. Set them in your shell profile (`~/.zshenv`, `~/.bashrc`, etc.):

```sh
# Required: Gemini models (default for all tools)
export GOOGLE_API_KEY=...

# Required: web search (used by `agent` and `qq`)
export TAVILY_API_KEY=...

# Optional: OpenAI models (selectable in `agent`)
export OPENAI_API_KEY=...

# Optional: DeepSeek models (selectable in `agent`)
export DEEPSEEK_API_KEY=...
```

Reload your shell or run `source ~/.zshenv` after editing.

## Usage

```sh
# Interactive agent session
agent "explain what changed in this PR"

# Quick question
qq what is the difference between a hard link and a symlink

# Pipe output into a question
git log --oneline -20 | qq which of these commits are likely to affect performance

# Generate a commit message for staged changes
git commit -m "$(commitmsg)"
```
