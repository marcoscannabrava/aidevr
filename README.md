# AI Developer Agent

An intelligent Python agent built with DSPy framework that takes high-level coding goals and iteratively generates, executes, and refines code to achieve them.

## Features

- **Plan Generation**: Breaks down high-level goals into concrete, sequential tasks
- **Code Generation**: Creates Python code for specific tasks with context awareness
- **Safe Execution**: Runs generated code in isolated subprocess with timeout protection
- **Result Analysis**: Evaluates execution results and determines task completion
- **Iterative Workflow**: Continues through tasks until completion or failure
- **Modern CLI**: Built with Click for intuitive command-line interaction
- **Type Safety**: Full Pydantic integration for structured LLM outputs

## Architecture

The agent uses three core DSPy reasoning modules:

1. **`GeneratePlan`**: Decomposes goals into structured plans with tasks
2. **`GenerateCode`**: Writes Python code for individual tasks
3. **`AnalyzeResult`**: Evaluates execution results and completion status

## Project Structure

```
ai-coder-agent/
├── ai_coder_agent.py          # Main agent implementation
├── demo_simple_workflow.py    # Working demo without DSPy dependencies
├── tests/                     # Test suite
│   ├── __init__.py
│   ├── test_core_functions.py # Core functionality tests
│   ├── test_ai_coder_agent.py # Full unit tests
│   └── test_integration.py    # Integration tests
├── pyproject.toml             # Project configuration and dependencies
├── .gitignore                 # Git ignore rules
└── README.md                  # This file
```

## Installation

```bash
pip install aidevr
```

## Usage

### Command Line Interface

The agent is available as `aidevr` command after installation:

```bash
# Basic usage with environment API key
export OPENAI_API_KEY='your-api-key'
aidevr "Create a Python file that calculates fibonacci numbers"

# Using dummy LLM (for testing)
aidevr "Create a simple hello world program"

# With specific model and verbose output
aidevr "Write a web scraper" --model gpt-4o --verbose

# With custom timeout and token limits
aidevr "Build a data analysis script" --timeout 120 --max-tokens 3000
```

### CLI Options

```
Options:
  --api-key TEXT       OpenAI API key (defaults to OPENAI_API_KEY env var)
  --model TEXT         Language model to use (default: gpt-3.5-turbo)
  --max-tokens INT     Maximum tokens for LLM responses (default: 2000)
  --timeout INT        Code execution timeout in seconds (default: 60)
  --verbose            Enable verbose output
  --help               Show help message
```

### Additional Commands

```bash
# Show version information
aidevr version

# Show current configuration
aidevr config

# Show configuration in JSON format
aidevr config --format json
```

### Examples

**Simple file creation:**
```bash
aidevr "Create a Python file named 'greetings.py' that prints 'Hello, Agent!'"
```

**API interaction:**
```bash
aidevr "Write a script that fetches current time from worldtimeapi.org and saves it to 'time.txt'" --verbose
```

**Data processing:**
```bash
aidevr "Create a script that reads a CSV file and calculates statistics" --model gpt-4o
```

**Complex task with custom settings:**
```bash
aidevr "Build a web scraper with error handling and rate limiting" \
  --model gpt-4o \
  --max-tokens 4000 \
  --timeout 180 \
  --verbose
```

### Testing

```bash
uv run pytest
```

## Known Issues

- Current DSPy version has dependency conflicts with aiohttp in some environments
- The main agent requires valid OpenAI API key or uses dummy responses
- Generated code runs with full system permissions (use with caution)
- CLI entry point may not work due to DSPy import issues (use `python ai_coder_agent.py` as fallback)

## License

GPL3 - use as you want, keep it open source
