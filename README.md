# AI Coder Agent

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
uv install -e .

# install with development dependencies:
# uv pip install -e ".[dev]"

# or using pip:
# pip install -e .
```

## Usage

### Command Line Interface

The agent is available as `aicoder` command after installation:

```bash
# Basic usage with environment API key
export OPENAI_API_KEY='your-api-key'
aicoder "Create a Python file that calculates fibonacci numbers"

# Using dummy LLM (for testing)
aicoder "Create a simple hello world program"

# With specific model and verbose output
aicoder "Write a web scraper" --model gpt-4o --verbose

# With custom timeout and token limits
aicoder "Build a data analysis script" --timeout 120 --max-tokens 3000
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
aicoder version

# Show current configuration
aicoder config

# Show configuration in JSON format
aicoder config --format json
```

### Examples

**Simple file creation:**
```bash
aicoder "Create a Python file named 'greetings.py' that prints 'Hello, Agent!'"
```

**API interaction:**
```bash
aicoder "Write a script that fetches current time from worldtimeapi.org and saves it to 'time.txt'" --verbose
```

**Data processing:**
```bash
aicoder "Create a script that reads a CSV file and calculates statistics" --model gpt-4o
```

**Complex task with custom settings:**
```bash
aicoder "Build a web scraper with error handling and rate limiting" \
  --model gpt-4o \
  --max-tokens 4000 \
  --timeout 180 \
  --verbose
```

## Demo

Run the simplified demo to see the agent workflow:

```bash
python demo_simple_workflow.py
```

This shows how the agent works conceptually with mock LLM responses.

## Development

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-coder-agent.git
cd ai-coder-agent

# Install with development dependencies using uv
uv pip install -e ".[dev]"

# Or using pip
pip install -e ".[dev]"
```

### Testing

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_core_functions.py

# Run with coverage
uv run pytest --cov=. --cov-report=html

# Run core function tests directly
python tests/test_core_functions.py
```

### Code Quality

```bash
# Format code
uv run black .

# Sort imports
uv run isort .

# Lint code
uv run flake8

# Type checking
uv run mypy ai_coder_agent.py
```

## Technical Details

### Safe Code Execution

- Code runs in separate subprocess with 60-second timeout
- Captures both stdout and stderr
- Automatic cleanup of temporary files
- No access to parent process environment

### Structured Output

Uses Pydantic models for predictable LLM responses:

```python
class Plan(BaseModel):
    description: str
    tasks: List[str]

class Analysis(BaseModel):
    description: str
    completed: bool
```

### Error Handling

- Graceful handling of code execution errors
- Timeout protection for infinite loops
- Validation of LLM outputs
- Fallback mechanisms for malformed responses

## Known Issues

- Current DSPy version has dependency conflicts with aiohttp in some environments
- The main agent requires valid OpenAI API key or uses dummy responses
- Generated code runs with full system permissions (use with caution)
- CLI entry point may not work due to DSPy import issues (use `python ai_coder_agent.py` as fallback)

## Working Components

✅ **Core Functions**: All core functionality tested and working  
✅ **Demo Workflow**: Complete agent workflow demonstration available  
✅ **Test Suite**: Comprehensive tests for all components  
✅ **Modern Tooling**: UV package manager and Click CLI interface  
✅ **Safe Execution**: Isolated subprocess execution with timeouts

## Contributing

1. Test core functions: `python test_core_functions.py`
2. Run demo: `python demo_simple_workflow.py`
3. Ensure all tests pass before submitting changes

## License

MIT License - Feel free to use and modify.
