# AI Coder Agent

An intelligent Python agent built with DSPy framework that takes high-level coding goals and iteratively generates, executes, and refines code to achieve them.

## Features

- **Plan Generation**: Breaks down high-level goals into concrete, sequential tasks
- **Code Generation**: Creates Python code for specific tasks with context awareness
- **Safe Execution**: Runs generated code in isolated subprocess with timeout protection
- **Result Analysis**: Evaluates execution results and determines task completion
- **Iterative Workflow**: Continues through tasks until completion or failure

## Architecture

The agent uses three core DSPy reasoning modules:

1. **`GeneratePlan`**: Decomposes goals into structured plans with tasks
2. **`GenerateCode`**: Writes Python code for individual tasks
3. **`AnalyzeResult`**: Evaluates execution results and completion status

## Files

- `ai_coder_agent.py` - Main agent implementation with DSPy integration
- `demo_simple_workflow.py` - Working demo without DSPy dependencies
- `test_core_functions.py` - Tests for core functionality
- `test_ai_coder_agent.py` - Full unit tests (requires DSPy)
- `test_integration.py` - Integration tests
- `requirements.txt` - Python dependencies

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

```bash
# With OpenAI API key
export OPENAI_API_KEY='your-api-key'
python ai_coder_agent.py "Create a Python file that calculates fibonacci numbers"

# Using dummy LLM (for testing)
python ai_coder_agent.py "Create a simple hello world program"

# With specific model
python ai_coder_agent.py "Write a web scraper" --model gpt-4o --api-key sk-xxx
```

### Examples

**Simple file creation:**
```bash
python ai_coder_agent.py "Create a Python file named 'greetings.py' that prints 'Hello, Agent!'"
```

**API interaction:**
```bash
python ai_coder_agent.py "Write a script that fetches current time from worldtimeapi.org and saves it to 'time.txt'"
```

**Data processing:**
```bash
python ai_coder_agent.py "Create a script that reads a CSV file and calculates statistics"
```

## Demo

Run the simplified demo to see the agent workflow:

```bash
python demo_simple_workflow.py
```

This shows how the agent works conceptually with mock LLM responses.

## Testing

Run the core function tests:
```bash
python test_core_functions.py
```

Run full test suite (if DSPy is working):
```bash
python -m pytest test_ai_coder_agent.py -v
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

- Current DSPy version has dependency conflicts with aiohttp
- The main agent requires valid OpenAI API key or uses dummy responses
- Generated code runs with full system permissions (use with caution)

## Contributing

1. Test core functions: `python test_core_functions.py`
2. Run demo: `python demo_simple_workflow.py`
3. Ensure all tests pass before submitting changes

## License

MIT License - Feel free to use and modify.
