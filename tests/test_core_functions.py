#!/usr/bin/env python
"""
Test core functions extracted from the AI Coder Agent.
"""

import os
import subprocess
import sys
import re
from pydantic import BaseModel, Field
from typing import List


# --- Extracted Core Functions ---

def execute_code(code: str, filename: str = "temp_agent_code.py") -> str:
    """
    Executes a string of Python code safely and captures its output.
    """
    # Clean up any previous temp file
    if os.path.exists(filename):
        os.remove(filename)

    try:
        # Write the code to a temporary file
        with open(filename, "w") as f:
            f.write(code)

        # Execute the file using a subprocess
        result = subprocess.run(
            [sys.executable, filename],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        # Combine stdout and stderr into a single log
        log = f"--- Execution Log ---\n"
        if result.stdout:
            log += f"STDOUT:\n{result.stdout}\n"
        if result.stderr:
            log += f"STDERR:\n{result.stderr}\n"
        if not result.stdout and not result.stderr:
            log += "No output captured. The script ran silently.\n"
        log += "--------------------"
        return log

    except subprocess.TimeoutExpired:
        return "Execution timed out after 60 seconds."
    except Exception as e:
        return f"An error occurred during execution setup: {e}"
    finally:
        # Ensure the temporary file is always cleaned up
        if os.path.exists(filename):
            os.remove(filename)


def extract_code(llm_output: str) -> str:
    """Extracts the Python code from a markdown-formatted string."""
    match = re.search(r"```python\n(.*?)```", llm_output, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback if the ```python block is not found
    return llm_output.strip()


# --- Pydantic Models ---

class Plan(BaseModel):
    """A structured plan to achieve a coding goal."""
    description: str = Field(description="A high-level summary of the plan.")
    tasks: List[str] = Field(description="A list of concrete, sequential coding tasks to achieve the goal.")


class Analysis(BaseModel):
    """An analysis of a code execution attempt."""
    description: str = Field(description="A detailed analysis of the execution result.")
    completed: bool = Field(description="Set to true if the task was successfully completed.")


# --- Tests ---

def test_execute_simple_code():
    """Test executing simple Python code."""
    code = "print('Hello, World!')"
    result = execute_code(code)
    assert "Hello, World!" in result
    assert "STDOUT:" in result
    print("✅ Simple code execution test passed!")


def test_execute_code_with_error():
    """Test executing code that produces an error."""
    code = "print(undefined_variable)"
    result = execute_code(code)
    assert "STDERR:" in result
    assert "NameError" in result
    print("✅ Error handling test passed!")


def test_execute_code_no_output():
    """Test executing code with no output."""
    code = "x = 1 + 1"
    result = execute_code(code)
    assert "No output captured" in result
    print("✅ No output test passed!")


def test_extract_code_with_markdown():
    """Test extracting code from markdown-formatted string."""
    llm_output = "Here's the code:\n```python\nprint('Hello')\nx = 1\n```\nThat's it!"
    extracted = extract_code(llm_output)
    assert extracted == "print('Hello')\nx = 1"
    print("✅ Markdown extraction test passed!")


def test_extract_code_without_markdown():
    """Test extracting code from plain string."""
    code = "print('Hello')\nx = 1"
    extracted = extract_code(code)
    assert extracted == code
    print("✅ Plain text extraction test passed!")


def test_plan_creation():
    """Test Plan model creation and validation."""
    plan = Plan(
        description="Test plan",
        tasks=["Task 1", "Task 2", "Task 3"]
    )
    assert plan.description == "Test plan"
    assert len(plan.tasks) == 3
    assert plan.tasks[0] == "Task 1"
    print("✅ Plan model test passed!")


def test_analysis_creation():
    """Test Analysis model creation and validation."""
    analysis = Analysis(
        description="Code executed successfully",
        completed=True
    )
    assert analysis.description == "Code executed successfully"
    assert analysis.completed is True
    print("✅ Analysis model test passed!")


def test_file_creation_code():
    """Test code that creates a file."""
    code = """
with open('test_output.txt', 'w') as f:
    f.write('Hello from generated code!')
print('File created successfully')
"""
    result = execute_code(code)
    assert "File created successfully" in result
    assert "STDOUT:" in result
    
    # Check if file was created and clean up
    if os.path.exists('test_output.txt'):
        with open('test_output.txt', 'r') as f:
            content = f.read()
        assert content == 'Hello from generated code!'
        os.remove('test_output.txt')
    
    print("✅ File creation test passed!")


def run_all_tests():
    """Run all tests."""
    print("🧪 Running core function tests...\n")
    
    test_execute_simple_code()
    test_execute_code_with_error()
    test_execute_code_no_output()
    test_extract_code_with_markdown()
    test_extract_code_without_markdown()
    test_plan_creation()
    test_analysis_creation()
    test_file_creation_code()
    
    print("\n🎉 All core function tests passed!")


if __name__ == "__main__":
    run_all_tests()
