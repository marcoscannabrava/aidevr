# Initial Prompt from Gemini 2.5
You are an expert Python developer specializing in building agentic systems with the DSPy framework.

Your task is to write a complete, self-contained Python script that implements an 'AI Developer' agent. The agent will take a high-level coding goal and iteratively generate, execute, and refine code to achieve it.

**Agent Logic:**

The agent operates in a loop, orchestrated by a main Python function. You must implement the following reasoning modules as DSPy Signatures:

1.  **`GeneratePlan(goal) -> Plan`**:
    *   **Input**: `goal` (a string describing the overall objective).
    *   **Output**: `Plan` (a Pydantic model with a `description` and a `tasks` list).
    *   **Purpose**: Decompose the high-level goal into a sequence of concrete coding tasks.

2.  **`GenerateCode(task_description, context) -> Code`**:
    *   **Input**: `task_description` (a string for one task from the plan) and `context` (a string containing relevant information like previous code or logs).
    *   **Output**: `code` (a string of Python code).
    *   **Purpose**: Write the code to accomplish a single task.

3.  **`AnalyzeResult(task_description, code, execution_log) -> Analysis`**:
    *   **Input**: The `task_description`, the generated `code`, and the `execution_log` (stdout/stderr).
    *   **Output**: `Analysis` (a Pydantic model with a `description` of what happened and a `completed` boolean flag).
    *   **Purpose**: Evaluate if the code ran successfully and achieved the task's intent.

**Implementation Requirements:**

*   **Framework**: Use `dspy` for all LLM interactions. Define the Signatures and use `dspy.Predict` or `dspy.ChainOfThought`.
*   **Structured Output**: Use Pydantic `BaseModel`s for `Plan` and `Analysis` and integrate them with DSPy Signatures using `dspy.Output(cls=...)`.
*   **Orchestration**: Write a main execution loop in Python that:
    1.  Calls `GeneratePlan` to create the initial work breakdown.
    2.  Iterates through the tasks. For each task:
        a. Calls `GenerateCode`.
        b. **Executes the generated code** safely (e.g., using `subprocess` to run it as a separate file and capture output).
        c. Calls `AnalyzeResult` with the execution log.
        d. If `analysis.completed` is false, stop or log the failure. If true, proceed.
*   **Configuration**: Configure a placeholder LLM (e.g., `dspy.Dummy` or `dspy.OpenAI` with placeholder keys).
*   **Code Quality**: Ensure the final script is modular, well-commented, and includes all necessary imports.


# Initial code proposed by Gemini

Of course. As a specialist in agentic systems with DSPy, I will construct a complete, self-contained Python script for an 'AI Developer' agent.

This script will include:
1.  **Pydantic Models** for structured data (`Plan`, `Analysis`).
2.  **DSPy Signatures** for the core reasoning modules (`GeneratePlan`, `GenerateCode`, `AnalyzeResult`).
3.  A safe **code execution environment** using `subprocess`.
4.  An **orchestration class** (`AIDevrAgent`) that manages the plan-code-execute-analyze loop.
5.  A **command-line interface** using `argparse` for easy use.

Here is the complete and self-contained script:

```python
#!/usr/bin/env python
# Filename: ai_coder_agent.py

import dspy
import os
import subprocess
import sys
import re
import argparse
from pydantic import BaseModel, Field
from typing import List, Optional

# --- 1. Pydantic Models for Structured Output ---
# These models define the expected structure for the outputs of our LLM calls,
# ensuring predictable and parsable results.

class Plan(BaseModel):
    """A structured plan to achieve a coding goal."""
    description: str = Field(description="A high-level summary of the plan.")
    tasks: List[str] = Field(description="A list of concrete, sequential coding tasks to achieve the goal.")

class Analysis(BaseModel):
    """An analysis of a code execution attempt."""
    description: str = Field(description="A detailed analysis of the execution result. Explain what worked, what didn't, and why.")
    completed: bool = Field(description="Set to true if the task was successfully completed, false otherwise.")


# --- 2. DSPy Signatures for Reasoning Modules ---
# These signatures define the inputs and outputs for each reasoning step of our agent.
# DSPy uses these to prompt the underlying LLM.

class GeneratePlan(dspy.Signature):
    """Decompose the high-level goal into a sequence of concrete coding tasks."""
    goal: dspy.InputField(desc="The overall objective for the AI Developer.")
    plan: dspy.OutputField(cls=Plan)

class GenerateCode(dspy.Signature):
    """Write Python code to accomplish a single task based on the plan and context."""
    task_description: dspy.InputField(desc="The specific task to be implemented in code.")
    context: dspy.InputField(desc="Relevant context, such as previous code, file listings, or error messages from prior steps.")
    code: dspy.OutputField(prefix="```python\n", suffix="\n```")

class AnalyzeResult(dspy.Signature):
    """Evaluate if the executed code successfully completed its task and achieved the intended outcome."""
    task_description: dspy.InputField(desc="The original description of the task.")
    code: dspy.InputField(desc="The Python code that was executed.")
    execution_log: dspy.InputField(desc="The captured stdout and stderr from the code execution.")
    analysis: dspy.OutputField(cls=Analysis)


# --- 3. Safe Code Execution ---
# This function handles the execution of LLM-generated code in a sandboxed manner.

def execute_code(code: str, filename: str = "temp_agent_code.py") -> str:
    """
    Executes a string of Python code safely and captures its output.

    Args:
        code: The Python code to execute.
        filename: The temporary file to save the code in.

    Returns:
        A string containing the captured stdout and stderr.
    """
    # Clean up any previous temp file
    if os.path.exists(filename):
        os.remove(filename)

    try:
        # Write the code to a temporary file
        with open(filename, "w") as f:
            f.write(code)

        # Execute the file using a subprocess
        # We use sys.executable to ensure the same Python interpreter is used
        result = subprocess.run(
            [sys.executable, filename],
            capture_output=True,
            text=True,
            timeout=60  # Safety timeout to prevent infinite loops
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


# --- 4. The AI Developer Agent ---
# This class orchestrates the entire process, from planning to execution.

class AIDevrAgent:
    def __init__(self, lm: dspy.LM):
        """Initializes the agent with its reasoning modules."""
        dspy.settings.configure(lm=lm)
        
        # Use dspy.Predict for simpler, direct prompting
        self.planner = dspy.Predict(GeneratePlan)
        self.coder = dspy.Predict(GenerateCode)
        self.analyzer = dspy.Predict(AnalyzeResult)

    def run(self, goal: str):
        """
        Runs the agent to achieve a given coding goal.
        
        Args:
            goal: The high-level objective.
        """
        print(f"🚀 \033[1;35mGoal:\033[0m {goal}\n")

        # 1. Generate Plan
        print("🤔 \033[1;33mGenerating plan...\033[0m")
        plan_result = self.planner(goal=goal)
        plan = plan_result.plan
        print(f"📝 \033[1;36mPlan:\033[0m {plan.description}")
        for i, task in enumerate(plan.tasks):
            print(f"  {i+1}. {task}")
        print("-" * 30)

        context = f"The overall goal is: {goal}\n"
        context += f"The plan is: {plan.description}\n"

        # 2. Iterate through tasks
        for i, task in enumerate(plan.tasks):
            print(f"\n▶️  \033[1;34mExecuting Task {i+1}/{len(plan.tasks)}:\033[0m {task}")
            
            # 2a. Generate Code
            print("   - 🤖 Generating code...")
            code_result = self.coder(task_description=task, context=context)
            generated_code = extract_code(code_result.code)
            
            print("\033[32m--- Generated Code ---\033[0m")
            print(generated_code)
            print("\033[32m----------------------\033[0m")

            # 2b. Execute Code
            print("   - ⚡ Executing code...")
            execution_log = execute_code(generated_code)
            print(execution_log)

            # 2c. Analyze Result
            print("   - 🧐 Analyzing result...")
            analysis_result = self.analyzer(
                task_description=task,
                code=generated_code,
                execution_log=execution_log
            )
            analysis = analysis_result.analysis
            
            print(f"   - \033[1;37mAnalysis:\033[0m {analysis.description}")

            # 2d. Check for completion and update context
            if analysis.completed:
                print("   - ✅ \033[1;32mTask Completed Successfully!\033[0m")
                context += f"\n--- Completed Task: {task} ---\n"
                context += f"Code:\n{generated_code}\n"
                context += f"Result: {analysis.description}\n"
            else:
                print("   - ❌ \033[1;31mTask Failed. Stopping agent.\033[0m")
                print("-" * 30)
                print("Agent stopped due to task failure.")
                return
            
            print("-" * 30)
        
        print("\n🎉 \033[1;32mAll tasks completed. Agent finished successfully!\033[0m")


# --- 5. Main Execution Block & CLI ---
def main():
    """Main function to parse arguments and run the agent."""
    parser = argparse.ArgumentParser(description="An AI Developer agent powered by DSPy.")
    parser.add_argument("goal", type=str, help="The high-level coding goal for the agent.")
    parser.add_argument("--api-key", type=str, default=os.getenv("OPENAI_API_KEY"), help="OpenAI API key.")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="The language model to use (e.g., 'gpt-4o', 'gpt-3.5-turbo').")

    args = parser.parse_args()

    if not args.api_key:
        print("⚠️ \033[1;33mWarning:\033[0m OPENAI_API_KEY not found in environment or command-line arguments.")
        print("Using a dummy LLM. The agent will produce placeholder text.")
        lm = dspy.Dummy(model="dummy-model")
    else:
        print(f"🔑 \033[1;32mAPI key found. Configuring LLM with model:\033[0m {args.model}")
        # Configure the DSPy language model
        lm = dspy.OpenAI(model=args.model, api_key=args.api_key, max_tokens=2000)

    agent = AIDevrAgent(lm)
    agent.run(args.goal)

if __name__ == "__main__":
    main()
```

### How to Run the Script

1.  **Save the Code**: Save the code above into a file named `ai_coder_agent.py`.

2.  **Install Dependencies**: Make sure you have the required libraries installed.
    ```bash
    pip install dspy-ai pydantic openai
    ```

3.  **Set Up API Key (Recommended)**: For the agent to function with a real LLM, set your OpenAI API key as an environment variable.
    ```bash
    # On Linux/macOS
    export OPENAI_API_KEY='your-api-key-here'

    # On Windows (Command Prompt)
    set OPENAI_API_KEY=your-api-key-here
    ```
    Alternatively, you can pass it as a command-line argument.

4.  **Execute the Agent**: Run the script from your terminal with a coding goal.

    **Example 1: Simple file creation**
    ```bash
    python ai_coder_agent.py "Create a Python file named 'greetings.py' that prints 'Hello, Agent!' to the console."
    ```

    **Example 2: A more complex task**
    ```bash
    python ai_coder_agent.py "Write a Python script that fetches the current time from the worldtimeapi.org for 'America/New_York' and saves it to a file named 'ny_time.txt'."
    ```
    *(Note: This requires the `requests` library. The agent should correctly generate code that uses it, but the environment where you run `ai_coder_agent.py` must have `requests` installed for the execution step to succeed: `pip install requests`)*

    **Example 3: Using a different model and passing the API key directly**
    ```bash
    python ai_coder_agent.py "List all files in the current directory and save the list to 'file_list.txt'." --model gpt-4o --api-key sk-xxxxxxxx
    ```

    **Example 4: Running with the Dummy LLM (no API key needed)**
    If you don't provide an API key, the script will default to `dspy.Dummy`, which is useful for testing the agent's logic flow without incurring costs.
    ```bash
    python ai_coder_agent.py "This is a test goal for the dummy LLM."
    ```

