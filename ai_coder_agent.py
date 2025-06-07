#!/usr/bin/env python
import dspy
import os
import subprocess
import sys
import re
import click
from pydantic import BaseModel, Field
from typing import List

# --- 1. Pydantic Models for Structured Output ---

class Plan(BaseModel):
    """A structured plan to achieve a coding goal."""
    description: str = Field(description="A high-level summary of the plan.")
    tasks: List[str] = Field(description="A list of concrete, sequential coding tasks to achieve the goal.")

class Analysis(BaseModel):
    """An analysis of a code execution attempt."""
    description: str = Field(description="A detailed analysis of the execution result. Explain what worked, what didn't, and why.")
    completed: bool = Field(description="Set to true if the task was successfully completed, false otherwise.")


# --- 2. DSPy Signatures for Reasoning Modules ---

class GeneratePlan(dspy.Signature):
    """Decompose the high-level goal into a sequence of concrete coding tasks."""
    goal: str = dspy.InputField(desc="The overall objective for the AI Developer.")
    plan: Plan = dspy.OutputField(cls=Plan)

class GenerateCode(dspy.Signature):
    """Write Python code to accomplish a single task based on the plan and context."""
    task_description: str = dspy.InputField(desc="The specific task to be implemented in code.")
    context: str = dspy.InputField(desc="Relevant context, such as previous code, file listings, or error messages from prior steps.")
    code: str = dspy.OutputField(prefix="```python\n", suffix="\n```")

class AnalyzeResult(dspy.Signature):
    """Evaluate if the executed code successfully completed its task and achieved the intended outcome."""
    task_description: str = dspy.InputField(desc="The original description of the task.")
    code: str = dspy.InputField(desc="The Python code that was executed.")
    execution_log: str = dspy.InputField(desc="The captured stdout and stderr from the code execution.")
    analysis: Analysis = dspy.OutputField(cls=Analysis)


# --- 3. Safe Code Execution ---

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

class AIDevrAgent:
    """Orchestrator class"""

    def __init__(self, lm: dspy.LM):
        """Initializes the agent with its reasoning modules."""
        dspy.settings.configure(lm=lm)
        
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


# --- 5. CLI Interface with Click ---
@click.command()
@click.argument('goal', type=str)
@click.option('--api-key', 
              default=lambda: os.getenv("OPENAI_API_KEY"), 
              help='OpenAI API key (defaults to OPENAI_API_KEY env var)')
@click.option('--model', 
              default="gpt-3.5-turbo", 
              help='Language model to use (e.g., gpt-4o, gpt-3.5-turbo)')
@click.option('--max-tokens', 
              default=2000, 
              type=int,
              help='Maximum tokens for LLM responses')
@click.option('--timeout', 
              default=60, 
              type=int,
              help='Code execution timeout in seconds')
@click.option('--verbose', 
              is_flag=True, 
              help='Enable verbose output')
def main(goal: str, api_key: str, model: str, max_tokens: int, timeout: int, verbose: bool):
    """
    AI Developer Agent - Generate, execute, and refine code to achieve coding goals.
    
    GOAL: High-level description of what you want the agent to accomplish.
    
    Examples:
    
        aidevr "Create a hello world program"
        
        aidevr "Write a script to fetch weather data and save to CSV" --model gpt-4o
        
        aidevr "Build a simple calculator with error handling" --verbose
    """
    # Configure output verbosity
    if verbose:
        click.echo(f"🔧 Configuration:")
        click.echo(f"   Model: {model}")
        click.echo(f"   Max tokens: {max_tokens}")
        click.echo(f"   Timeout: {timeout}s")
        click.echo(f"   API key: {'✓ Set' if api_key else '✗ Not set'}")
        click.echo()

    # Configure LLM
    if not api_key:
        click.echo("⚠️  " + click.style("Warning:", fg="yellow", bold=True) + 
                   " OPENAI_API_KEY not found.")
        click.echo("Using a dummy LLM. The agent will produce placeholder text.")
        lm = dspy.LM(model="dummy-model")
    else:
        if verbose:
            click.echo("🔑 " + click.style("API key found.", fg="green", bold=True) + 
                       f" Configuring LLM with model: {model}")
        lm = dspy.LM(model=model, api_key=api_key, max_tokens=max_tokens)

    # Create and run agent
    agent = AIDevrAgent(lm)
    
    try:
        agent.run(goal)
    except KeyboardInterrupt:
        click.echo("\n🛑 " + click.style("Agent interrupted by user.", fg="red"))
        sys.exit(1)
    except Exception as e:
        click.echo(f"\n❌ " + click.style("Agent failed:", fg="red") + f" {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


# --- Additional CLI Commands ---
@click.group()
def cli():
    """AI Developer Agent CLI"""
    pass


@cli.command()
def version():
    """Show version information."""
    click.echo("AI Developer Agent v1.0.0")
    click.echo("Built with DSPy framework")


@cli.command()
@click.option('--format', 'output_format', 
              type=click.Choice(['text', 'json']), 
              default='text',
              help='Output format')
def config(output_format):
    """Show current configuration."""
    config_data = {
        "api_key_set": bool(os.getenv("OPENAI_API_KEY")),
        "default_model": "gpt-3.5-turbo",
        "default_timeout": 60,
        "default_max_tokens": 2000
    }
    
    if output_format == 'json':
        import json
        click.echo(json.dumps(config_data, indent=2))
    else:
        click.echo("🔧 AI Developer Agent Configuration:")
        click.echo(f"   API Key: {'✓ Set' if config_data['api_key_set'] else '✗ Not set'}")
        click.echo(f"   Default Model: {config_data['default_model']}")
        click.echo(f"   Default Timeout: {config_data['default_timeout']}s")
        click.echo(f"   Default Max Tokens: {config_data['default_max_tokens']}")


if __name__ == "__main__":
    main()
