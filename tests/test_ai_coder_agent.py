#!/usr/bin/env python
"""
Unit tests for the AI Developer Agent.
"""

import pytest
import os
from unittest.mock import Mock, patch

from ai_coder_agent import (
    Plan,
    Analysis,
    GeneratePlan,
    GenerateCode,
    AnalyzeResult,
    execute_code,
    extract_code,
    AIDevrAgent,
)
import dspy


class TestCodeExecution:
    """Test code execution functionality."""

    def test_execute_simple_code(self):
        """Test executing simple Python code."""
        code = "print('Hello, World!')"
        result = execute_code(code)
        assert "Hello, World!" in result
        assert "STDOUT:" in result

    def test_execute_code_with_error(self):
        """Test executing code that produces an error."""
        code = "print(undefined_variable)"
        result = execute_code(code)
        assert "STDERR:" in result
        assert "NameError" in result

    def test_execute_code_no_output(self):
        """Test executing code with no output."""
        code = "x = 1 + 1"
        result = execute_code(code)
        assert "No output captured" in result

    def test_execute_code_cleanup(self):
        """Test that temporary files are cleaned up."""
        code = "print('test')"
        filename = "test_temp_file.py"
        execute_code(code, filename)
        assert not os.path.exists(filename)

    def test_extract_code_with_markdown(self):
        """Test extracting code from markdown-formatted string."""
        llm_output = (
            "Here's the code:\n```python\nprint('Hello')\nx = 1\n```\nThat's it!"
        )
        extracted = extract_code(llm_output)
        assert extracted == "print('Hello')\nx = 1"

    def test_extract_code_without_markdown(self):
        """Test extracting code from plain string."""
        code = "print('Hello')\nx = 1"
        extracted = extract_code(code)
        assert extracted == code

    def test_extract_code_empty_markdown(self):
        """Test extracting code from empty markdown block."""
        llm_output = "```python\n\n```"
        extracted = extract_code(llm_output)
        assert extracted == ""


class TestDSPySignatures:
    """Test DSPy signature definitions."""

    def test_generate_plan_signature(self):
        """Test GeneratePlan signature structure."""
        sig = GeneratePlan()
        assert hasattr(sig, "goal")
        assert hasattr(sig, "plan")

    def test_generate_code_signature(self):
        """Test GenerateCode signature structure."""
        sig = GenerateCode()
        assert hasattr(sig, "task_description")
        assert hasattr(sig, "context")
        assert hasattr(sig, "code")

    def test_analyze_result_signature(self):
        """Test AnalyzeResult signature structure."""
        sig = AnalyzeResult()
        assert hasattr(sig, "task_description")
        assert hasattr(sig, "code")
        assert hasattr(sig, "execution_log")
        assert hasattr(sig, "analysis")


class TestAIDevrAgent:
    """Test the main AIDevrAgent class."""

    @pytest.fixture
    def mock_lm(self):
        """Create a mock language model."""
        return Mock(spec=dspy.LM)

    @pytest.fixture
    def agent(self, mock_lm):
        """Create an AIDevrAgent instance with mock LM."""
        return AIDevrAgent(mock_lm)

    def test_agent_initialization(self, mock_lm):
        """Test agent initialization."""
        agent = AIDevrAgent(mock_lm)
        assert agent.planner is not None
        assert agent.coder is not None
        assert agent.analyzer is not None

    @patch("ai_coder_agent.execute_code")
    @patch("builtins.print")
    def test_agent_run_success_flow(self, mock_print, mock_execute, agent):
        """Test successful agent execution flow."""
        # Mock the DSPy predictors
        mock_plan = Plan(description="Test plan", tasks=["Task 1"])
        mock_plan_result = Mock()
        mock_plan_result.plan = mock_plan
        agent.planner = Mock(return_value=mock_plan_result)

        mock_code_result = Mock()
        mock_code_result.code = "```python\nprint('test')\n```"
        agent.coder = Mock(return_value=mock_code_result)

        mock_analysis = Analysis(description="Success", completed=True)
        mock_analysis_result = Mock()
        mock_analysis_result.analysis = mock_analysis
        agent.analyzer = Mock(return_value=mock_analysis_result)

        mock_execute.return_value = (
            "--- Execution Log ---\nSTDOUT:\ntest\n--------------------"
        )

        # Run the agent
        agent.run("Test goal")

        # Verify calls
        agent.planner.assert_called_once()
        agent.coder.assert_called_once()
        agent.analyzer.assert_called_once()
        mock_execute.assert_called_once()

    @patch("ai_coder_agent.execute_code")
    @patch("builtins.print")
    def test_agent_run_failure_flow(self, mock_print, mock_execute, agent):
        """Test agent execution flow with task failure."""
        # Mock the DSPy predictors
        mock_plan = Plan(description="Test plan", tasks=["Task 1", "Task 2"])
        mock_plan_result = Mock()
        mock_plan_result.plan = mock_plan
        agent.planner = Mock(return_value=mock_plan_result)

        mock_code_result = Mock()
        mock_code_result.code = "```python\nprint('test')\n```"
        agent.coder = Mock(return_value=mock_code_result)

        # First task fails
        mock_analysis = Analysis(description="Failed", completed=False)
        mock_analysis_result = Mock()
        mock_analysis_result.analysis = mock_analysis
        agent.analyzer = Mock(return_value=mock_analysis_result)

        mock_execute.return_value = (
            "--- Execution Log ---\nSTDERR:\nError\n--------------------"
        )

        # Run the agent
        agent.run("Test goal")

        # Verify it stops after first failure
        agent.planner.assert_called_once()
        agent.coder.assert_called_once()  # Only called once for first task
        agent.analyzer.assert_called_once()  # Only called once for first task


class TestIntegration:
    """Integration tests using dummy LM."""

    def test_dummy_lm_integration(self):
        """Test the agent with dummy LM to verify basic flow."""
        lm = dspy.LM(model="dummy-model")
        agent = AIDevrAgent(lm)

        # This should run without errors (though with dummy output)
        with patch("builtins.print"):  # Suppress output for test
            try:
                agent.run("Create a simple hello world program")
                # If we get here, the basic flow works
                assert True
            except Exception as e:
                pytest.fail(f"Integration test failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
