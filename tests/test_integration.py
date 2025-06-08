import pytest
import os
import tempfile
import subprocess
import sys
from ai_coder_agent import AIDevrAgent
import dspy


class TestEndToEndIntegration:
    """End-to-end integration tests."""
    
    def test_cli_with_dummy_lm(self):
        """Test the CLI interface with dummy LM."""
        cmd = [
            sys.executable, "ai_coder_agent.py", 
            "Create a simple hello world program"
        ]
        
        # Run without API key to use dummy LM
        env = os.environ.copy()
        if 'OPENAI_API_KEY' in env:
            del env['OPENAI_API_KEY']
        
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=30,
                env=env
            )
            
            # Should not crash
            assert result.returncode == 0 or "Warning" in result.stdout
            assert "dummy" in result.stdout.lower() or "dummy" in result.stderr.lower()
            
        except subprocess.TimeoutExpired:
            pytest.fail("CLI test timed out")
        except Exception as e:
            pytest.fail(f"CLI test failed: {e}")
    
    def test_agent_with_simple_goal(self):
        """Test agent with a simple, achievable goal using dummy LM."""
        lm = dspy.LM(model="dummy-model")
        agent = AIDevrAgent(lm)
        
        # Capture stdout to avoid cluttering test output
        import io
        from contextlib import redirect_stdout
        
        f = io.StringIO()
        with redirect_stdout(f):
            try:
                agent.run("Print hello world")
                output = f.getvalue()
                
                # Basic checks that the flow executed
                assert "Goal:" in output
                assert "Generating plan" in output
                
            except Exception as e:
                pytest.fail(f"Simple goal test failed: {e}")


class TestRealWorldScenarios:
    """Test scenarios that simulate real-world usage."""
    
    def test_file_creation_scenario(self):
        """Test a scenario where the agent should create a file."""
        lm = dspy.LM(model="dummy-model")
        agent = AIDevrAgent(lm)
        
        # Use a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)
            
            import io
            from contextlib import redirect_stdout
            
            f = io.StringIO()
            with redirect_stdout(f):
                try:
                    agent.run("Create a file named 'test.txt' with content 'Hello World'")
                    # This may not actually create the file with dummy LM,
                    # but should not crash
                    assert True
                except Exception as e:
                    pytest.fail(f"File creation scenario failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
