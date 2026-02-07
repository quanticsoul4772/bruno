#!/usr/bin/env python3
"""Test single agent with abliterated model via Ollama + CrewAI.

This script demonstrates using an abliterated Bruno model with CrewAI
for agentic coding tasks.
"""

from crewai import LLM, Agent, Crew, Task
from dotenv import load_dotenv

# Load Ollama configuration
load_dotenv(".env.crewai")


def test_single_agent():
    """Test single agent with abliterated model."""
    print("Initializing abliterated coding agent...")

    # Create LLM using CrewAI's LLM class with Ollama backend
    # Use openai/ prefix to force OpenAI-compatible API format
    bruno_llm = LLM(
        model="openai/bruno-coder-7b",
        base_url="http://localhost:11434/v1",
        api_key="not-needed",
    )

    # Create agent
    coder = Agent(
        role="Python Developer",
        goal="Write clean, functional Python code",
        backstory="Expert Python developer who writes concise, working code without over-engineering",
        llm=bruno_llm,
        verbose=True,
        allow_delegation=False,
    )

    # Create task
    task = Task(
        description=(
            "Write a Python function called 'is_prime' that checks if a number is prime. "
            "Include error handling for invalid inputs. "
            "Add a simple test to demonstrate it works."
        ),
        agent=coder,
        expected_output="Complete Python code with function definition and test",
    )

    # Create crew with single agent
    crew = Crew(agents=[coder], tasks=[task], verbose=True)

    # Execute
    print("\nExecuting task...\n")
    result = crew.kickoff()

    print("\n" + "=" * 80)
    print("RESULT:")
    print("=" * 80)
    print(result)
    print("=" * 80)

    return result


if __name__ == "__main__":
    test_single_agent()
