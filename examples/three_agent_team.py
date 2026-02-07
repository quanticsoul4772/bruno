#!/usr/bin/env python3
"""Three-agent development team using abliterated Bruno models via CrewAI.

This script demonstrates a complete development team:
- Frontend Agent: React/UI development
- Backend Agent: API/database development
- Test Agent: QA and test generation
"""

import sys
import traceback

from crewai import LLM, Agent, Crew, Process, Task


def create_development_team():
    """Create a 3-agent development team."""
    print("Initializing 3-agent development team...")

    # Create LLMs for each agent (using Ollama backend)
    frontend_llm = LLM(
        model="ollama/frontend-agent",
        base_url="http://localhost:11434",
        timeout=600,
        max_retries=3,
    )

    backend_llm = LLM(
        model="ollama/backend-agent",
        base_url="http://localhost:11434",
        timeout=600,
        max_retries=3,
    )

    test_llm = LLM(
        model="ollama/test-agent",
        base_url="http://localhost:11434",
        timeout=600,
        max_retries=3,
    )

    # Create specialized agents
    frontend_dev = Agent(
        role="Frontend Developer",
        goal="Build responsive, user-friendly React components",
        backstory="Expert in React, TypeScript, Tailwind CSS. Writes clean, concise code without over-engineering.",
        llm=frontend_llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3,
        max_retry_limit=3,
    )

    backend_dev = Agent(
        role="Backend Developer",
        goal="Create scalable FastAPI endpoints and database schemas",
        backstory="Expert in FastAPI, PostgreSQL, async patterns. Focuses on clean architecture without premature optimization.",
        llm=backend_llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3,
        max_retry_limit=3,
    )

    qa_engineer = Agent(
        role="QA Engineer",
        goal="Write comprehensive test suites with high coverage",
        backstory="Expert in pytest, coverage analysis, edge cases. Proactively writes tests for all code.",
        llm=test_llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3,
        max_retry_limit=3,
    )

    return frontend_dev, backend_dev, qa_engineer


def run_example_task():
    """Run an example development task with the 3-agent team."""
    frontend_dev, backend_dev, qa_engineer = create_development_team()

    # Define tasks
    task1 = Task(
        description=(
            "Design a REST API for user authentication. "
            "Include endpoints for: register, login, logout, refresh token. "
            "Define request/response schemas with FastAPI. "
            "Use JWT tokens for authentication. "
            "Return ONLY the code, no explanations."
        ),
        agent=backend_dev,
        expected_output="Complete FastAPI router code with endpoints and schemas",
    )

    task2 = Task(
        description=(
            "Create a login form React component. "
            "Include email and password fields with validation. "
            "Add a submit button and error display. "
            "Use Tailwind CSS for styling. "
            "Return ONLY the code, no explanations."
        ),
        agent=frontend_dev,
        expected_output="Complete React component with TypeScript types",
    )

    task3 = Task(
        description=(
            "Write pytest tests for the authentication API. "
            "Test all endpoints: register, login, logout, refresh. "
            "Include edge cases: invalid credentials, expired tokens, duplicate registration. "
            "Return ONLY the code, no explanations."
        ),
        agent=qa_engineer,
        expected_output="Complete pytest test suite with fixtures and assertions",
    )

    # Create crew with sequential process
    crew = Crew(
        agents=[backend_dev, frontend_dev, qa_engineer],
        tasks=[task1, task2, task3],
        process=Process.sequential,
        verbose=True,
    )

    # Execute
    print("\n" + "=" * 80)
    print("EXECUTING DEVELOPMENT TASK")
    print("=" * 80 + "\n")

    try:
        result = crew.kickoff()
        print("\n" + "=" * 80)
        print("FINAL RESULT")
        print("=" * 80)
        print(result)
        print("=" * 80)
        return result
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"CREW FAILED: {type(e).__name__}: {e}")
        print("=" * 80)
        traceback.print_exc()
        return None


if __name__ == "__main__":
    try:
        result = run_example_task()
        if result:
            print("\nSUCCESS - All 3 agents completed their tasks!")
        else:
            print("\nFAILED - Check errors above")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)
