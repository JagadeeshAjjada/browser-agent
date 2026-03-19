"""
Personal Assistant Agent — CLI
================================
Controls a real browser to execute natural language commands.
Asks for permission before any sensitive action (payments, forms, deletions, etc.)

Setup:
    pip install -r requirements.txt
    playwright install chromium
    # Add GEMINI_API_KEY (or GROQ_API_KEY) to .env
    # Set LLM_PROVIDER=gemini in .env

Usage:
    python execution/personal_assistant.py
"""

import asyncio
import os
import sys

from dotenv import load_dotenv

load_dotenv()

# Shared LLM factory (single source of truth)
from llm_utils import get_llm  # noqa: E402


# ── Permission System ────────────────────────────────────────────────────────

# Responses that mean "no" — kept in sync with web_ui.py
DENY_WORDS = {"no", "n", "deny", "cancel", "stop"}


def ask_permission(question: str) -> str:
    """
    Pauses execution and prompts the user for permission or information.
    Returns whatever the user types.
    """
    print("\n" + "=" * 55)
    print("  ⚠  AGENT NEEDS YOUR APPROVAL")
    print("=" * 55)
    print(f"\n  {question}\n")
    print("=" * 55)
    response = input("  Your response (yes / no / provide info): ").strip()
    print()
    return response


# ── Agent Runner ─────────────────────────────────────────────────────────────

async def run_agent(task: str):
    """
    Runs the browser agent for a given task.
    Registers a custom ask_human action so the LLM can request permission.
    """
    from browser_use import Agent, BrowserSession, Controller
    from browser_use.agent.views import ActionResult

    controller = Controller()

    @controller.action(
        "Ask the human user for permission before performing any sensitive action "
        "(purchases, payments, form submissions, deletions, sending messages, sign-ups). "
        "Also use this to ask for missing information such as login credentials, "
        "delivery addresses, or any personal details you need to complete the task."
    )
    def ask_human(question: str) -> ActionResult:
        response = ask_permission(question)
        if response.lower() in DENY_WORDS:
            return ActionResult(
                extracted_content="User denied this action. Do NOT proceed with it.",
                error="Permission denied by user.",
            )
        return ActionResult(extracted_content=f"User approved and said: {response}")

    # Visible browser so the user can watch everything happen in real time
    browser_session = BrowserSession(headless=False, keep_alive=False)

    llm = get_llm()

    agent = Agent(
        task=task,
        llm=llm,
        controller=controller,
        browser_session=browser_session,
        max_actions_per_step=5,
        max_failures=3,
        extend_system_message="""
You are a helpful personal assistant that can control the browser to complete tasks.

PERMISSION RULES — follow these strictly:
- Before clicking "Place Order", "Buy Now", "Confirm Purchase", "Pay", or anything that charges money → call ask_human FIRST
- Before submitting any form that contains personal info, financial details, or passwords → call ask_human FIRST
- Before deleting, removing, or overwriting any data → call ask_human FIRST
- Before sending any email, message, or public post → call ask_human FIRST
- Before signing up or creating an account anywhere → call ask_human FIRST
- If you need a password, credit card, OTP, or any sensitive info → call ask_human to request it
- If you are unsure whether an action is sensitive → call ask_human to be safe

Always be transparent. Narrate what you are doing in your thoughts.
If you encounter a CAPTCHA, call ask_human and tell the user to solve it manually.
""",
    )

    print("\n  Agent is working... (watch the browser window)\n")
    history = await agent.run(max_steps=50)
    return history


# ── Main CLI Loop ────────────────────────────────────────────────────────────

def print_banner():
    print("\n" + "=" * 55)
    print("   Personal Assistant Agent")
    print("=" * 55)
    provider = os.getenv("LLM_PROVIDER", "gemini").upper()
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash") if provider == "GEMINI" else "see .env"
    print(f"   LLM: {provider} ({model})  |  Browser: Chromium (visible)")
    print("=" * 55)
    print("   Type any command in plain English.")
    print("   Type 'exit' to quit.")
    print("=" * 55 + "\n")


def main():
    print_banner()

    # Validate LLM config on startup — fail fast with a clear message
    try:
        get_llm()
        print("  ✓ LLM config OK\n")
    except ValueError as e:
        print(f"  [Config Error] {e}")
        print("  Please update your .env file and try again.\n")
        sys.exit(1)

    while True:
        try:
            command = input("You: ").strip()

            if not command:
                continue

            if command.lower() in ["exit", "quit", "q", "bye"]:
                print("\nAgent: Goodbye!\n")
                break

            history = asyncio.run(run_agent(command))

            print("\n" + "-" * 55)
            print("Agent: Task complete.")
            result = history.final_result()
            if result:
                print(f"\n  {result}")
            print("-" * 55 + "\n")

        except KeyboardInterrupt:
            print("\n\nAgent: Interrupted. Goodbye!\n")
            sys.exit(0)

        except Exception as e:
            print(f"\nAgent: Something went wrong — {e}")
            print("Please try again or rephrase your command.\n")


if __name__ == "__main__":
    main()
