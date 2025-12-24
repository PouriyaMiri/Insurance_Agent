from __future__ import annotations
from app_common import build_context, console, is_exit_phrase, print_agent
from dialogue import dialogue_manager


def start_call() -> None:
    ctx = build_context("./docs")

    print_agent(
        "Hello! Before we start: this conversation will be recorded for quality and claims handling in line with GDPR. "
        "By continuing, you consent to this recording. How can I help today?"
    )

    while True:
        user = console.input("[bold cyan]You[/bold cyan]: ").strip()
        if not user:
            continue
        if is_exit_phrase(user):
            break

        result = dialogue_manager(user, ctx.state, ctx.rag)
        print_agent(result.response_text, title="Agent")
        if result.end_call:
            break


if __name__ == "__main__":
    start_call()
