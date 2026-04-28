"""CLI entrypoint that runs ``LinkedInWebAgent`` exactly once.

Usage:

    poetry run python -m src.utility.browser
    # or
    poetry run python src/utility/browser.py

Each invocation:
- Constructs a fresh :class:`LinkedInWebAgent` bound to the local ``.browser_profile``
  Chromium user-data directory (so saved login state is reused).
- Renders a live spinner whose suffix updates as the agent moves through stages.
- On completion, prints the elapsed wall-clock time (formatted as minutes + seconds)
  and the aggregate token usage reported by ChatAnthropic.
"""

import asyncio

from src.utility.linkedin_web_agent import LinkedInWebAgent, _spin_until
from src.utility.spinner import Spinner


async def main() -> None:
    """Run LinkedInWebAgent once with a CLI spinner and print the summary."""
    agent = LinkedInWebAgent(
        user_data_dir=".browser_profile",
        profile_directory="Default",
    )

    spinner = Spinner()
    current_stage = ["starting"]
    stop = asyncio.Event()
    spin_task = asyncio.create_task(_spin_until(stop, current_stage, spinner))

    try:
        result = await agent.run(
            on_progress=lambda msg: current_stage.__setitem__(0, msg),
        )
    except Exception:
        stop.set()
        await spin_task
        spinner.finish("LinkedIn agent failed")
        raise

    stop.set()
    await spin_task
    spinner.finish(f"LinkedIn agent complete in {result['elapsed_pretty']}")

    usage = result.get("token_usage", {})
    print(
        f"Tokens — input: {usage.get('input_tokens', 0)}, "
        f"output: {usage.get('output_tokens', 0)}, "
        f"total: {usage.get('total_tokens', 0)} "
        f"(ai_turns={usage.get('ai_message_count', 0)}, "
        f"tool_calls={usage.get('tool_call_count', 0)})"
    )


if __name__ == "__main__":
    asyncio.run(main())
