"""Model Chat CLI — ultra-modern local LLM utility.

A Textual TUI for discovering, chatting with, stress-testing and
comparing local AI models across your network.
"""
import sys


def main() -> None:
    from ui.app import ModelChatApp
    app = ModelChatApp()
    app.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
