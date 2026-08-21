"""
chat.py — Interactive command-line front end for the chatbot.

Run after train_model.py has produced the model artifacts.
"""

from chatbot import ChatBot

BANNER = """
==================================================
  AI Support Chatbot  (type 'quit' to exit)
  (type 'debug' before a message to see intent scores)
==================================================
"""


def main():
    print(BANNER)
    bot = ChatBot()
    print("Bot: Hi! I'm your support assistant. How can I help you today?\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "bye"):
            print("Bot: Goodbye! Have a great day.")
            break

        if user_input.lower().startswith("debug "):
            msg = user_input[6:]
            for intent, score in bot.top_intents(msg):
                print(f"    {intent:15s} {score:.1%}")
            continue

        result = bot.get_response(user_input)
        print(f"Bot: {result['response']}")
        print(f"     [intent: {result['intent']}, confidence: {result['confidence']:.0%}]\n")


if __name__ == "__main__":
    main()
