from langchain_core.messages import HumanMessage
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

@tool
def add(a: float, b: float) -> str:
    """Adds two numbers together"""
    return f"The sum of {a} and {b} is {a+b}"

@tool
def subtract(a: float, b: float) -> str:
    """Subtract two numbers together"""
    return f"The different of {a} and {b} is {a-b}"

@tool
def multiply(a: float, b: float) -> str:
    """Subtract two numbers together"""
    return f"The product of {a} and {b} is {a*b}"

@tool
def divide(a: float, b: float) -> str:
    """Subtract two numbers together"""
    return f"The quotient of {a} and {b} is {a/b}"

def main():
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    tools = [add, subtract, multiply, divide]
    agent_executor = create_react_agent(model, tools)

    print("Welcome! I'm your AI assistant. Type 'quit' to exit.")
    print("You can ask me to perform calculations or chat with me.")

    while True:
        user_input = input("\nYou: ").strip()

        if user_input == "quit":
            break

        print("\nAssistant: ", end="")
        for chunk in agent_executor.stream({"messages": [HumanMessage(content=user_input)]}):
            if "agent" in chunk and "messages" in chunk["agent"]:
                for message in chunk["agent"]["messages"]:
                    print(message.content, end="")
        print()

if __name__ == "__main__":
    main()
