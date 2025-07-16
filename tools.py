# from langchain_core.tools import tool

# import numexpr

# # from langchain.tools import LLMMathTool
# # from langchain_community.tools import WikipediaQueryRun
# # from langchain.utilities import WikipediaAPIWrapper


# @tool
# def calculator(expression: str) -> str:
#     """Evaluate a math expression using numexpr (e.g., '5 * (3 + 2)')."""
#     return str(numexpr.evaluate(expression.strip()))


# from langchain.chat_models import ChatOpenAI
from langchain.tools import tool


# Define a calculator tool
@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression and return the result."""
    import numexpr

    try:
        result = numexpr.evaluate(expression).item()
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"


# Initialize the LLM
# llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.6)

# Bind the tool to the LLM
# llm_with_tools = llm.bind_tools([calculator])

# Invoke the LLM with a prompt that uses the tool
# response = llm_with_tools.invoke("What is 48 * (7 + 3)?")

# print(response.content)  # Should print "480"
