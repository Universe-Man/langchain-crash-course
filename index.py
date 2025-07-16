import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import AIMessage, HumanMessage
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI

# from tools import calculator

# from langchain.agents import AgentType, initialize_agent, load_tools

# from langchain.schema import AgentAction, AgentFinish
# from langchain.llms import OpenAI

# from langchain.agents.openai_functions.prompt import PROMPT
# import langchain

# print(langchain.__version__)

# from langchain.agents import AgentExecutor, create_tool_calling_agent


load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.6)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# llm_with_tools = llm.bind_tools([calculator])
# tools = load_tools(["wikipedia", "llm-math"], llm=llm)

# agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

# agent.run("When was Elon Musk born? What is his age right now in 2025?")

# print(response)

# memory.save_context(
#     {"input": "What's the weather like?"},
#     {"output": "I'm sorry, I don't have real-time weather information."},
# )


prompt_template_name = PromptTemplate(
    input_variables=["cuisine"],
    # MessagesPlaceholder(variable_name="chat_history"),
    template="I want to open a restaurant for {cuisine} food. Suggest a fancy name for this.",
)

name_chain = {"cuisine": RunnablePassthrough()} | prompt_template_name | llm


prompt_template_items = PromptTemplate(
    input_variables=["restaurant_name"],
    # MessagesPlaceholder(variable_name="chat_history"),
    template="Suggest me some menu items for {restaurant_name}. Return it as a comma separated list.",
)

menu_items_chain = (
    {"restaurant_name": RunnablePassthrough()} | prompt_template_items | llm
)

full_chain = RunnableSequence(
    RunnablePassthrough().assign(restaurant_name=name_chain)
    | RunnablePassthrough().assign(menu_items=menu_items_chain)
)


result = full_chain.invoke({"cuisine": "American"})

# memory.save_context(
#     {"cuisine": "American"}, {"output": result["restaurant_name"].content}
# )
# memory.save_context(
#     {"restaurant_name": result["restaurant_name"].content},
#     {"output": result["menu_items"].content},
# )


print(result["restaurant_name"].content)
print(result["menu_items"].content)
print(memory)
