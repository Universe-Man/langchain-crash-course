import os
from dotenv import load_dotenv

# from langchain.llms import OpenAI
# from langchain_community.llms import OpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

# from langchain.chains import SimpleSequentailChain

# llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
# result = llm.invoke("What is the capital of France?")
# print(result.content)

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
# openai = OpenAI(model_name="gpt-3.5-turbo-instruct")
# print(f"The API Key is: {openai_api_key}")

# llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.6)
# name = llm.invoke(
#     "I want to open a restaurant for Italian food. Suggest a fancy name for this."
# )
# print(name.content)
# name = chain.invoke(prompt_template_name.format(cuisine="Italian"))

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.6)

prompt_template_name = PromptTemplate(
    input_variables=["cuisine"],
    template="I want to open a restaurant for {cuisine} food. Suggest a fancy name for this.",
)
# prompt_template_name.format(cuisine="Italian")
name_chain = {"cuisine": RunnablePassthrough()} | prompt_template_name | llm
# name = name_chain.invoke({"cuisine": "American"})
# print(name.content)

prompt_template_items = PromptTemplate(
    input_variables=["restaurant_name"],
    template="Suggest me some menu items for {restaurant_name}. Return it as a comma separated list.",
)

menu_items_chain = (
    {"restaurant_name": RunnablePassthrough()} | prompt_template_items | llm
)
# items = items_chain.invoke({"restaurant_name": name.content})

# print(name.content, items.content)

chain = {"cuisine": name_chain} | menu_items_chain
result = chain.invoke({"cuisine": "Mexican"})
print(result.content)
