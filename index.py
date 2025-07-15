import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSequence

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.6)

prompt_template_name = PromptTemplate(
    input_variables=["cuisine"],
    template="I want to open a restaurant for {cuisine} food. Suggest a fancy name for this.",
)

name_chain = {"cuisine": RunnablePassthrough()} | prompt_template_name | llm

prompt_template_items = PromptTemplate(
    input_variables=["restaurant_name"],
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
print(result["restaurant_name"].content)
print(result["menu_items"].content)
