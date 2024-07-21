from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
from langchain.chains import SequentialChain

import os

os.environ['GOOGLE_API_KEY'] = 'AIzaSyBvvS6ndhoQk_0SP2TYxuf7L1CldR2_X38'


llm = ChatGoogleGenerativeAI(model= 'gemini-pro', temperature=0.6)
#name = llm.invoke("I want to open a restaurant for Italin food. Suggest a fancy name for this.")
#print(name)
#prompt_template_name = PromptTemplate(
#    input_variables= ['cuisine'],
#    template='I want to open a restaurant for {cuisine} food. Suggest a fancy name for this. only one name please'
#)
#prompt_template_name.format(cuisine="Indian")
#chain = LLMChain(llm=llm, prompt=prompt_template_name)
#chain.run("Indian Kerala")


prompt_template_name = PromptTemplate (
    input_variables= ['cuisine'],
    template='I want to open a restaurant for {cuisine} food, Suggest a fancy name. Only one name please'
)

prompt_template_restaurant_menu = PromptTemplate (
    input_variables=["restaurant_name"],
    template='Suggest some menu items for {restaurant_name}. Retunt it as comma seperated list'
)

name_chain = LLMChain(llm=llm, prompt=prompt_template_name, output_key="restaurant_name")
food_item_chain = LLMChain(llm=llm, prompt=prompt_template_restaurant_menu, output_key="menu_items")

#simpleSequentialChain = SimpleSequentialChain(chains =[name_chain, food_item_chain])
#response = simpleSequentialChain.run("Maxican")
#print(response)

sequentialChain = SequentialChain (
    chains=[name_chain, food_item_chain],
    input_variables = ['cuisine'],
    output_variables = ['restaurant_name','menu_items']
)
result = sequentialChain({'cuisine','Arabic'})

print("==========")
print(result)