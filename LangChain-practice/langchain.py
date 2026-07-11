import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import (PromptTemplate, 
                                    FewShotChatMessagePromptTemplate,
                                    HumanMessagePromptTemplate,
                                    ChatPromptTemplate,
                                    AIMessagePromptTemplate)
from langchain_core.output_parsers import (StrOutputParser, 
                                           CommaSeparatedListOutputParser)
from langchain_classic.output_parsers import DatetimeOutputParser

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

chat = ChatOpenAI(
    model="gpt-5.4-mini", 
    api_key=api_key,
    max_completion_tokens=2000,
    temperature=1,
    seed=365
)

# Example 1
# response = chat.invoke("What is the capital of France?")
# print(response.content)

# Example 2
# message_s = SystemMessage(content="You are Marv, a chatbot that reluctantly answers questions with sarcastic responses.")
# message_h = HumanMessage(content="I've recently adopted a dog. Can you suggest some dog names?")
# response2 = chat.invoke([message_s, message_h])
# print(response2.content)

# Example 3
# message_h_dog = HumanMessage(content="I've recently adopted a dog. Can you suggest some dog names?")
# message_ai_dog = AIMessage(content="Oh, absolutely. Because nothing screams 'I'm a responsible pet owner' like asking a chatbot to name your new furball. How about 'Bark Twain' (if it's a literary hound)? Or 'Snoopy' (if it's a cartoon character)? Or 'Max' (if it's a superhero?)")

# message_h_cat = HumanMessage(content = ''' I've recently adopted a cat. Can you suggest some cat names? ''')
# message_ai_cat = AIMessage(content = ''' Oh, absolutely. Because nothing screams "I'm a unique and creative individual" 
# like asking a chatbot to name your cat. How about "Furry McFurFace", "Sir Meowsalot", or "Catastrophe"? ''')

# message_h_fish = HumanMessage(content = ''' I've recently adopted a fish. Can you suggest some fish names? ''')

# response3 = chat.invoke([message_h_dog, message_ai_dog, message_h_cat, message_ai_cat, message_h_fish])
# print(response3.content)

# Example 4
# TEMPLATE = """
# System: 
# {description}

# Human:
# I've recently adopted a {pet}.
# Could you suggest some {pet} names?
# """

# prompt_template = PromptTemplate.from_template(template=TEMPLATE);
# prompt_value = prompt_template.invoke({
#     'description': ''' The chatbot should reluctantly answer questions with sarcastic responses. ''',
#     'pet': 'dog'
# })
# print(prompt_value.text)
# response_4 = chat.invoke(prompt_value.text)
# print(response_4.content)

# Example 5
# TEMPLATE_S = "{description}"
# TEMPLATE_H = """
# I've recently adopted a {pet}.
# Could you suggest some {pet} names?
# """

# message_template_s = SystemMessagePromptTemplate.from_template(template=TEMPLATE_S)
# message_template_h = HumanMessagePromptTemplate.from_template(template=TEMPLATE_H)

# chat_template = ChatPromptTemplate.from_messages([message_template_s, message_template_h])

# chat_value = chat_template.invoke({
#     'description': ''' The chatbot should reluctantly answer questions with sarcastic responses. ''',
#     'pet': 'dog'
# })

# chat_response = chat.invoke(chat_value)
# print(chat_response.content)

# Example 6
# TEMPLATE_H = """
# I've recently adopted a {pet}.
# Could you suggest some {pet} names?
# """
# TEMPLATE_AI = "{response}"

# message_template_h = HumanMessagePromptTemplate.from_template(template=TEMPLATE_H)
# message_template_ai = AIMessagePromptTemplate.from_template(template=TEMPLATE_AI)

# example_template = ChatPromptTemplate.from_messages([message_template_h, message_template_ai])

# examples = [{'pet':'dog', 
#              'response':'''Oh, absolutely. Because nothing screams "I'm a responsible pet owner" 
# like asking a chatbot to name your new furball. How about "Bark Twain" (if it's a literary hound)? '''}, 
            
#             {'pet':'cat', 
#              'response':'''Oh, absolutely. Because nothing screams "I'm a unique and creative individual" 
#              like asking a chatbot to name your cat. How about "Furry McFurFace", "Sir Meowsalot", or "Catastrophe"? '''}, 
            
#             {'pet':'fish', 
#              'response':
#              '''Oh, absolutely. Because nothing screams "I'm a fun and quirky pet owner" 
#              like asking a chatbot to name your fish. How about "Fin Diesel", "Gill Gates", or "Bubbles"?'''}]


# few_shot_prompt = FewShotChatMessagePromptTemplate(examples = examples,
#                                                    example_prompt = example_template,
#                                                    input_variables = ['pet'] 
# )

# chat_template = ChatPromptTemplate.from_messages([few_shot_prompt, message_template_h])
# chat_value = chat_template.invoke({'pet':'rabbit'})

# for i in chat_value.messages:
#     print(f'{i.type}: {i.content}\n')

# response = chat.invoke(chat_value)
# print(response.content)

# Example 7
# message_h = HumanMessage(content = "Can you give me an interesting fact I probably didn't know about?")
# response = chat.invoke([message_h])

# str_output_parser = StrOutputParser()
# response_parsed = str_output_parser.invoke(response);
# print(response_parsed)

# Example 8
# message_h = HumanMessage(content = f''' I've recently adopted a dog. Could you suggest some dog names? 
# {CommaSeparatedListOutputParser().get_format_instructions}''')

# print(message_h.content)

# response = chat.invoke([message_h])
# print(response.content)

# list_output_parser = CommaSeparatedListOutputParser()

# response_parsed = list_output_parser.invoke(response)
# print(response_parsed)

# for i in response_parsed:
#     print(i)

# print(len(response_parsed))

# Example 9
from datetime import datetime

parser = DatetimeOutputParser()

message_h = HumanMessage(content=f'''When was the Danish poet Piet Hein born?
{parser.get_format_instructions()}''')

response = chat.invoke([message_h])
print(response.content)

response_parsed = parser.invoke(response)
print(response_parsed)
print(response_parsed.year)
print(response_parsed.month)
print(datetime.now() - response_parsed)

