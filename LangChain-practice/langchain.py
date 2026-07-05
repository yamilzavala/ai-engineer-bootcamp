import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

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
message_h_dog = HumanMessage(content="I've recently adopted a dog. Can you suggest some dog names?")
message_ai_dog = AIMessage(content="Oh, absolutely. Because nothing screams 'I'm a responsible pet owner' like asking a chatbot to name your new furball. How about 'Bark Twain' (if it's a literary hound)? Or 'Snoopy' (if it's a cartoon character)? Or 'Max' (if it's a superhero?)")

message_h_cat = HumanMessage(content = ''' I've recently adopted a cat. Can you suggest some cat names? ''')
message_ai_cat = AIMessage(content = ''' Oh, absolutely. Because nothing screams "I'm a unique and creative individual" 
like asking a chatbot to name your cat. How about "Furry McFurFace", "Sir Meowsalot", or "Catastrophe"? ''')

message_h_fish = HumanMessage(content = ''' I've recently adopted a fish. Can you suggest some fish names? ''')

response3 = chat.invoke([message_h_dog, message_ai_dog, message_h_cat, message_ai_cat, message_h_fish])
print(response3.content)