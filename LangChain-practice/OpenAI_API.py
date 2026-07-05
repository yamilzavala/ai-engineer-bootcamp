from dotenv import load_dotenv
import os
import openai

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
# print(api_key)

openai.api_key = api_key

client = openai.OpenAI()

# EXAMPLE ONE START
# completion = client.chat.completions.create(model = 'gpt-5-nano', 
#                                             messages = [{'role': 'system', 
#                                                          'content': 'You are Marv, a chatbot that reluctantly answers questions with sarcastic responses.'}, 
#                                                         {'role': 'user', 
#                                                           'content': ''' I've recently adopted a dog. Could you suggest some dog names? '''}])
# print(completion)
# print('------------')
# print(completion.choices[0].message.content)
# EXAMPLE ONE END

# EXAMPLE TWO START
# conpletionsTwo = client.chat.completions.create(
#     model='gpt-5-nano',
#     messages=[
#         {
#             'role': 'system',
#             'content': 'You are an assitant that classified comments as positive, negative, or neutral.'
#         },
#         {
#             'role': 'user',
#             'content': '''Oh, that movie was the worst movie that I've ever seen.'''
#         }
#     ],
# )

# print(conpletionsTwo.choices[0].message.content)
# EXAMPLE TWO END

# EXAMPLE THREE START
conpletionsThree = client.chat.completions.create(
    model='gpt-5-nano',
    messages=[
        {
            'role': 'user',
            'content': '''Could you explain briefly what a black hole is?'''
        }
    ],
    max_completion_tokens = 2000, 
    temperature = 1, 
    seed = 365, 
    stream = True,
)

for chunk in conpletionsThree:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
# EXAMPLE THREE END




