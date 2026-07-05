from dotenv import load_dotenv
import os
import openai

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
# print(api_key)

openai.api_key = api_key

client = openai.OpenAI()
completion = client.chat.completions.create(model = 'gpt-5-nano', 
                                            messages = [{'role': 'system', 'content': 'You are a helpful assistant.'}, 
                                                        {'role': 'user', 'content': 'Hello, how are you?'}])
print(completion.choices[0].message.content)



