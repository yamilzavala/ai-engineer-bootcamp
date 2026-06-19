from openai import OpenAI
from config import api_key

client = OpenAI(api_key=api_key)

def generate_text(prompt, max_tokens=10, temperature=0.7):
    response = client.chat.completions.create(
        max_tokens=max_tokens,
        temperature=temperature,
        model='gpt-3.5-turbo',
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

prompt = 'Once upon a time'
generated_text = generate_text(prompt, max_tokens=20, temperature=0)
print(prompt, generated_text)
