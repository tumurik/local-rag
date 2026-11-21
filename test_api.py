from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

response = client.chat.completions.create(
    model="gemma4:e4b",
    messages=[{"role": "user", "content": "Say hello in one sentence."}],
    temperature=0.3,
    reasoning_effort="none",
    max_tokens=100,
)
print(response.choices[0].message.content)
print(response.choices[0].finish_reason)