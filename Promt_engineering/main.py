import os
from huggingface_hub import InferenceClient
from test_env import test_queries
# from utils import load_instruction
# utils.py
def load_instruction(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# Load instruction.md content
instruction_prompt = load_instruction("instruction.md")

# Set up the client
client = InferenceClient(
   
)

# Loop through test queries and send them
for query in test_queries:
    messages = [
        {"role": "system", "content": instruction_prompt},
        {"role": "user", "content": f"Optimize the following SQL query:\n\n{query}"}
    ]

    completion = client.chat.completions.create(
        model="defog/llama-3-sqlcoder-8b",
        messages=messages,
    )

    optimized_query = completion.choices[0].message.content
    print("Original Query:\n", query)
    print("Optimized Query:\n", optimized_query)
    print("-" * 60)
