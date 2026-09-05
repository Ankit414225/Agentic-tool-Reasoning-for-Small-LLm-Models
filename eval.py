import json, re, os
from qwen_agent.agents import Assistant
from qwen_agent.llm.schema import Message, ContentItem
llm_cfg = {
    'model': 'Qwen/Qwen3-1.7B',
    'model_server': 'http://localhost:8000/v1',
    'api_key': 'EMPTY',
}
bot = Assistant(llm=llm_cfg)

def clean_answer(raw_content):
    return re.sub(r'<think>.*?</think>', ' ', raw_content, flags=re.DOTALL).strip()

def run_one_example(row):
  
    corpus_text = "\n\n".join(
        f"Title: {p['title']}\n{p['paragraph_text']}" for p in row['context']
    )
    
    prompt = f"Context:\n{corpus_text}\n\nQuestion: {row['question']}\nAnswer:"

    messages = [
        Message(role="user", content=prompt)
    ]

    final_response = None
    for response in bot.run(messages=messages):
        final_response = response

    model_answer = clean_answer(final_response[-1]['content'])
    return model_answer

results = []
correct=[]
for i, row in enumerate(ds):
    model_answer = run_one_example(row)
    correct.append(row['answer'].lower() in model_answer.lower())
    results.append({
        'id': row['id'],
        'question': row['question'],
        'expected': row['answer'],
        'model_answer': model_answer,
        'correct': correct,
        'hops': row['metadata']['hops'],
    })
    print(f"[{i+1}/{len(ds)}] hops={row['metadata']['hops']} correct={correct[-1]}")
accuracy = sum(correct) / len(correct)
print(f"Accuracy: {accuracy:.2%}")
