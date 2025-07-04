from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import re
import torch

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None
)

generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
    
def clean_story(text):
    
    match = re.search(r'<\|assistant\|>\n(.*)', text, re.DOTALL)
    if match:
        text = match.group(1)

    text = re.sub(r'<\|.*?\|>', '', text)
    text = re.sub(r'You are a helpful assistant.*?\.', '', text, flags=re.IGNORECASE)
    
    lines = text.splitlines()
    story_lines = []
    for line in lines:
        line = line.strip("-• ")  
        if line:
            story_lines.append(line)

    story = " ".join(story_lines)
    return story.strip()

def format_prompt(system_msg, user_msg):
    return f"<|system|>\n{system_msg}\n<|user|>\n{user_msg}\n<|assistant|>\n"

def generate_story(prompt, max_new_tokens=300, temperature=0.8):
    
     formatted_prompt = format_prompt(
        "You are a helpful assistant that writes creative short stories.", prompt )

     result = generator(
        formatted_prompt,
        max_new_tokens=max_new_tokens,
        num_return_sequences=1,
        temperature=temperature,
        top_k=50,
        top_p=0.95,
        return_full_text=True )
     return clean_story(result[0]['generated_text'])
