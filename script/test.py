from config import *
from model import ModelWrapper


evidences = "If James wants to have a long vacation, then James's favorite season is summer. If James wants to have a long vacation, then James's favorite season is summer"
statement = "If James's favorite season is not fall, then James's favorite season is not winter"
prompt = f"Given the evidence: {evidences}, please correct the following wrong statement: {statement}. The correct statement should be: "

# prompt = f"Is the following statement correct or wrong? Statement:{statement}."
modelwrapper = ModelWrapper('Llama2_13b')
model = modelwrapper.model
tokenizer = modelwrapper.tokenizer
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"].to(modelwrapper.device)
outputs = model.generate(input_ids, max_new_tokens=200)
response = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
print(response)