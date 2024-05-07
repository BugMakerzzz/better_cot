from config import deberta_path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
model = AutoModelForSequenceClassification.from_pretrained(deberta_path).to('cuda')
tokenizer = AutoTokenizer.from_pretrained(deberta_path)

premise = "If someone is round then they are green"
statement = "The cat is not nice"

input = tokenizer(premise, statement, return_tensors="pt")
output = model(input["input_ids"].to('cuda'))  # device = "cuda:0" or "cpu"
prediction = torch.softmax(output["logits"][0], -1).tolist()
label_names = ["contradiction", "neutral", "entailment"]
prediction = {name: round(float(pred) * 100, 1) for pred, name in zip(prediction, label_names)}
print(prediction)