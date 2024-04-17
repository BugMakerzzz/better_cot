from typing import Any
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import *
import torch 
import numpy as np

def zero_grad(*obj):
    if len(obj) > 1:
        for subobj in obj:
            zero_grad(subobj)
    elif hasattr(obj[0], "parameters"):
        for subobj in obj[0].parameters():
            zero_grad(subobj)
    elif obj[0].grad is not None:
        obj[0].grad.data.zero_()


class ModelWrapper():

    def __init__(self, model_name):   
        self.model_name = model_name     
        self.is_baichuan = model_name.startswith('Baichuan')
        self.is_llama = model_name.startswith('Llama') 
        self.is_mistral = model_name.startswith('Mistral')
        self.is_falcon = model_name.startswith('Falcon')

        if self.is_llama:
            if '7b' in model_name:
                if 'chat' in model_name:
                    path = llama2_7b_chat_path
                else:
                    path = llama2_7b_path
            else:
                if 'chat' in model_name:
                    path = llama2_13b_chat_path
                else:
                    path = llama2_13b_path
        elif self.is_mistral:
            path = mistral_7b_path
        else:
            pass 
        self.tokenizer = AutoTokenizer.from_pretrained(path, torch_dtype=torch.float16)
        self.model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map='auto')
        self.n_layers = self.model.config.num_hidden_layers
        self.n_neurons = self.model.config.hidden_size
        self.n_heads = self.model.config.num_attention_heads
        self.device = self.model.device
        self.lm_head = self.model.lm_head
        self.inp_embed = self.model.get_input_embeddings()
    
       
    def __call__(self, input_ids, labels=None, return_dict=True, output_attentions=False, output_hidden_states=False) -> Any:
        if labels:
            outputs = self.model(
                input_ids=input_ids.to(self.device),
                labels=labels.to(self.device),
                return_dict=return_dict,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
        else:
            outputs = self.model(
                input_ids=input_ids.to(self.device),
                return_dict=return_dict,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
        return outputs
    
    def tokenize(self, text):
        return self.tokenizer.tokenize(text.strip())
    
    
    def generate(self, input):
        with torch.no_grad():
            self.model.eval()
            inputs = self.tokenizer(input, return_tensors="pt")
            input_ids = inputs["input_ids"].to(self.device)
            outputs = self.model.generate(input_ids, max_new_tokens=500)
            response = self.tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
            return response
    
    
    def prepare4explain(self, inp, ref):
        text = inp.strip() + " " + ref.strip()
        inp = self.tokenize(inp)
        ref = self.tokenize(ref)
        txt = self.tokenize(text)
        assert len(txt) - (len(inp) + len(ref)) <= 1, str(txt) + " | " + str(inp) + " | " + str(ref)
        # the insert blank may be splitted into multiple tokens
        ref = txt[len(inp):]
        return inp, ref, text
        
        
    def input_explain(self, inps, refs):
        self.model.eval()
        inps, refs, texts = self.prepare4explain(inps, refs) 
        ids = self.tokenizer(texts, return_tensors="pt")['input_ids']
        embs = self.inp_embed(ids.to(self.device)).detach().requires_grad_()
        bias = 0
        probs = torch.softmax(self.model(inputs_embeds=embs)["logits"], -1)
        
        ref = torch.tensor(self.tokenizer.convert_tokens_to_ids(refs)).long()
        obj = probs[0, torch.arange(len(inps) - bias, len(inps) + len(ref)- bias), ref]  
        grad = []
        for j in range(len(ref)): 
            zero_grad(self.model, embs)
            obj[j].backward(retain_graph=True)
            grad.append(embs.grad.data[0, 1 - bias:1 + len(inps) - bias].detach().cpu())
        
        with torch.no_grad():
            # importance
            emb = embs[0, 1 - bias:1 + len(inps) - bias].unsqueeze(0).cpu()
            grad = torch.stack(grad, 0).cpu()
            expl = (grad * emb).sum(axis=-1).T
            
            expls = expl.numpy()

        return inps, refs, expls
