from typing import Any
from transformers import AutoTokenizer, AutoModelForCausalLM
from .config import *
import torch 
import numpy as np
import torch.nn.functional as F
from .openai_chat import chat_generate
import math
# from peft import PeftModel

class ModelWrapper():

    def __init__(self, model_name, remote=False):   
        self.model_name = model_name     
        self.is_llama = model_name.startswith('Llama') 
        self.is_vicuna = model_name.startswith('Vicuna')
        self.is_mistral = model_name.startswith('Mistral')
        self.is_qwen = model_name.startswith('Qwen')
        self.is_phi = model_name.startswith('Phi')
        self.is_yi = model_name.startswith('Yi')
        self.is_gemma = model_name.startswith('Gemma')
        self.is_o1 = model_name.startswith('o1')
        self.is_chat = True if 'chat' in model_name else False
        self.is_close = False
        self.skip_ids = []
        self.split_token = None 
        self.start_bias = 1
        self.end_bias = 1
        
        if self.is_llama:
            if '7b' in model_name:
                if self.is_chat:
                    path = llama2_7b_chat_path
                else:
                    path = llama2_7b_path
            elif 'Llama3_1' in model_name:
                self.skip_ids = [220, 25]
                self.split_token = '<|end_header_id|>'
                self.start_bias = 2
                self.end_bias = 3
                path = llama3_1_8b_chat_path
            elif 'Llama3' in model_name:
                self.skip_ids = [220, 25]
                self.split_token = '<|end_header_id|>'
                self.start_bias = 2
                self.end_bias = 3
                if self.is_chat:
                    path = llama3_8b_chat_path
                else:
                    path = llama3_8b_path
            elif 'moe' in model_name:
                path = llama_moe_path
            else:
                if self.is_chat:
                    path = llama2_13b_chat_path
                else:
                    path = llama2_13b_path
        elif self.is_mistral:
            self.skip_ids = [28747, 28705]
            self.split_token = 'INST'
            self.start_bias = 2
            self.end_bias = 3
            if self.is_chat:
                path = mistral_7b_chat_path
            else:
                path = mistral_7b_path
        elif self.is_phi:
            path = phi2_path
        elif self.is_qwen:
            self.split_token = '<|im_start|>'
            self.start_bias = 3
            self.end_bias = 2
            if 'Qwen1' in model_name:
                path = qwen1_8b_path
            else:
                if '3b' in model_name:
                    path = qwen2_5_3b_chat_path
                elif '7b' in model_name:
                    path = qwen2_5_7b_chat_path
                else:
                    path = qwen2_5_14b_chat_path
        elif self.is_yi:
            path = yi_1_5_6b_chat_path
        elif self.is_gemma:
            self.skip_ids = [235248,]
            self.split_token = '<start_of_turn>'
            self.start_bias = 3
            self.end_bias = 2
            if self.is_chat:
                path = gemma_2_9b_chat_path
            else:
                path = gemma_2_9b_path
        else:
            path = None
            self.is_close = True
            self.is_chat = True 
        
        if path:
            if remote:
                path = path.format(dir='usercache')
            else:
                path = path.format(dir='publiccache')
        
        if not self.is_close:
            self.tokenizer = AutoTokenizer.from_pretrained(path, torch_dtype=torch.float16, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map='auto', trust_remote_code=True)
            self.device = self.model.device
            self.lm_head = self.model.lm_head
            self.inp_embed = self.model.get_input_embeddings()

        # if self.is_qwen:
        #     self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

       
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
       
    
    def cal_logits(self, text, pred):
        with torch.no_grad():
            self.model.eval()
            text = self.tokenizer(text, return_tensors="pt")['input_ids'].to(self.device)
            pred_length = len(self.tokenizer(pred, return_tensors="pt")['input_ids'][0])
            logits = self.model(text)[0]
            logits = logits[0, -pred_length-1:-1, :]
            probs = F.softmax(logits, dim=-1)
            prob = []
            for i in range(pred_length):
                if text[0, i-pred_length] not in self.skip_ids:
                    prob.append(probs[i, text[0, i-pred_length]].item())
            prob = np.mean(np.array(prob))
            del text
            del logits
        return prob
    
    
    def cal_entropy(self, text):
        inputs = self.tokenizer(text, return_tensors='pt')
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        with torch.no_grad():
            # 获取模型输出，labels设置为input_ids以计算交叉熵
            outputs = self.model(input_ids, attention_mask=attention_mask, labels=input_ids)
            # outputs.loss 是交叉熵损失（平均每个token的损失）
            loss = outputs.loss
        
        # 计算总熵：交叉熵损失乘以token数量
        entropy = loss.item()
        return entropy
    
    
    def cal_cond_entropy(self, text, pred):
        with torch.no_grad():
            self.model.eval()
            text = self.tokenizer(text, return_tensors="pt")['input_ids'].to(self.device)
            pred_length = len(self.tokenizer(pred, return_tensors="pt")['input_ids'][0])
            logits = self.model(text).logits
            logits = logits[:, -pred_length-1:-1, :]
            # 计算目标句子每个词的条件概率
            entropys = []
            for i in range(pred_length):
                # 当前词的概率分布
                if text[0, i-pred_length] not in self.skip_ids:
                    probs = torch.softmax(logits[0, i], dim=-1)
                    target_id = text[0, i-pred_length]
                    prob = probs[target_id].item()  # 目标词的概率值
                    log_prob = math.log(prob + 1e-9)  # 加小值避免 log(0)
                    entropys.append(prob * log_prob)
            
            # 计算信息熵
            entropy = -sum(entropys) / len(entropys)
            return entropy
    # def merge_lora(self, lora_path):
    #     lora_model = PeftModel.from_pretrained(self.model, lora_path)
    #     self.model = lora_model
    
    
    def generate(self, input, sample_cnt=1):
        if self.is_close:
            result = chat_generate([input], model=self.model_name, max_tokens=5000, sample_cnt=sample_cnt)
            res = result[0][-1]['choices'][0]['message']['content']
        else:
          
            with torch.no_grad():
                self.model.eval()
                if self.is_chat:
                    inputs = self.tokenizer.apply_chat_template(input, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(self.device)
                else:
                    inputs = self.tokenizer(input, return_tensors="pt")["input_ids"].to(self.device)
                
                if sample_cnt > 1:
                    outputs = self.model.generate(inputs, 
                                                max_new_tokens=2048, 
                                                do_sample=True,
                                                temperature=0.7,
                                                top_k=50,
                                                top_p=0.95,
                                                num_return_sequences=sample_cnt,
                                                repetition_penalty=1.0)
                    res = [self.tokenizer.decode(outputs[i][len(inputs[0]):], skip_special_tokens=True) for i in range(sample_cnt)]
                else:
                    outputs = self.model.generate(inputs, max_new_tokens=2048, do_sample=False)
                    res = self.tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
                    
        return res

        
        
    def input_explain(self, inps, refs, L=10, b=1, p=2, eps=1e-7):
        def zero_grad(*obj):
            if len(obj) > 1:
                for subobj in obj:
                    zero_grad(subobj)
            elif hasattr(obj[0], "parameters"):
                for subobj in obj[0].parameters():
                    zero_grad(subobj)
            elif obj[0].grad is not None:
                obj[0].grad.data.zero_()

        self.model.eval()
        text = inps + refs 
        ids = self.tokenizer(text, return_tensors="pt")['input_ids'].to(self.device)
        pred_length = len(self.tokenizer(refs, return_tensors="pt")['input_ids'][0])
        embs = self.inp_embed(ids).detach().requires_grad_()
     
        probs = torch.softmax(self.model(inputs_embeds=embs)["logits"], -1)
        ref = ids[0, -pred_length:]
        
        obj = probs[0, range(len(ids[0])-pred_length-1, len(ids[0])-1), ref]  
        grad = []
        for j in range(pred_length): 
            zero_grad(self.model, embs)
            if ref[j] not in self.skip_ids:
                obj[j].backward(retain_graph=True)
                grad.append(embs.grad.data[0, :len(ids[0])].detach().cpu())

        with torch.no_grad():
            # importance
            emb = embs[0, :len(ids[0])].unsqueeze(0).cpu()
            grad = torch.stack(grad, 0).cpu()
            # expl = (grad * emb).sum(axis=-1).T
            grad_int = torch.zeros_like(emb*grad)
            for i in range(20):
                k = (i+1) / 20
                grad_int += k * grad
            expl = 1 / 20 * emb * grad_int
            expl = expl.sum(axis=-1).T
            
            expls = expl.numpy()
            zeros = np.zeros_like(expls)
            expls = expls / (expls.max(axis=0, keepdims=True) + eps)

            # expls = np.ceil(expls * L)
            expls = np.where(expls <= 0, zeros, expls)
          
            del ids
            del probs
            del obj
            del emb
            del grad
            del grad_int
            del expl
            # # l1 = expls.sum(axis=-1)
            # # lp = (expls ** p).sum(axis=-1) ** (1. / p) + eps
            # # input_scores = (l1 / lp)
            # input_scores = expls
        return expls
        # return inps, refs, input_scores
