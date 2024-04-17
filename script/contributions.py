import torch
import torch.nn.functional as F
from functools import partial
import collections
import torch.nn as nn
from pyaml_env import parse_config
config = parse_config("./config.yaml")


def get_module(model, module_name, layer, model_layer_name=None):
    parsed_module_name = module_name.split('.')
    tmp_module = model
    if model_layer_name:
        parsed_layer_name = model_layer_name.split('.')
        # Loop to find layers module
        for sub_module in parsed_layer_name:
            tmp_module = getattr(tmp_module, sub_module)
        # Select specific layer
        tmp_module = tmp_module[layer]
    # Loop over layer module to find module_name
    for sub_module in parsed_module_name:
        tmp_module = getattr(tmp_module, sub_module)
    return tmp_module

class ModelWrapper(nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()

        self.model = model

        self.modules_config = config['models'][model.config.model_type]

        self.num_attention_heads = self.model.config.num_attention_heads
        self.attention_head_size = int(self.model.config.hidden_size / self.model.config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

    def save_activation(self,name, mod, inp, out):
        self.func_inputs[name].append(inp)
        self.func_outputs[name].append(out)

    def clean_hooks(self):
        for k, v in self.handles.items():
            self.handles[k].remove()

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def get_model_info(self):
        ln1_name = self.modules_config['ln1']
        ln2_name = self.modules_config['ln2']
        values_name = self.modules_config['values']
        model_layer_name = self.modules_config['layer']
        dense_name = self.modules_config['dense']
        pre_layer_norm = self.modules_config['pre_layer_norm']
        lnf_name = self.modules_config['final_layer_norm']


        return {'ln1_name': ln1_name,
                'ln2_name': ln2_name,
                'values_name': values_name,
                'model_layer_name': model_layer_name,
                'dense_name': dense_name,
                'pre_layer_norm': pre_layer_norm,
                'lnf_name': lnf_name}

    def get_modules_model(self, layer):
        model_info_dict = self.get_model_info()  
        model_layer_name = model_info_dict['model_layer_name']

        dense = get_module(self.model, model_info_dict['dense_name'], layer, model_layer_name)
        fc1 = get_module(self.model, model_info_dict['fc1_name'], layer, model_layer_name)
        fc2 = get_module(self.model, model_info_dict['fc2_name'], layer, model_layer_name)
        ln1 = get_module(self.model, model_info_dict['ln1_name'], layer, model_layer_name)
        ln2 = get_module(self.model, model_info_dict['ln2_name'], layer, model_layer_name)

        return {'dense': dense,
                'fc1': fc1,
                'fc2': fc2,
                'ln1': ln1,
                'ln2': ln2}

    @torch.no_grad()
    def get_contributions(self, hidden_states_model, attentions, func_inputs, func_outputs):
        #   hidden_states_model: Representations from previous layer and inputs to self-attention. (batch, seq_length, all_head_size)
        #   attentions: Attention weights calculated in self-attention. (batch, num_heads, seq_length, seq_length)

        ln1_name = self.modules_config['ln1']
        ln2_name = self.modules_config['ln2']
        values_name = self.modules_config['values']
        model_layer_name = self.modules_config['layer']
        dense_name = self.modules_config['dense']
        pre_layer_norm = self.modules_config['pre_layer_norm']
        if pre_layer_norm == 'True':
            pre_layer_norm = True
        elif pre_layer_norm == 'False':
            pre_layer_norm = False

        model_importance_list = []
        resultants_list = []

        contributions_data = {}

        try:
            num_layers = self.model.config.n_layers
        except:
            num_layers = self.model.config.num_hidden_layers

        for layer in range(num_layers):
            hidden_states = hidden_states_model[layer].cpu().detach()
            attention_probs = attentions[layer].cpu().detach()

            #   value_layer: Value vectors calculated in self-attention. (batch, num_heads, seq_length, head_size)
            #   dense: Dense layer in self-attention. nn.Linear(all_head_size, all_head_size)
            #   LayerNorm: nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            #   pre_ln_states: Vectors just before LayerNorm (batch, seq_length, all_head_size)

            value_layer = self.transpose_for_scores(func_outputs[model_layer_name + '.' + str(layer) + '.' + values_name][0]).cpu().detach()
            
            dense = get_module(self.model, dense_name, layer, model_layer_name).cpu()
            ln1 = get_module(self.model, ln1_name, layer, model_layer_name).cpu()

            pre_ln_states = func_inputs[model_layer_name + '.' + str(layer) + '.' + ln1_name][0][0].cpu()
            post_ln_states = func_outputs[model_layer_name + '.' + str(layer) + '.' + ln1_name][0].cpu()
            
            # VW_O
            dense_bias = dense.bias
            
            dense = dense.weight.view(self.all_head_size, self.num_attention_heads, self.attention_head_size).cpu().detach()
            transformed_layer = torch.einsum('bhsv,dhv->bhsd', value_layer.to('cuda'), dense.to('cuda')).cpu() #(batch, num_heads, seq_length, all_head_size)
            del dense, value_layer
            # AVW_O
            # (batch, num_heads, seq_length, seq_length, all_head_size)
            weighted_layer = torch.einsum('bhks,bhsd->bhksd', attention_probs.to('cuda'), transformed_layer.to('cuda')).cpu()
            del transformed_layer, attention_probs
            
            # Sum each weighted vectors αf(x) over all heads:
            # (batch, seq_length, seq_length, all_head_size)
            summed_weighted_layer = weighted_layer.sum(dim=1) # sum over heads
            del weighted_layer

            # Make residual matrix (batch, seq_length, seq_length, all_head_size)
            hidden_shape = hidden_states.size()
            device = hidden_states.device
            residual = torch.einsum('sk,bsd->bskd', torch.eye(hidden_shape[1], dtype=torch.float16).to('cuda'), hidden_states.to('cuda')).cpu()
            del hidden_states
            # AVW_O + residual vectors -> (batch,seq_len,seq_len,embed_dim)
            residual_weighted_layer = summed_weighted_layer + residual
            del summed_weighted_layer, residual
            
            @torch.no_grad()
            def l_transform(x, w_ln):
                '''Computes mean and performs hadamard product with ln weight (w_ln) as a linear transformation.'''
                ln_param_transf = torch.diag(w_ln)
                ln_mean_transf = torch.eye(w_ln.size(0), dtype=torch.float16).to(w_ln.device) - \
                    1 / w_ln.size(0) * torch.ones_like(ln_param_transf).to(w_ln.device)
                out = torch.einsum(
                    '... e , e f , f g -> ... g',
                    x.to('cuda'),
                    ln_mean_transf.to('cuda'),
                    ln_param_transf.to('cuda')
                ).cpu()
                del ln_mean_transf, ln_param_transf
                x.cpu()
                return out

            if pre_layer_norm == False:
                # LN 1
                ln1_weight = ln1.weight.data
                ln1_eps = ln1.variance_epsilon
                # ln1_bias = ln1.bias
                # Transformed vectors T_i(x_j)
                transformed_vectors = l_transform(residual_weighted_layer, ln1_weight)
            else:
                transformed_vectors = residual_weighted_layer

            # Output vectors 1 per source token
            attn_output = transformed_vectors.sum(dim=2)

            if pre_layer_norm == False:
                # Lb_O
                ln_std_coef = 1/(pre_ln_states + ln1_eps).std(-1, unbiased=False).view(1, -1, 1)
                if dense_bias:
                    dense_bias_term = l_transform(dense_bias, ln1_weight)
                    resultant = (attn_output + dense_bias_term)*ln_std_coef
                else:
                    resultant = (attn_output)*ln_std_coef
                # y_i
                transformed_vectors_std = l_transform(residual_weighted_layer, ln1_weight)*ln_std_coef.unsqueeze(-1)
                real_resultant = post_ln_states
            else:
                dense_bias_term = dense_bias
                resultant = attn_output + dense_bias_term
                transformed_vectors_std = transformed_vectors
                pre_ln2_states = func_inputs[model_layer_name + '.' + str(layer) + '.' + ln2_name][0][0]
                if pre_ln2_states.dim() == 2:
                    pre_ln2_states = pre_ln2_states.unsqueeze(0)
                real_resultant = pre_ln2_states
                
            # Assert interpretable expression of attention is equal to the output of the attention block
            assert torch.dist(resultant, real_resultant).item() < 1e-3 * real_resultant.numel()
            del real_resultant, transformed_vectors, attn_output
            
            importance_matrix = -F.pairwise_distance(transformed_vectors_std, resultant.unsqueeze(2),p=1)
            
            model_importance_list.append(torch.squeeze(importance_matrix))
            
            resultants_list.append(torch.squeeze(resultant))
            
    


        contributions_model = torch.stack(model_importance_list)
        resultants_model = torch.stack(resultants_list)

        contributions_data['contributions'] = contributions_model
        contributions_data['resultants'] = resultants_model

        
        return contributions_data


    def __call__(self, input_model):
        with torch.no_grad():
            self.handles = {}
            for name, module in self.model.named_modules():
                self.handles[name] = module.register_forward_hook(partial(self.save_activation, name))

            self.func_outputs = collections.defaultdict(list)
            self.func_inputs = collections.defaultdict(list)

            output = self.model(**input_model, output_hidden_states=True, output_attentions=True)
            hidden_states = output['hidden_states']
            attentions = output['attentions']
            
            contributions_data = self.get_contributions(hidden_states, attentions, self.func_inputs, self.func_outputs)

            # Clean forward_hooks dictionaries
            self.clean_hooks()
            return contributions_data

