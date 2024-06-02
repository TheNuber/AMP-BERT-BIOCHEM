import torch
from torch import nn
from typing import Optional, Tuple, Union
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions


def change_device(device, *tensors):
    r"""
        Moves a series of PyTorch tensors to a certain device, and returns them.
        
        device   -- the identifier of the device
        tensors  -- Variable list of PyTorch tensors
    """
    changed = ()
    for t in tensors:
        if t is not None and isinstance(t, torch.Tensor):
            t = t.to(device)
        changed = changed + (t,)
    return changed

class MultiGPUBertEncoder(nn.Module):

    def __init__(self, encoder):
        r"""
            Copies a BertEncoder class, separating the BertLayer stack in half and sending
            each substack into a different GPU
        """
        super().__init__()
        self.config = encoder.config
        
        num_layers_1 = self.config.num_hidden_layers // 2
        self.layer1 = nn.ModuleList(encoder.layer[:num_layers_1]).to("cuda:0")
        self.layer2 = nn.ModuleList(encoder.layer[num_layers_1:]).to("cuda:1")
        
        self.gradient_checkpointing = encoder.gradient_checkpointing

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
    
        # Variable initialization
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        next_decoder_cache = () if use_cache else None
        
        # START CHANGE: MOVE TENSORS TO GPU 0
        hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, past_key_values = change_device(
            "cuda:0", 
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_values
        )
        # END CHANGE
        
        # Feed each layer as originally developed
        for i, layer_module in enumerate(self.layer1):

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # START CHANGE: MOVE TENSORS TO GPU 1
        hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, past_key_values = change_device(
            "cuda:1", 
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_values
        )
        # END CHANGE
        
        # Feed each layer as originally developed
        for i, layer_module in enumerate(self.layer2):   
        
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
            
        # START CHANGE: MOVE TENSORS BACK TO GPU 0
        hidden_states, next_decoder_cache, all_hidden_states, all_self_attentions, all_cross_attentions = change_device(
            "cuda:0", 
            hidden_states, 
            next_decoder_cache,
            all_hidden_states,
            all_self_attentions,
            all_cross_attentions
        )
        # END CHANGE
        
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
    
from transformers.models.bert.modeling_bert import BertModel

class MultiGPUBertModel(BertModel):
    def __init__(self, bert_model: BertModel):
        super(MultiGPUBertModel, self).__init__(bert_model.config)
        
        self.embeddings = bert_model.embeddings.to("cuda:0")
      # self.encoder = bert_model.encoder
        self.encoder = MultiGPUBertEncoder(bert_model.encoder)
        self.pooler = bert_model.pooler.to("cuda:0")
        
from transformers.models.bert.modeling_bert import BertForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput

import numpy as np

class MultiGPUBertForSequenceClassification(BertForSequenceClassification):
    def __init__(self, bert_model_for_sequence_classification):
        super(MultiGPUBertForSequenceClassification, self).__init__(bert_model_for_sequence_classification.config)
                
        self.num_labels = bert_model_for_sequence_classification.num_labels
        self.config = bert_model_for_sequence_classification.config

      # self.bert = BertModel(config)
        self.bert = MultiGPUBertModel(bert_model_for_sequence_classification.bert)

        self.dropout = nn.Dropout(0.01).to("cuda:0")
        self.classifier = nn.Linear(self.config.hidden_size, self.num_labels).to("cuda:0")
        
    def forward(
            self,
            dataloader_item = None,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ) -> SequenceClassifierOutput:
            r"""
                Como el clasificador base de la librería, pero sin la morralla de devolver diccionarios si se lo pides (nunca)
                Además puedes pasarle s
            """
            if dataloader_item != None:
                input_ids = dataloader_item['input_ids'].to("cuda:0")
                attention_mask = dataloader_item['attention_mask'].to("cuda:0") 
    
            # Paso los inputs por el bert para que me los devuelva codificados
            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
                
            # Aqui los recojo
            pooled_output = outputs[1]
    
            # Aplico dropout
            dropout_output = self.dropout(pooled_output)
            
            # Clasifico 
            logits = self.classifier(dropout_output)
            
            return SequenceClassifierOutput(
                loss=None,
                logits=logits,
                hidden_states=pooled_output,
                attentions=outputs.attentions,
            )
        
class MultiGPUBertForPeptideClassification(BertForSequenceClassification):
    def __init__(self, bert_model_for_sequence_classification, biochem_cols):
        super(MultiGPUBertForPeptideClassification, self).__init__(bert_model_for_sequence_classification.config)
                
        self.num_labels = bert_model_for_sequence_classification.num_labels
        self.config = bert_model_for_sequence_classification.config
        self.biochem_cols = biochem_cols

      # self.bert = BertModel(config)
        self.bert = MultiGPUBertModel(bert_model_for_sequence_classification.bert)

        self.dropout = nn.Dropout(0.01).to("cuda:0")
        self.classifier = nn.Linear(self.config.hidden_size + len(self.biochem_cols), self.num_labels).to("cuda:0")
        
    def forward(
            self,
            dataloader_item = None,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            biochem_info: Optional[torch.Tensor] = None,
        ) -> SequenceClassifierOutput:
            r"""
                Si se pasa un elemento de un dataloader, se sacan todos los parametros de ahí
                Si no, se usan el resto
            """
            if dataloader_item != None:
                input_ids = dataloader_item['input_ids'].to("cuda:0")
                attention_mask = dataloader_item['attention_mask'].to("cuda:0")
                biochem_info = dataloader_item['biochem_info'].to('cuda:0')

    
            # Paso los inputs por el bert para que me los devuelva codificados
            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
            )
                
            # Aqui los recojo
            pooled_output = outputs[1]
    
            # Aplico dropout
            dropout_output = self.dropout(pooled_output)
            
            # Concateno la información bioquimica
            output_with_biochem = torch.cat([dropout_output, biochem_info], dim = 1)
            
            # Clasifico 
            logits = self.classifier(output_with_biochem)
            
            return SequenceClassifierOutput(
                loss=None,
                logits=logits,
                hidden_states=pooled_output,
                attentions=outputs.attentions,
            )        
        

class NNClassifier(BertForSequenceClassification):
    def __init__(self, bert_model_for_sequence_classification, num_layers):
        super(NNClassifier, self).__init__(bert_model_for_sequence_classification.config)

        self.num_labels = bert_model_for_sequence_classification.num_labels
        self.config = bert_model_for_sequence_classification.config
        
      # self.bert = BertModel(config)
        self.bert = MultiGPUBertModel(bert_model_for_sequence_classification.bert)

        self.dropouts = nn.ModuleList([nn.Dropout(0.0).to("cuda:0") for i in range(num_layers)])
        self.hidden = nn.ModuleList([nn.Linear(self.config.hidden_size, self.config.hidden_size).to("cuda:0") for i in range(num_layers)])
        self.activations = nn.ModuleList([nn.LeakyReLU() for i in range(num_layers)])
        
        self.classifier = nn.Linear(self.config.hidden_size, self.num_labels).to("cuda:0")
        
    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ) -> SequenceClassifierOutput:
            r"""
                Como el clasificador base de la librería, pero sin la morralla de devolver diccionarios si se lo pides (nunca)
            """
    
            # Paso los inputs por el bert para que me los devuelva codificados
            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
                
            # Aqui los recojo
            tensor_outputs = outputs[1]

            for dropout, linear, activation in zip(self.dropouts, self.hidden, self.activations):
                
                # Aplico dropout, paso por la capa oculta y uso la funcion de activacion
                # print(outputs)
                tensor_outputs = dropout(tensor_outputs)
                # print(outputs)
                tensor_outputs = linear(tensor_outputs)
                # print(outputs)
                tensor_outputs = activation(tensor_outputs)
                # print(outputs)
                                
            # Clasifico 
            logits = self.classifier(tensor_outputs)
        
            return SequenceClassifierOutput(
                loss=None,
                logits=logits,
                hidden_states=outputs[1],
                attentions=outputs.attentions,
            )
 