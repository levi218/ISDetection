import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModel, BertModel

class NextSentenceClassifier(nn.Module):

    def __init__(self, bert_model_name='bert-base-uncased', state_dict=None):
        super(NextSentenceClassifier, self).__init__()

        # Instantiating BERT model object
        config = AutoConfig.from_pretrained(
            bert_model_name, output_hidden_states=True)

        if state_dict is not None:
            self.bert_layer = BertModel(config)
        else:
            self.bert_layer = AutoModel.from_pretrained(
                bert_model_name, config=config)

        if bert_model_name == 'bert-large-uncased':
            self.input_length = 1024
        else:
            self.input_length = 768

        self.fc1 = nn.Linear(self.input_length, 2)

        if state_dict is not None:
            self.load_state_dict(state_dict)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        '''
        Inputs:
            -input_ids : Tensor of shape [B, T] containing token ids of sequences
            -attn_masks : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
        '''

        device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')

        if input_ids is None:
            input_ids = torch.zeros(1, 64).long().to(device)
        if attention_mask is None:
            attention_mask = torch.zeros(1, 64).long().to(device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(1, 64).long().to(device)

        # Feeding the input to BERT model to obtain contextualized representations
        res = self.bert_layer(input_ids, attention_mask=attention_mask,
                              token_type_ids=token_type_ids, return_dict=True)

        # Obtaining the representation of [CLS] head

        logits = res['pooler_output']

        logits = self.fc1(logits)

        return logits
