import torch

import torch.nn as nn


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class GalaChad(nn.Module):
    """ Make a classifier over a fine tuned bert model.

    Parameters
    __________
    bertFineTuned: BertModel
        A bert fine tuned instance

    """

    def __init__(self, bertFineTuned):
        super(GalaChad, self).__init__()
        self.bertFineTuned = bertFineTuned
        self.out_class = nn.Linear(768, 256)
        self.out_tokens = nn.Linear(768, 256)
        self.dropout = nn.Dropout(p=0.1)
        self.device = self.bertFineTuned.device

    def forward(self, ids, mask, embeddings=False, token_contrast=False, do_all=False, ml_contrast=False):
        """ Define how to perform each call

        Parameters
        __________
        ids: torch tensor (num_sentences, max_length) or (num_sentences, max_length, 768) of embeddings=True
            - the tokenised sentences, if already embedded from EDA.py then they embedded in 768 vector space
        mask: torch tensor (num_sentences, max_length)
            - the mask of the tensor 1 for unmasked, 0 otherwise
        embeddings - bool, if the ids are embedded already
        token_contrast - bool, if we want to get just the subword vectors returns (num_sentences,max_length, 768)
        do_all- bool, if we want the model to do everything (both sentence and subword vectors and return ((num_sentences, 768),(num_sentences,max_length, 768))
        ml_contrast- bool, if we want the model to do return the average embedding of the token linear layer returns (num_sentences, 768) from out_tokens
        Returns: torch tensor, (num_sentences, 768)
        _______
        """
        if not embeddings:
            ids = torch.squeeze(ids, dim=1)
            mask = torch.squeeze(mask, dim=1)
            total = self.bertFineTuned(ids, attention_mask=mask)
        else:
            total = self.bertFineTuned(inputs_embeds=ids, attention_mask=mask)
        if do_all:
            pooled_out = mean_pooling(total, mask)
            ap = self.dropout(pooled_out)
            output = self.out_class(ap)
            output = torch.tanh(output)
            tok_contrast = self.dropout(total[0])
            tok_contrast = self.out_tokens(tok_contrast)
            tok_contrast = torch.tanh(tok_contrast)
            return output, tok_contrast
        else:
            if token_contrast:
                if ml_contrast:
                    pooled_out = mean_pooling(total, mask)
                    tok_contrast = self.dropout(pooled_out)
                    tok_contrast = self.out_tokens(tok_contrast)
                    tok_contrast = torch.tanh(tok_contrast)
                    return tok_contrast
                else:
                    tok_contrast = self.dropout(total[0])
                    tok_contrast = self.out_tokens(tok_contrast)
                    tok_contrast = torch.tanh(tok_contrast)
                    return tok_contrast
            else:
                pooled_out = mean_pooling(total, mask)
                ap = self.dropout(pooled_out)
                output = self.out_class(ap)
                output = torch.tanh(output)
                return output
