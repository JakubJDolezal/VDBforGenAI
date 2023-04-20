from __future__ import annotations
import faiss
from transformers import AutoTokenizer, XLMRobertaModel, AutoModel
from typing import Union
from VectorisationAndIndexCreation import SearchFunctions
import transformers as transformers
import re
import numpy as np


class VectorDatabase:
    def __init__(self,
                 list_of_strings: list[str],
                 encoder: Union[str, transformers.PreTrainedModel] = None,
                 tokenizer: Union[str, transformers.PreTrainedTokenizer] = None,
                 batch_size: int = 128,
                 splitting_choice: str = "paragraphs",
                 preload_index: bool = True
                 ):
        """

        :param list_of_strings: List of your strings that represent the context you care about
        :param encoder: Transformer model from huggingface in torch, defaults to XLMRobertaModel, can be given as model or string location or huggingface location
        :param tokenizer: Tokenizer of the above model, defaults to XLMRobertaModelTokeniser, can be a different location than the model
        :param batch_size: Batch size for encoding
        :param splitting_choice: What is the size of the context you wish to consider. Options are "paragraphs" and "sentence"
        :param preload_index: Whether you want to preload the index and keep it in memory
        """
        if encoder is not None:
            if encoder is str:
                self.encoder = AutoModel.from_pretrained(encoder)
                if tokenizer is None:
                    self.tokenizer = AutoTokenizer.from_pretrained(encoder)
                elif tokenizer is str:
                    self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
            else:
                self.encoder = encoder
                self.tokenizer = tokenizer
        else:
            self.encoder = XLMRobertaModel.from_pretrained("xlm-roberta-base")
            self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

        self.splitting_choice = splitting_choice
        self.list_of_list_of_strings = self.split_list(list_of_strings)
        self.batch_size = batch_size
        list_list_of_vectors = [
            SearchFunctions.vectorise_to_numpy(self.encoder, self.tokenizer, self.list_of_list_of_strings[i],
                                               batch_size=self.batch_size) for i in
            range(len(self.list_of_list_of_strings))]
        lengths = [len(lst) for lst in list_list_of_vectors]
        # create an array of indices indicating which original list each element in the flattened list corresponds to
        self.map_to_list_of_lists = np.concatenate([np.repeat(i, l) for i, l in enumerate(lengths)])
        self.map_to_list_of_lists_index = np.concatenate(
            [np.linspace(0, l, num=l, endpoint=False) for i, l in enumerate(lengths)])
        self.list_of_context_vectors_flattened = np.concatenate(list_list_of_vectors, axis=0)
        if preload_index:
            d = encoder.config.hidden_size()
            self.index = faiss.IndexFlatIP(d)
            self.index.add(self.list_of_context_vectors_flattened)
        else:
            self.index = None

    def get_context_from_entire_database(self, text, num_context=1):
        """

        :param text:
        :param num_context: how many instances of context you want
        :return: string of the context that was found
        """
        if self.index is not None:
            indices_returned = SearchFunctions.search_database(None, self.encoder, self.tokenizer, text,
                                                               num_samples=num_context, index=self.index)
        else:
            indices_returned = SearchFunctions.search_database(self.list_of_context_vectors_flattened, self.encoder,
                                                               self.tokenizer, text, num_samples=num_context)
        if num_context == 1:
            return self.list_of_list_of_strings[self.map_to_list_of_lists[indices_returned[0][0]]][
                self.map_to_list_of_lists_index[indices_returned[0][0]]]
        else:
            list_of_returned_contexts = [
                self.list_of_list_of_strings[self.map_to_list_of_lists[indices_returned[0][i]]][
                    self.map_to_list_of_lists_index[indices_returned[0][i]]] for i in
                range(num_context)]
            return ' '.join(list_of_returned_contexts)

    def add_string_to_context(self, string):
        previous_length = len(self.list_of_list_of_strings)
        self.list_of_list_of_strings.append(self.split_list([string]))
        self.map_to_list_of_lists = np.concatenate(
            [self.map_to_list_of_lists, np.repeat(previous_length, len(self.list_of_list_of_strings[-1]))])
        self.map_to_list_of_lists_index = np.concatenate(
            [self.map_to_list_of_lists_index, np.linspace(0, previous_length, num=previous_length, endpoint=False)])
        vector_list_of_string = SearchFunctions.vectorise_to_numpy(self.encoder, self.tokenizer,
                                                                   self.list_of_list_of_strings[-1],
                                                                   batch_size=self.batch_size)
        self.list_of__context_vectors_flattened = np.concatenate([self.list_of_context_vectors_flattened,
                                                                  vector_list_of_string], axis=0)

    def add_list_of_strings_to_context(self, new_list_of_strings):
        previous_length = len(self.list_of_list_of_strings)

        new_list_of_list_of_strings = self.split_list(new_list_of_strings)
        list_list_of_vectors = [
            SearchFunctions.vectorise_to_numpy(self.encoder, self.tokenizer, new_list_of_list_of_strings[i],
                                               batch_size=self.batch_size) for i in
            range(len(self.list_of_list_of_strings))]
        lengths = [len(lst) for lst in list_list_of_vectors]
        # create an array of indices indicating which original list each element in the flattened list corresponds to
        new_map_to_list_of_lists = np.concatenate([np.repeat(i, l) for i, l in enumerate(lengths)])
        new_map_to_list_of_lists_index = np.concatenate(
            [np.linspace(0, l, num=l, endpoint=False) for i, l in enumerate(lengths)])
        new_list_of_context_vectors_flattened = np.concatenate(list_list_of_vectors, axis=0)

        self.list_of_list_of_strings.extend(new_list_of_list_of_strings)
        self.map_to_list_of_lists = np.concatenate(
            [self.map_to_list_of_lists, new_map_to_list_of_lists])
        self.map_to_list_of_lists_index = np.concatenate(
            [self.map_to_list_of_lists_index, new_map_to_list_of_lists_index])
        self.list_of_context_vectors_flattened = np.concatenate([self.list_of__context_vectors_flattened,
                                                                  new_list_of_context_vectors_flattened], axis=0)

    def split_list(self, input_list):
        if self.splitting_choice == "sentences":
            # Split each string in the input list into sentences and add them to a new list
            result = []
            for s in input_list:
                # Use regular expressions to split the string into sentences
                sentences = re.findall(r".*?[.?!\n]+(?=\s|$|[A-Z])", s, re.DOTALL)
                # Remove empty strings and add the list of sentences to the result
                result.append([sentence.strip() for sentence in sentences if sentence.strip()])
            return result
        elif self.splitting_choice == "paragraphs":
            paragraphs_list = []
            for string in input_list:
                paragraphs = string.split('\n')  # Split the string into paragraphs using newline as delimiter
                paragraphs_list.append(paragraphs)
            return paragraphs_list
        else:
            # Return the input list as is but make into list of lists of single strings for consistency, i.e. whole documents
            return [[input_list[i]] for i in range(0, len(input_list))]
