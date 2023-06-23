from __future__ import annotations
import faiss
import numpy
import numpy as np
import torch

from transformers import AutoTokenizer, AutoModel
from typing import Union

from VDBforGenAI.Utilities.StringUtilities import split_string_to_dict
from VDBforGenAI.VectorisationAndIndexCreation import SearchFunctions
import transformers as transformers
import re
import os
from VDBforGenAI.Utilities.Loading import load_docx, load_pdf
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer




class VectorDatabase:
    def __init__(self,
                 encoder: Union[str, transformers.PreTrainedModel, bool] = None,
                 tokenizer: Union[str, transformers.PreTrainedTokenizer] = None,
                 batch_size: int = 128,
                 splitting_choice: str = "length",
                 index_location: str = './index',
                 preload_index: bool = False,
                 index_of_summarised_vector: int = 0,
                 hidden_size: int = False,
                 retain_strings = True
                 ):
        """

        :param encoder: Transformer model from huggingface in torch, defaults to facebook/dpr-ctx_encoder-single-nq-base
        , can be given as model or string location or string huggingface repo, it has to have the property self.encoder.config.hidden_size
        :param tokenizer: Tokenizer of the above model, defaults to
        facebook/dpr-ctx_encoder-single-nq-base, can be a different location than the model
        :param batch_size: Batch size for encoding
        :param splitting_choice: What is the size of the context you wish to consider. Options are
        "paragraphs" and "sentence", 'length'
        :param preload_index: Whether you want to preload the index and keep it in memory
        :param retain_strings: Whether you wish to retain the strings withing the VectorDatabase
        """
        self.index = None
        self.retain_strings = retain_strings
        self.list_of_context_vectors_flattened = None
        self.map_to_list_of_lists = None
        self.map_to_list_of_lists_index = None
        self.list_of_lists_of_strings = None
        self.list_dict_value_num = None
        self.list_locations = None
        self.index_of_summarised_vector = index_of_summarised_vector
        self.index_loc = index_location

        if preload_index and os.path.exists(index_location):
            self.load_index()
            self.index_loaded = True
        else:
            self.index_loaded = False

        # This instantiates the dictionary holding the levels and their possible values (usually based on folder structure of import)
        self.dlv = None
        if encoder is not None and encoder is not False:
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
            self.encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
            self.tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
        if hidden_size:
            self.d = hidden_size
        else:
            self.d = self.encoder.config.hidden_size

        self.batch_size = batch_size
        self.splitting_choice = splitting_choice

    # Get relevant indices or pieces of text

    def get_context_from_entire_database(self, text, num_context=1):
        """

        :param text: the prompt text
        :param num_context: how many instances of context you want
        :return: string of the context that was found
        """
        if self.index is not None:
            indices_returned = SearchFunctions.search_database(None, self.encoder, self.tokenizer, text,
                                                               self.index_of_summarised_vector, self.d,
                                                               num_samples=num_context, index=self.index)
        else:
            indices_returned = SearchFunctions.search_database(self.list_of_context_vectors_flattened, self.encoder,
                                                               self.tokenizer, text, self.index_of_summarised_vector,
                                                               self.d,
                                                               num_samples=num_context)
        if num_context == 1:
            return self.list_of_lists_of_strings[int(self.map_to_list_of_lists[int(indices_returned)])][int(
                self.map_to_list_of_lists_index[int(indices_returned)])]
        else:
            list_of_returned_contexts = [
                self.list_of_lists_of_strings[int(self.map_to_list_of_lists[indices_returned[i]])][
                    int(self.map_to_list_of_lists_index[indices_returned[i]])] for i in
                range(num_context)]
            return ' '.join(list_of_returned_contexts)

    def get_context_indices_from_entire_database(self, text, num_context=1):
        """

        :param text: the prompt text
        :param num_context: how many instances of context you want
        :return: indices of the context that was found corresponding used indices in list_of_lists_of_strings
        """
        if self.index is not None:
            indices_returned = SearchFunctions.search_database(None, self.encoder, self.tokenizer, text,
                                                               self.index_of_summarised_vector, self.d,
                                                               num_samples=num_context, index=self.index)
        else:
            indices_returned = SearchFunctions.search_database(self.list_of_context_vectors_flattened, self.encoder,
                                                               self.tokenizer, text, self.index_of_summarised_vector, self.d,
                                                               num_samples=num_context)
        if num_context == 1:
            return (int(self.map_to_list_of_lists[int(indices_returned)]), int(
                self.map_to_list_of_lists_index[int(indices_returned)]))
        else:
            list_of_returned_context_indices = [(int(self.map_to_list_of_lists[indices_returned[i]]),
                                                 int(self.map_to_list_of_lists_index[indices_returned[i]])) for i in
                                                range(num_context)]
            return list_of_returned_context_indices

    def get_context_from_index(self, text: str, loc_index: faiss.Index, selection_map_to_list_of_lists: numpy.array,
                               selection_map_to_list_of_lists_index: numpy.array, num_context: int = 1):
        """

        :param selection_map_to_list_of_lists_index: the selection mapping to the index within each specific document
        :param selection_map_to_list_of_lists: the selection mapping to the specific document
        :param loc_index: index of a selection
        :param text: the prompt text
        :param num_context: how many instances of context you want
        :return: string of the context that was found
        """
        indices_returned = SearchFunctions.search_database(None, self.encoder, self.tokenizer, text,
                                                           self.index_of_summarised_vector, self.d,
                                                           num_samples=num_context, index=loc_index)

        return indices_returned

    def get_context_indices_from_selection(self, text: str, level: int, key: str, num_context=1):
        """

        :param text: the prompt text
        :param level: which level we want
        :param key: which key on this level we want (often either directory or file)
        :param num_context: how many instances of context you want
        :return: string of context
        """
        selection = self.dlv[level] == self.list_dict_value_num[level][key]
        selection_map_to_list_of_lists = self.map_to_list_of_lists[
            np.isin(self.map_to_list_of_lists, np.argwhere(selection))]
        selection_map_to_list_of_lists_index = self.map_to_list_of_lists_index[
            np.isin(self.map_to_list_of_lists, np.argwhere(selection))]
        selection_list_of_context_vectors_flattened = self.list_of_context_vectors_flattened[
            np.isin(self.map_to_list_of_lists, np.argwhere(selection))]
        loc_index = faiss.IndexFlatIP(self.d)
        loc_index.add(selection_list_of_context_vectors_flattened)
        indices_returned=self.get_context_from_index(text, loc_index, selection_map_to_list_of_lists,
                                           selection_map_to_list_of_lists_index, num_context=num_context)
        if num_context == 1:
            return (int(selection_map_to_list_of_lists[int(indices_returned)]), int(
                selection_map_to_list_of_lists_index[int(indices_returned)]))
        else:
            list_of_returned_context_indices = [(int(selection_map_to_list_of_lists[indices_returned[i]]),
                                                 int(selection_map_to_list_of_lists_index[indices_returned[i]])) for i in
                                                range(num_context)]
            return list_of_returned_context_indices

    def get_context_from_selection(self, text: str, level: int, key: str, num_context=1):
        """

        :param text: the prompt text
        :param level: which level we want
        :param key: which key on this level we want (often either directory or file)
        :param num_context: how many instances of context you want
        :return: string of context
        """
        selection = self.dlv[level] == self.list_dict_value_num[level][key]
        selection_map_to_list_of_lists = self.map_to_list_of_lists[
            np.isin(self.map_to_list_of_lists, np.argwhere(selection))]
        selection_map_to_list_of_lists_index = self.map_to_list_of_lists_index[
            np.isin(self.map_to_list_of_lists, np.argwhere(selection))]
        selection_list_of__context_vectors_flattened = self.list_of_context_vectors_flattened[
            np.isin(self.map_to_list_of_lists, np.argwhere(selection))]
        loc_index = faiss.IndexFlatIP(self.d)
        loc_index.add(selection_list_of__context_vectors_flattened)
        indices_returned=self.get_context_from_index(text, loc_index, selection_map_to_list_of_lists,
                                           selection_map_to_list_of_lists_index, num_context=num_context)
        if num_context == 1:
            return self.list_of_lists_of_strings[int(selection_map_to_list_of_lists[int(indices_returned)])][int(
                selection_map_to_list_of_lists_index[int(indices_returned)])]
        else:
            list_of_returned_contexts = [
                self.list_of_lists_of_strings[int(selection_map_to_list_of_lists[indices_returned[i]])]
                                                 [int(selection_map_to_list_of_lists_index[indices_returned[i]])] for i in
                                                range(num_context)]
            return ' '.join(list_of_returned_contexts)

    def add_string_to_context(self, string, preload_index=None, dlv_handled=False):
        if preload_index is None:
            preload_index = self.index_loaded
        if self.list_of_context_vectors_flattened is None:
            self.initial_string_addition([string])
            previous_length = 0
        else:
            previous_length = self.map_to_list_of_lists[-1] + 1
            if self.retain_strings:
                self.list_of_lists_of_strings.extend(
                    self.split_list_of_strings_into_lists_of_lists_of_strings([string]))
            else:
                self.list_of_lists_of_strings = self.split_list_of_strings_into_lists_of_lists_of_strings([string])

            self.map_to_list_of_lists = np.concatenate(
                [self.map_to_list_of_lists, np.repeat(previous_length, len(self.list_of_lists_of_strings[-1]))])
            self.map_to_list_of_lists_index = np.concatenate(
                [self.map_to_list_of_lists_index, np.linspace(0, len(self.list_of_lists_of_strings[-1]),
                                                              num=len(self.list_of_lists_of_strings[-1]),
                                                              endpoint=False)])
            vector_list_of_string = SearchFunctions.vectorise_to_numpy(self.encoder, self.tokenizer,
                                                                       self.list_of_lists_of_strings[-1],
                                                                       self.batch_size, self.index_of_summarised_vector)
            self.list_of_context_vectors_flattened = np.concatenate([self.list_of_context_vectors_flattened,
                                                                     vector_list_of_string], axis=0)
        if self.retain_strings == False:
            self.list_of_lists_of_strings = None

        if preload_index:
            self.reload_total_index()
        else:
            self.index_loaded = False
        if not dlv_handled:
            self.add_to_dlv({0: 'String ' + str(previous_length)})

    def add_list_of_strings_to_context(self, new_list_of_strings, preload_index=None, dlv_handled=False):
        if preload_index:
            preload_index = self.index_loaded

        if self.list_of_context_vectors_flattened is None:
            self.initial_string_addition(new_list_of_strings)
            previous_length = 0
        else:
            previous_length = self.map_to_list_of_lists[-1] + 1
            new_list_of_list_of_strings = self.split_list_of_strings_into_lists_of_lists_of_strings(new_list_of_strings)
            list_list_of_vectors = [
                SearchFunctions.vectorise_to_numpy(self.encoder, self.tokenizer, new_list_of_list_of_strings[i],
                                                   self.batch_size, self.index_of_summarised_vector) for i in
                range(len(new_list_of_strings))]
            lengths = [len(lst) for lst in list_list_of_vectors]
            # create an array of indices indicating which original list each element in the flattened list corresponds to
            new_map_to_list_of_lists = np.concatenate([np.repeat(i, l) for i, l in enumerate(lengths)])
            new_map_to_list_of_lists_index = np.concatenate(
                [np.linspace(0, l, num=l, endpoint=False) for i, l in enumerate(lengths)])
            new_list_of_context_vectors_flattened = np.concatenate(list_list_of_vectors, axis=0)

            if self.retain_strings:
                self.list_of_lists_of_strings.extend(new_list_of_list_of_strings)

            self.map_to_list_of_lists = np.concatenate(
                [self.map_to_list_of_lists, new_map_to_list_of_lists])
            self.map_to_list_of_lists_index = np.concatenate(
                [self.map_to_list_of_lists_index, new_map_to_list_of_lists_index])
            self.list_of_context_vectors_flattened = np.concatenate([self.list_of_context_vectors_flattened,
                                                                     new_list_of_context_vectors_flattened], axis=0)
            if preload_index:
                self.reload_total_index()
            else:
                self.index_loaded = False

        if not dlv_handled:
            for i in range(len(new_list_of_strings)):
                self.add_to_dlv({0: 'String' + str(previous_length + i)})

    def initial_string_addition(self, list_of_strings):
        if self.retain_strings:
            self.list_of_lists_of_strings = self.split_list_of_strings_into_lists_of_lists_of_strings(list_of_strings)
        list_list_of_vectors = [
            SearchFunctions.vectorise_to_numpy(self.encoder, self.tokenizer, self.list_of_lists_of_strings[i],
                                               self.batch_size, self.index_of_summarised_vector) for i in
            range(len(self.list_of_lists_of_strings))]
        lengths = [len(lst) for lst in list_list_of_vectors]
        # create an array of indices indicating which original list each element in the flattened list corresponds to
        self.map_to_list_of_lists = np.concatenate([np.repeat(i, l) for i, l in enumerate(lengths)])
        self.map_to_list_of_lists_index = np.concatenate(
            [np.linspace(0, l, num=l, endpoint=False) for i, l in enumerate(lengths)])
        self.list_of_context_vectors_flattened = np.concatenate(list_list_of_vectors, axis=0)

    # In construction section

    def add_string_to_context_with_precalculated_vector(self, string, vectors, preload_index=None, dlv_handled=False,
                                                        presplit=False):
        if preload_index is None:
            preload_index = self.index_loaded
        if self.list_of_context_vectors_flattened is None:
            self.initial_string_addition_with_precalculated_vector([string], [vectors.numpy()])
            previous_length = 0
        else:
            previous_length = len(self.list_of_context_vectors_flattened)
            if self.retain_strings:

                if not presplit:
                    self.list_of_lists_of_strings.extend(
                        self.split_list_of_strings_into_lists_of_lists_of_strings([string]))
                else:
                    self.list_of_lists_of_strings.extend(string)

            self.map_to_list_of_lists = np.concatenate(
                [self.map_to_list_of_lists, np.repeat(previous_length, len(self.list_of_lists_of_strings[-1]))])
            self.map_to_list_of_lists_index = np.concatenate(
                [self.map_to_list_of_lists_index, np.linspace(0, len(self.list_of_lists_of_strings[-1]),
                                                              num=len(self.list_of_lists_of_strings[-1]),
                                                              endpoint=False)])
            vector_list_of_string = vectors.numpy()
            self.list_of_context_vectors_flattened = np.concatenate([self.list_of_context_vectors_flattened,
                                                                     vector_list_of_string], axis=0)
        if preload_index:
            self.reload_total_index()
        else:
            self.index_loaded = False
        if not dlv_handled:
            self.add_to_dlv({0: 'String ' + str(previous_length)})

    def add_list_of_strings_to_context_with_precalculated_vectors(self, new_list_of_strings, vectors,
                                                                  preload_index=None, dlv_handled=False,
                                                                  presplit=False):
        if preload_index:
            preload_index = self.index_loaded

        if self.list_of_lists_of_strings is None:
            self.initial_string_addition_with_precalculated_vector(new_list_of_strings, vectors)
            previous_length = 0

        else:
            previous_length = len(self.list_of_context_vectors_flattened)
            lengths = [len(lst) for lst in vectors]
            # create an array of indices indicating which original list each element in the flattened list corresponds to
            new_map_to_list_of_lists = np.concatenate([np.repeat(i, l) for i, l in enumerate(lengths)])
            new_map_to_list_of_lists_index = np.concatenate(
                [np.linspace(0, l, num=l, endpoint=False) for i, l in enumerate(lengths)])
            new_list_of_context_vectors_flattened = np.concatenate(vectors, axis=0)

            if self.retain_strings:
                if not presplit:
                    self.list_of_lists_of_strings.extend(
                        self.split_list_of_strings_into_lists_of_lists_of_strings(new_list_of_strings))
                else:
                    self.list_of_lists_of_strings.extend(new_list_of_strings)
            self.map_to_list_of_lists = np.concatenate(
                [self.map_to_list_of_lists, new_map_to_list_of_lists])
            self.map_to_list_of_lists_index = np.concatenate(
                [self.map_to_list_of_lists_index, new_map_to_list_of_lists_index])
            self.list_of_context_vectors_flattened = np.concatenate([self.list_of_context_vectors_flattened,
                                                                     new_list_of_context_vectors_flattened], axis=0)
            if preload_index:
                self.reload_total_index()
            else:
                self.index_loaded = False

        if not dlv_handled:
            for i in range(len(new_list_of_strings)):
                self.add_to_dlv({0: 'String' + str(previous_length + i)})

    def initial_string_addition_with_precalculated_vector(self, list_of_strings, vectors):
        if self.retain_strings:
            self.list_of_lists_of_strings = list_of_strings
        lengths = [len(lst) for lst in vectors]
        # create an array of indices indicating which original list each element in the flattened list corresponds to
        self.map_to_list_of_lists = np.concatenate([np.repeat(i, l) for i, l in enumerate(lengths)])
        self.map_to_list_of_lists_index = np.concatenate(
            [np.linspace(0, l, num=l, endpoint=False) for i, l in enumerate(lengths)])
        self.list_of_context_vectors_flattened = np.concatenate(vectors, axis=0)

    # String splitting function

    def split_list_of_strings_into_lists_of_lists_of_strings(self, input_list: list, splitting_choice: str = None,
                                                             max_length: int = 500):
        """
        :param input_list: list of strings
        :param splitting_choice: how to split, options are sentences, paragraphs, length or not at all
        :param max_length: if splitting by length how long the strings should be
        :return: list of list of strings
        """
        if splitting_choice is None:
            splitting_choice = self.splitting_choice

        if splitting_choice == "sentences":
            # Split each string in the input list into sentences and add them to a new list
            result = []
            for s in input_list:
                # Use regular expressions to split the string into sentences
                sentences = re.findall(r".*?[.?!\n]+(?=\s|$|[A-Z])", s, re.DOTALL)
                # Remove empty strings and add the list of sentences to the result
                result.append([sentence.strip() for sentence in sentences if sentence.strip()])
            # Remove any empty sub-lists from the result list
            result = [x for x in result if x]
            return result
        elif splitting_choice == "paragraphs":
            paragraphs_list = []
            for string in input_list:
                paragraphs = string.split('\n')  # Split the string into paragraphs using newline as delimiter
                # Remove empty strings and add the list of paragraphs to the result
                paragraphs_list.append([paragraph.strip() for paragraph in paragraphs if paragraph.strip()])
            # Remove any empty sub-lists from the result list
            paragraphs_list = [x for x in paragraphs_list if x]
            return paragraphs_list
        elif splitting_choice == "length":
            # Split each string in the input list into substrings of maximum length and add them to a new list
            result = []
            for s in input_list:
                if max_length is None:
                    # If no maximum length is specified, return the input list as is
                    result.append([s])
                else:
                    # Split the string into substrings of maximum length and add them to the result
                    substrings = [s[i:i + max_length] for i in range(0, len(s), max_length)]
                    # Remove empty strings and add the list of substrings to the result
                    result.append([substring.strip() for substring in substrings if substring.strip()])
            # Remove any empty sub-lists from the result list
            result = [x for x in result if x]
            return result
        else:
            # Return the input list as is but make into list of lists of single strings for consistency, i.e. whole documents
            return [[input_list[i]] for i in range(0, len(input_list))]

    # Loading functions for files and lists of strings

    def load_pdf(self, filename, divide_by_filepath=None, preload_index=None):
        """
        loads the pdf, adds it to all the arrays and the dictionary of levels and values
        :param filename: The pdf to load
        :param divide_by_filepath: whether it should be added to the dlv with folders/subfolders as levels and values
        :return:
        """
        pdf_string = load_pdf(filename)
        self.add_to_filenames(filename)
        if divide_by_filepath:
            filename_divided = split_string_to_dict(filename)
            self.add_string_to_context(pdf_string, dlv_handled=True, preload_index=preload_index)
            self.add_to_dlv(filename_divided)
        else:
            self.add_string_to_context(pdf_string)

    def load_docx(self, filename, divide_by_filepath=True, preload_index=None):
        """
        loads the docx, adds it to all the arrays and the dictionary of levels and values
        :param filename: The docx to load
        :param divide_by_filepath: whether it should be added to the dlv with folders/subfolders as levels and values
        :return:
        """
        word_string = load_docx(filename)
        self.add_to_filenames(filename)
        if divide_by_filepath:
            filename_divided = split_string_to_dict(filename)
            self.add_string_to_context(word_string, dlv_handled=True, preload_index=preload_index)
            self.add_to_dlv(filename_divided)
        else:
            self.add_string_to_context(word_string)

    def load_txt(self, filename, divide_by_filepath=True, preload_index=None):
        """
        loads the txt, adds it to all the arrays and the dictionary of levels and values
        :param filename: The txt to load
        :param divide_by_filepath: whether it should be added to the dlv with folders/subfolders as levels and values
        :return:
        """
        with open(filename, 'r') as f:
            # read contents of file as a string
            txt_string = f.read()
        self.add_to_filenames(filename)
        if divide_by_filepath:
            filename_divided = split_string_to_dict(filename)
            self.add_string_to_context(txt_string, dlv_handled=True, preload_index=preload_index)
            self.add_to_dlv(filename_divided)
        else:
            self.add_string_to_context(txt_string)

    def load_string_list_with_divisions(self, string_list, divisions, filenames=None):
        if filenames == None:
            filenames = [''] * len(string_list)
        preload_index = False
        for i in range(0, len(string_list)):
            if i == len(string_list) - 1:
                preload_index = None
            self.add_to_filenames(filenames[i])
            self.add_to_dlv(divisions[i])
            self.add_string_to_context(string_list[i], dlv_handled=True, preload_index=preload_index)

    # In construction function

    def load_string_list_with_divisions_and_vectors(self, string_list, divisions, vectors, filenames=None):
        """

        :param string_list: list of strings we wish to add
        :param divisions: list of divisions we wish to use in these strings (could be folder structure or anything else)
        :param vectors: torch tensor of precalculated vectors
        :param filenames: optional list of filenames where we got these string lists from
        :return:
        """
        if filenames == None:
            filenames = [''] * len(string_list)
        for i in range(0, len(string_list)):
            self.add_to_filenames(filenames[i])
            self.add_to_dlv(divisions[i])
        self.add_list_of_strings_to_context_with_precalculated_vectors(string_list, vectors, dlv_handled=True,
                                                                         presplit=True)

    def load_pdf_list(self, list_of_filenames, divide_by_filepath=True):
        preload_index = False
        for i in range(len(list_of_filenames)):
            item = list_of_filenames[i]
            if i == len(list_of_filenames) - 1:
                preload_index = None
            self.load_pdf(item, divide_by_filepath, preload_index=preload_index)

    def load_docx_list(self, list_of_filenames, divide_by_filepath=True):
        preload_index = False
        for i in range(len(list_of_filenames)):
            item = list_of_filenames[i]
            if i == len(list_of_filenames) - 1:
                preload_index = None
            self.load_docx(item, divide_by_filepath, preload_index=preload_index)

    def load_txt_list(self, list_of_filenames, divide_by_filepath=True):
        preload_index = False
        for i in range(len(list_of_filenames)):
            item = list_of_filenames[i]
            if i == len(list_of_filenames) - 1:
                preload_index = None
            self.load_txt(item, divide_by_filepath, preload_index=preload_index)

    def load_all_in_directory(self, directory):
        """
        Loads all pdfs, txts, and docxs in directory
        :param directory:
        :return:
        """
        docx_docs = []
        # doc_docs = []
        txt_docs = []
        pdf_docs = []

        # loop through all files and subdirectories
        for root, dirs, files in os.walk(directory):
            # find all docx/doc files in current directory
            for file in files:
                if file.endswith('.docx'):
                    docx_docs.append(os.path.join(root, file))
                # find all txt files in current directory
                elif file.endswith('.txt'):
                    txt_docs.append(os.path.join(root, file))
                # find all pdf files in current directory
                elif file.endswith('.pdf'):
                    pdf_docs.append(os.path.join(root, file))
        self.load_pdf_list(pdf_docs)
        self.load_docx_list(docx_docs)
        self.load_txt_list(txt_docs)

    # Loading/reloading and saving faiss index

    def save_index(self):
        faiss.write_index(self.index, self.index_loc)

    def save_index_and_unload(self):
        faiss.write_index(self.index, self.index_loc)
        self.index = None
        self.index_loaded = False

    def load_index(self):
        self.index = faiss.read_index(self.index_loc)

    def reload_total_index(self):
        self.index = faiss.IndexFlatIP(self.d)
        self.index.add(self.list_of_context_vectors_flattened)

    # Dealing with the the divided levels and values

    def add_to_dlv(self, filename_divided):
        """
        Adds the divided filename into the dictionary of divided levels and values(dlv)
        :param filename_divided: the categories/folders this is in
        :return:
        """
        if self.dlv is None:
            self.make_dlv_base()

        done = []
        for i in filename_divided.keys():
            if i not in self.dlv.keys():
                self.add_dlv_level(i)
                done.append(i)
            if filename_divided[i] not in self.list_dict_value_num[i].keys():
                self.list_dict_value_num[i][filename_divided[i]] = len(self.list_dict_value_num[i].keys())
        for i in self.dlv.keys():
            if i not in done:
                if i not in filename_divided.keys():
                    self.dlv[i] = np.concatenate((self.dlv[i], np.array([-1])))
                else:
                    self.dlv[i] = np.concatenate(
                        (self.dlv[i], np.array([self.list_dict_value_num[i][filename_divided[i]]])))

    def add_list_to_dlv(self, list_filename_divided):
        """
        Adds list of divided filenames into the dictionary of divided levels and values(dlv)
        :param list_filename_divided: list of the categories, I presume the first one has all the categories
        :return:
        """
        if self.dlv is None:
            self.make_dlv_base()

        done = []
        for i in list_filename_divided[0].keys():
            if i not in self.dlv.keys():
                self.add_dlv_level(i)
                done.append(i)
            if list_filename_divided[i] not in self.list_dict_value_num[i].keys():
                self.list_dict_value_num[i][list_filename_divided[i]] = len(self.list_dict_value_num[i].keys())
        for j in range(0,len(list_filename_divided)):
            for i in self.dlv.keys():
                if i not in done:
                    if i not in list_filename_divided[j].keys():
                        self.dlv[i] = np.concatenate((self.dlv[i], np.array([-1])))
                    else:
                        self.dlv[i] = np.concatenate(
                            (self.dlv[i], np.array([self.list_dict_value_num[i][list_filename_divided[j][i]]])))

    def add_to_filenames(self, filename):
        if self.list_locations is None:
            self.list_locations = []
        self.list_locations.append(filename)

    def make_dlv_base(self):
        """
        Makes the dictionary of levels and values
        :return:
        """
        self.dlv = {}
        self.list_dict_value_num = {}

    def add_dlv_level(self, i):
        """
        Adds level i ot dictionary of levels and values and sets all current existing files to 0 on that level
        :param i: the level to add
        :return:
        """
        if self.map_to_list_of_lists[-1] == 0:
            self.dlv[i] = np.zeros(1)
            self.list_dict_value_num[i] = {}
        else:
            self.dlv[i] = np.zeros(self.map_to_list_of_lists[-1]+1)-1
            self.dlv[i][-1]=0
            self.list_dict_value_num[i] = {}



def reload_index(list_of_context_vectors_flattened, location, d: int = 256):
    """

    :param list_of_context_vectors_flattened: numpy array of vectors
    :param location: string of where this faiss index is saved or should be saved
    :param d: dimension of faiss index
    :return: returns the faiss index
    """
    if os.path.isfile(location):
        index = faiss.read_index(location)
    else:
        index = faiss.IndexFlatIP(d)
        index.add(list_of_context_vectors_flattened)
        faiss.write_index(index, location)

    return index


def search_database(
        encoder: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        text: str,
        num_samples: int = 1,
        index: faiss.Index = None,
):
    """

    :param encoder: The encoder you are using
    :param tokenizer: The tokenizer for your encoder
    :param text: String that you wish to find the context for
    # :param index_of_summarised_vector: which vector is the summary vector for your encoder
    :param num_samples: number of things you wish to return
    :param index: optional index you have saved off otherwise created from your searched vectors
    :return: positions:positions from you document that match the query most closely
    """

    encoder.eval()
    data = tokenizer.encode_plus(
        text,
        max_length=512,
        return_tensors='pt')
    ids = data['input_ids'].to(encoder.device)
    mask = data['attention_mask'].to(encoder.device)
    with torch.no_grad():
        vector = encoder(ids, mask)
    _, indices = index.search(vector.numpy(), num_samples)
    return indices[0][:num_samples]


def get_context_from_index(text: str, loc_index: faiss.Index, encoder: transformers.PreTrainedModel,
                           tokenizer: transformers.PreTrainedTokenizer, num_context: int = 1):
    """
    :param text: the prompt text
    :param loc_index: index of a selection
    :param encoder: encoder like cbert
    :param tokenizer: its tokeniser
    :param num_context: how many instances of context you want
    :return: string of the context that was found
    """
    indices_returned = search_database(encoder, tokenizer, text,
                                       num_samples=num_context, index=loc_index)

    return indices_returned


def get_context_indices_from_selection(text: str, level: int, key: str, encoder: transformers.PreTrainedModel,
                                       tokenizer: transformers.PreTrainedTokenizer, dlv: dict,
                                       list_dict_value_num: list,
                                       map_to_list_of_lists: list, map_to_list_of_lists_index: list,
                                       list_of_context_vectors_flattened: np.array,
                                       num_context=1, d: int = 256, loc_index: faiss.IndexFlatIP = None,
                                       selection: int = None):
    """
    Gets the most relevant context indices from a selection of documents based on an encoder you pass
    :param text: the prompt text
    :param level: which level we want
    :param key: which key on this level we want (often either directory or file)
    :param encoder: encoder like cbert
    :param tokenizer: its tokeniser
    :param dlv: dictionary of levels and values
    example={0: array([0., 0., 0., 0., 0., 0.]),
     1: array([0., 0., 0., 0., 0., 0.]),
     2: array([0., 0., 0., 0., 0., 0.]),
     -1: array([0., 1., 2., 3., 4., 5.]),
     3: array([-1., -1., -1., -1.,  0.,  1.])}
    :param list_dict_value_num: list of dicts with values of levels and their respective numbers i.e.
    example=
    {0: {'.': 0},
     1: {'ExampleNotebooks': 0},
     2: {'ExampleFolder': 0},
     -1: {'1911.02116.pdf': 0,
      'UoS_Standard_Ts_and_Cs_for_Services_for_attaching_to_invoices_Final_Version.pdf': 1,
      '2105.00572.pdf': 2,
      'Document1.txt': 3,
      'Document2.txt': 4,
      'Document3.txt': 5},
     3: {'Subfolder2': 0, 'SubfolderOfLies': 1}}
    :param map_to_list_of_lists: array([0, 0, 0, ..., 5, 5, 5])
    :param map_to_list_of_lists_index: array([0., 1., 2., ..., 0., 1., 2.])
    :param list_of_context_vectors_flattened: numpy array of vectors
    :param num_context: how many instances of context you want
    :param d: dimension of your vectors
    :param loc_index: the faiss index with your vectors
    # :param index_of_summarised_vector: which vector is the summary vector for your encoder

    :return: context indices
    """
    if selection is None:
        selection = np.argwhere(dlv[level] == list_dict_value_num[level][key])
    selection_map_to_list_of_lists = map_to_list_of_lists[
        np.isin(map_to_list_of_lists, selection)]
    selection_map_to_list_of_lists_index = map_to_list_of_lists_index[
        np.isin(map_to_list_of_lists, selection)]
    if loc_index is None:
        selection_list_of_context_vectors_flattened = list_of_context_vectors_flattened[
            np.isin(map_to_list_of_lists, selection)]
        loc_index = faiss.IndexFlatIP(d)
        loc_index.add(selection_list_of_context_vectors_flattened)
    indices_returned = get_context_from_index(text, loc_index, encoder,
                                              tokenizer, num_context=num_context)
    if num_context == 1:
        return [int(selection_map_to_list_of_lists[int(indices_returned)]), int(
            selection_map_to_list_of_lists_index[int(indices_returned)])]
    else:
        list_of_returned_context_indices = [(int(selection_map_to_list_of_lists[indices_returned[i]]),
                                             int(selection_map_to_list_of_lists_index[indices_returned[i]])) for i in
                                            range(num_context)]
        return list_of_returned_context_indices


def get_context_from_selection(text: str, level: int, key: str, encoder: transformers.PreTrainedModel,
                               tokenizer: transformers.PreTrainedTokenizer, dlv: dict, list_dict_value_num: list,
                               map_to_list_of_lists: list,
                               map_to_list_of_lists_index: list, list_of_context_vectors_flattened: np.array,
                               list_of_lists_of_strings: list,
                               num_context=1, d: int = 256, loc_index: faiss.IndexFlatIP = None,
                               selection: int = None):
    """
    Gets the most relevant context from a selection of documents based on an encoder you pass
    :param text: the prompt text
    :param level: which level we want
    :param key: which key on this level we want (often either directory or file)
    :param encoder: encoder like cbert
    :param tokenizer: its tokeniser
    :param dlv: dictionary of levels and values
    example={0: array([0., 0., 0., 0., 0., 0.]),
     1: array([0., 0., 0., 0., 0., 0.]),
     2: array([0., 0., 0., 0., 0., 0.]),
     -1: array([0., 1., 2., 3., 4., 5.]),
     3: array([-1., -1., -1., -1.,  0.,  1.])}
    :param list_dict_value_num: list of dicts with values of levels and their respective numbers i.e.
    example=
    {0: {'.': 0},
     1: {'ExampleNotebooks': 0},
     2: {'ExampleFolder': 0},
     -1: {'1911.02116.pdf': 0,
      'UoS_Standard_Ts_and_Cs_for_Services_for_attaching_to_invoices_Final_Version.pdf': 1,
      '2105.00572.pdf': 2,
      'Document1.txt': 3,
      'Document2.txt': 4,
      'Document3.txt': 5},
     3: {'Subfolder2': 0, 'SubfolderOfLies': 1}}
    :param map_to_list_of_lists: array([0, 0, 0, ..., 5, 5, 5])
    :param map_to_list_of_lists_index: array([0., 1., 2., ..., 0., 1., 2.])
    :param list_of_context_vectors_flattened: numpy array of vectors
    :param list_of_lists_of_strings: the strings to retrieve from

    optional parameters with defaults

    :param num_context: how many instances of context you want
    :param d: dimension of your vectors
    :param loc_index: the faiss index with your vectors
    # :param index_of_summarised_vector: which vector is the summary vector for your encoder
    :param selection:



    :return: context
    """
    if selection is None:
        selection = np.argwhere(dlv[level] == list_dict_value_num[level][key])
    selection_map_to_list_of_lists = map_to_list_of_lists[
        np.isin(map_to_list_of_lists, selection)]
    selection_map_to_list_of_lists_index = map_to_list_of_lists_index[
        np.isin(map_to_list_of_lists, selection)]
    if loc_index is None:
        selection_list_of__context_vectors_flattened = list_of_context_vectors_flattened[
            np.isin(map_to_list_of_lists, selection)]
        loc_index = faiss.IndexFlatIP(d)
        loc_index.add(selection_list_of__context_vectors_flattened)
    indices_returned = get_context_from_index(text, loc_index, encoder,
                                              tokenizer, num_context=num_context)
    if num_context == 1:
        return list_of_lists_of_strings[int(selection_map_to_list_of_lists[int(indices_returned)])][int(
            selection_map_to_list_of_lists_index[int(indices_returned)])]
    else:
        list_of_returned_contexts = [
            list_of_lists_of_strings[int(selection_map_to_list_of_lists[indices_returned[i]])]
            [int(selection_map_to_list_of_lists_index[indices_returned[i]])] for i in
            range(num_context)]
        return ' '.join(list_of_returned_contexts)


def get_all_relevant_contracts(list_of_lens_summed: list, text: str, level: int,
                               encoder: transformers.PreTrainedModel,
                               tokenizer: transformers.PreTrainedTokenizer, dlv: dict,
                               list_dict_value_num: list,
                               map_to_list_of_lists: list, map_to_list_of_lists_index: list,
                               list_of_context_vectors_flattened: np.array,
                               num_context=1, d: int = 256, loc_indices: list(faiss.IndexFlatIP) = None,
                               number_of_documents: int = 2):
    """
    Gets the most relevant context from a selection of documents based on an encoder you pass
    :param list_of_lens_summed: length of all preceding documents summed as a list
    :param text: the prompt text
    :param level: which level we want
    :param key: which key on this level we want (often either directory or file)
    :param encoder: encoder like cbert
    :param tokenizer: its tokeniser
    :param dlv: dictionary of levels and values
    example={0: array([0., 0., 0., 0., 0., 0.]),
     1: array([0., 0., 0., 0., 0., 0.]),
     2: array([0., 0., 0., 0., 0., 0.]),
     -1: array([0., 1., 2., 3., 4., 5.]),
     3: array([-1., -1., -1., -1.,  0.,  1.])}
    :param list_dict_value_num: list of dicts with values of levels and their respective numbers i.e.
    example=
    {0: {'.': 0},
     1: {'ExampleNotebooks': 0},
     2: {'ExampleFolder': 0},
     -1: {'1911.02116.pdf': 0,
      'UoS_Standard_Ts_and_Cs_for_Services_for_attaching_to_invoices_Final_Version.pdf': 1,
      '2105.00572.pdf': 2,
      'Document1.txt': 3,
      'Document2.txt': 4,
      'Document3.txt': 5},
     3: {'Subfolder2': 0, 'SubfolderOfLies': 1}}
    :param map_to_list_of_lists: array([0, 0, 0, ..., 5, 5, 5])
    :param map_to_list_of_lists_index: array([0., 1., 2., ..., 0., 1., 2.])
    :param list_of_context_vectors_flattened: numpy array of vectors
    :param num_context: how many instances of context you want
    :param d: dimension of your vectors
    :param loc_index: the faiss index with your vectors
    # :param index_of_summarised_vector: which vector is the summary vector for your encoder
    :param number_of_documents: How many documents should be retrieved
    :return: list of top number_of_documents candidate string indices as 2 lists
    """
    if loc_indices==None:
        loc_indices=[None]*len(dlv[level])
    candidates = []
    flat_list = []
    vector = vectorise(text, encoder=encoder, tokenizer=tokenizer)
    for i in range(len(dlv[level])):
        candidates.append(get_context_indices_from_selection(text, 0, '', encoder, tokenizer, dlv, list_dict_value_num,
                                                             map_to_list_of_lists,
                                                             map_to_list_of_lists_index,
                                                             list_of_context_vectors_flattened,
                                                             num_context, d, loc_indices[i], i))
        if num_context == 1:
            flat_list.append(list_of_lens_summed[candidates[-1][0]] + candidates[-1][1])
        else:
            for i in range(0,num_context):
                flat_list.append(list_of_lens_summed[candidates[-1][i][0]] + candidates[-1][i][1])
    candidates_flat_list = [item for sublist in candidates for item in sublist]
    if isinstance(list_of_context_vectors_flattened, np.ndarray):
        list_of_context_vectors_flattened = torch.from_numpy(list_of_context_vectors_flattened).float()
    select_context_vectors = list_of_context_vectors_flattened[flat_list]
    dot_product = torch.flatten(torch.matmul(select_context_vectors, vector.T))
    # Compute the L2 norm of each vector in n_vectors
    norms = torch.norm(select_context_vectors, dim=1)

    # Normalize the dot product results
    normalized_output = dot_product / norms
    order_tensor = torch.argsort(normalized_output)
    if number_of_documents> len(dlv[level]):
        number_of_documents=len(dlv[level])
    values, indices = torch.topk(order_tensor, k=number_of_documents, largest=False)
    sorted_candidates = [candidates_flat_list[i] for i in indices]
    return sorted_candidates


def get_relevant_contracts_above_cutoff(list_of_lens_summed: list, text: str, level: int,
                                        encoder: transformers.PreTrainedModel,
                                        tokenizer: transformers.PreTrainedTokenizer, dlv: dict,
                                        list_dict_value_num: list,
                                        map_to_list_of_lists: list, map_to_list_of_lists_index: list,
                                        list_of_context_vectors_flattened: np.array,
                                        num_context=1, d: int = 256, loc_indices: list(faiss.IndexFlatIP) = None,
                                        cutoff: int = 0.8):
    """
    Gets the most relevant context from a selection of documents based on an encoder you pass
    :param list_of_lens_summed: length of all preceding documents summed as a list
    :param text: the prompt text
    :param level: which level we want
    :param key: which key on this level we want (often either directory or file)
    :param encoder: encoder like cbert
    :param tokenizer: its tokeniser
    :param dlv: dictionary of levels and values
    example={0: array([0., 0., 0., 0., 0., 0.]),
     1: array([0., 0., 0., 0., 0., 0.]),
     2: array([0., 0., 0., 0., 0., 0.]),
     -1: array([0., 1., 2., 3., 4., 5.]),
     3: array([-1., -1., -1., -1.,  0.,  1.])}
    :param list_dict_value_num: list of dicts with values of levels and their respective numbers i.e.
    example=
    {0: {'.': 0},
     1: {'ExampleNotebooks': 0},
     2: {'ExampleFolder': 0},
     -1: {'1911.02116.pdf': 0,
      'UoS_Standard_Ts_and_Cs_for_Services_for_attaching_to_invoices_Final_Version.pdf': 1,
      '2105.00572.pdf': 2,
      'Document1.txt': 3,
      'Document2.txt': 4,
      'Document3.txt': 5},
     3: {'Subfolder2': 0, 'SubfolderOfLies': 1}}
    :param map_to_list_of_lists: array([0, 0, 0, ..., 5, 5, 5])
    :param map_to_list_of_lists_index: array([0., 1., 2., ..., 0., 1., 2.])
    :param list_of_context_vectors_flattened: numpy array of vectors
    :param num_context: how many instances of context you want
    :param d: dimension of your vectors
    :param loc_indices: the list of faiss indices with your vectors
    # :param index_of_summarised_vector: which vector is the summary vector for your encoder
    :param cutoff: how close we want the cutoff for document relevancy
    :return: list of pruned candidate string indices
    """
    if loc_indices==None:
        loc_indices=[None]*len(dlv[level])
    candidates = []
    flat_list = []
    vector = vectorise(text, encoder=encoder, tokenizer=tokenizer)

    for i in range(len(dlv[level])):
        candidates.append(get_context_indices_from_selection(text, -1, '', encoder, tokenizer, dlv, list_dict_value_num,
                                                             map_to_list_of_lists,
                                                             map_to_list_of_lists_index,
                                                             list_of_context_vectors_flattened,
                                                             num_context, d, loc_indices[i], i))
        if num_context == 1:
            flat_list.append(list_of_lens_summed[candidates[-1][0]] + candidates[-1][1])
        else:
            for i in range(0,num_context):
                flat_list.append(list_of_lens_summed[candidates[-1][i][0]] + candidates[-1][i][1])
    candidates_flat_list = [item for sublist in candidates for item in sublist]
    if isinstance(list_of_context_vectors_flattened, np.ndarray):
        list_of_context_vectors_flattened = torch.from_numpy(list_of_context_vectors_flattened).float()
    select_context_vectors = list_of_context_vectors_flattened[flat_list]
    dot_product = torch.flatten(torch.matmul(select_context_vectors, vector.T))
    # Compute the L2 norm of each vector in n_vectors
    norms = torch.norm(select_context_vectors, dim=1)

    # Normalize the dot product results
    normalized_output = dot_product / norms
    candidate_pruning = normalized_output > cutoff
    order_tensor = torch.argsort(normalized_output)
    pruned_candidates = [candidates_flat_list[i] for i in order_tensor if candidate_pruning[i]]
    return pruned_candidates


def vectorise(text: str, encoder: transformers.PreTrainedModel, tokenizer: transformers.PreTrainedTokenizer):
    """

    :param text: the prompt text
    :param encoder: encoder like cbert
    :param tokenizer: its tokeniser
    :return: encoded vector
    """
    encoder.eval()
    data = tokenizer.encode_plus(
        text,
        max_length=512,
        return_tensors='pt')
    ids = data['input_ids'].to(encoder.device)
    mask = data['attention_mask'].to(encoder.device)
    with torch.no_grad():
        vector = encoder(ids, mask)
    return vector


def batch_vectorise(text: list, encoder: transformers.PreTrainedModel, tokenizer: transformers.PreTrainedTokenizer):
    """

    :param text: the prompt text
    :param encoder: encoder like cbert
    :param tokenizer: its tokeniser
    :return: encoded vector
    """
    encoder.eval()
    data = tokenizer.batch_encode_plus(
        text,
        max_length=512,
        return_tensors='pt', truncation=True, padding=True)
    ids = data['input_ids'].to(encoder.device)
    mask = data['attention_mask'].to(encoder.device)
    with torch.no_grad():
        vector = encoder(ids, mask)
    return vector
