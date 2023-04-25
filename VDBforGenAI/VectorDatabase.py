from __future__ import annotations
import faiss
import numpy
from transformers import AutoTokenizer, AutoModel
from typing import Union

from VDBforGenAI.Utilities.StringUtilities import split_string_to_dict
from VDBforGenAI.VectorisationAndIndexCreation import SearchFunctions
import transformers as transformers
import re
import numpy as np
import os


class VectorDatabase:
    def __init__(self,
                 encoder: Union[str, transformers.PreTrainedModel] = None,
                 tokenizer: Union[str, transformers.PreTrainedTokenizer] = None,
                 batch_size: int = 128,
                 splitting_choice: str = "paragraphs",
                 ):
        """

        :param encoder: Transformer model from huggingface in torch, defaults to XLMRobertaModel, can be given as
        model or string location or huggingface location :param tokenizer: Tokenizer of the above model, defaults to
        XLMRobertaModelTokeniser, can be a different location than the model :param batch_size: Batch size for
        encoding :param splitting_choice: What is the size of the context you wish to consider. Options are
        "paragraphs" and "sentence" :param preload_index: Whether you want to preload the index and keep it in memory
        """
        self.index = None
        self.list_of_context_vectors_flattened = None
        self.map_to_list_of_lists = None
        self.map_to_list_of_lists_index = None
        self.list_of_lists_of_strings = None
        self.list_dict_value_num = None
        # This instantiates the dictionary holding the levels and their possible values (usually based on folder structure of import)
        self.dlv = None
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
            self.encoder = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
            self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        self.d = self.encoder.config.hidden_size

        self.batch_size = batch_size
        self.splitting_choice = splitting_choice
        self.index_loaded = False

    def reload_total_index(self):
        self.index = faiss.IndexFlatIP(self.d)
        self.index.add(self.list_of_context_vectors_flattened)

    def get_context_from_entire_database(self, text, num_context=1):
        """

        :param text: the prompt text
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
            return self.list_of_lists_of_strings[int(self.map_to_list_of_lists[int(indices_returned)])][int(
                self.map_to_list_of_lists_index[int(indices_returned)])]
        else:
            list_of_returned_contexts = [
                self.list_of_lists_of_strings[int(self.map_to_list_of_lists[indices_returned[i]])][
                    int(self.map_to_list_of_lists_index[indices_returned[i]])] for i in
                range(num_context)]
            return ' '.join(list_of_returned_contexts)

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
                                                           num_samples=num_context, index=loc_index)
        if num_context == 1:
            return self.list_of_lists_of_strings[int(selection_map_to_list_of_lists[int(indices_returned)])][int(
                selection_map_to_list_of_lists_index[int(indices_returned)])]
        else:
            list_of_returned_contexts = [
                self.list_of_lists_of_strings[int(selection_map_to_list_of_lists[indices_returned[i]])][
                    int(selection_map_to_list_of_lists_index[indices_returned[i]])] for i in
                range(num_context)]
            return ' '.join(list_of_returned_contexts)

    def get_index_and_context_from_selection(self, text: str, level: int, key: int, num_context=1):
        """

        :param text: the prompt text
        :param level: which level we
        :param key:
        :param num_context: how many instances of context you want
        :return: string of context
        """
        selection = self.dlv[level] == self.list_dict_value_num[level][key]
        selection_map_to_list_of_lists = self.map_to_list_of_lists[np.isin(self.map_to_list_of_lists,np.argwhere(selection))]
        selection_map_to_list_of_lists_index = self.map_to_list_of_lists_index[np.isin(self.map_to_list_of_lists,np.argwhere(selection))]
        selection_list_of__context_vectors_flattened = self.list_of_context_vectors_flattened[np.isin(self.map_to_list_of_lists,np.argwhere(selection))]
        loc_index = faiss.IndexFlatIP(self.d)
        loc_index.add(selection_list_of__context_vectors_flattened)
        return self.get_context_from_index(text, loc_index, selection_map_to_list_of_lists,
                                           selection_map_to_list_of_lists_index, num_context=num_context)

    def add_string_to_context(self, string, preload_index=None, dlv_handled=False):
        if preload_index is None:
            preload_index = self.index_loaded
        if self.list_of_lists_of_strings is None:
            self.initial_string_addition([string])
        else:
            previous_length = len(self.list_of_lists_of_strings)
            self.list_of_lists_of_strings.extend(self.split_string_into_list_of_strings([string]))
            self.map_to_list_of_lists = np.concatenate(
                [self.map_to_list_of_lists, np.repeat(previous_length, len(self.list_of_lists_of_strings[-1]))])
            self.map_to_list_of_lists_index = np.concatenate(
                [self.map_to_list_of_lists_index, np.linspace(0, len(self.list_of_lists_of_strings[-1]),
                                                              num=len(self.list_of_lists_of_strings[-1]), endpoint=False)])
            vector_list_of_string = SearchFunctions.vectorise_to_numpy(self.encoder, self.tokenizer,
                                                                       self.list_of_lists_of_strings[-1],
                                                                       batch_size=self.batch_size)
            self.list_of_context_vectors_flattened = np.concatenate([self.list_of_context_vectors_flattened,
                                                                     vector_list_of_string], axis=0)
        if preload_index:
            self.reload_total_index()
        else:
            self.index_loaded = False
        if not dlv_handled:
            self.add_to_dlv({0: 'String' + str(previous_length)})

    def add_list_of_strings_to_context(self, new_list_of_strings, preload_index=None, dlv_handled=False):
        if preload_index:
            preload_index = self.index_loaded
        previous_length = len(self.list_of_lists_of_strings)

        if self.list_of_lists_of_strings is None:
            self.initial_string_addition([new_list_of_strings])
        else:
            new_list_of_list_of_strings = self.split_string_into_list_of_strings(new_list_of_strings)
            list_list_of_vectors = [
                SearchFunctions.vectorise_to_numpy(self.encoder, self.tokenizer, new_list_of_list_of_strings[i],
                                                   batch_size=self.batch_size) for i in
                range(len(self.list_of_lists_of_strings))]
            lengths = [len(lst) for lst in list_list_of_vectors]
            # create an array of indices indicating which original list each element in the flattened list corresponds to
            new_map_to_list_of_lists = np.concatenate([np.repeat(i, l) for i, l in enumerate(lengths)])
            new_map_to_list_of_lists_index = np.concatenate(
                [np.linspace(0, l, num=l, endpoint=False) for i, l in enumerate(lengths)])
            new_list_of_context_vectors_flattened = np.concatenate(list_list_of_vectors, axis=0)

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
                self.add_to_dlv({0: 'String' + str(previous_length + 1)})

    def initial_string_addition(self, list_of_strings):
        self.list_of_lists_of_strings = self.split_string_into_list_of_strings(list_of_strings)
        list_list_of_vectors = [
            SearchFunctions.vectorise_to_numpy(self.encoder, self.tokenizer, self.list_of_lists_of_strings[i],
                                               batch_size=self.batch_size) for i in
            range(len(self.list_of_lists_of_strings))]
        lengths = [len(lst) for lst in list_list_of_vectors]
        # create an array of indices indicating which original list each element in the flattened list corresponds to
        self.map_to_list_of_lists = np.concatenate([np.repeat(i, l) for i, l in enumerate(lengths)])
        self.map_to_list_of_lists_index = np.concatenate(
            [np.linspace(0, l, num=l, endpoint=False) for i, l in enumerate(lengths)])
        self.list_of_context_vectors_flattened = np.concatenate(list_list_of_vectors, axis=0)

    def split_string_into_list_of_strings(self, input_list):
        if self.splitting_choice == "sentences":
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
        elif self.splitting_choice == "paragraphs":
            paragraphs_list = []
            for string in input_list:
                paragraphs = string.split('\n')  # Split the string into paragraphs using newline as delimiter
                # Remove empty strings and add the list of paragraphs to the result
                paragraphs_list.append([paragraph.strip() for paragraph in paragraphs if paragraph.strip()])
            # Remove any empty sub-lists from the result list
            paragraphs_list = [x for x in paragraphs_list if x]
            return paragraphs_list
        else:
            # Return the input list as is but make into list of lists of single strings for consistency, i.e. whole documents
            return [[input_list[i]] for i in range(0, len(input_list))]

    def load_pdf(self, filename, divide_by_filepath=None):
        pdf_string = VectorDatabase.Utilities.Loading.load_pdf(filename)
        if divide_by_filepath:
            filename_divided = split_string_to_dict(filename)
            self.add_to_dlv(filename_divided)
            self.add_string_to_context(pdf_string , dlv_handled=True)
        else:
            self.add_string_to_context(pdf_string)

    def load_docx(self, filename, divide_by_filepath=True):
        word_string = VectorDatabase.Utilities.Loading.load_docx(filename)
        if divide_by_filepath:
            filename_divided = split_string_to_dict(filename)
            self.add_to_dlv(filename_divided)
            self.add_string_to_context(word_string , dlv_handled=True)
        else:
            self.add_string_to_context(word_string)

    def load_txt(self, filename, divide_by_filepath=True):
        with open(filename, 'r') as f:
            # read contents of file as a string
            txt_string = f.read()
        if divide_by_filepath:
            filename_divided = split_string_to_dict(filename)
            self.add_string_to_context(txt_string, dlv_handled=True)
            self.add_to_dlv(filename_divided)
        else:
            self.add_string_to_context(txt_string)

    def load_string_list_with_divisions(self, string_list, divisions):
        for i in range(0, len(string_list)):
            self.add_to_dlv(divisions[i])
            self.add_string_to_context(string_list[i])

    def load_pdf_list(self, list_of_filenames, divide_by_filepath=True):
        for item in list_of_filenames:
            self.load_pdf(item, divide_by_filepath)

    def load_docx_list(self, list_of_filenames, divide_by_filepath=True):
        for item in list_of_filenames:
            self.load_docx(item, divide_by_filepath)

    def load_txt_list(self, list_of_filenames, divide_by_filepath=True):
        for item in list_of_filenames:
            self.load_txt(item, divide_by_filepath)

    def load_all_in_directory(self,directory):
        docx_docs = []
        doc_docs = []
        txt_docs = []
        pdf_docs = []

        # loop through all files and subdirectories
        for root, dirs, files in os.walk(directory):
            # find all docx/doc files in current directory
            for file in files:
                if file.endswith('.docx'):
                    docx_docs.append(os.path.join(root, file))
                elif file.endswith('.doc'):
                    doc_docs.append(os.path.join(root, file))
                # find all txt files in current directory
                elif file.endswith('.txt'):
                    txt_docs.append(os.path.join(root, file))
                # find all pdf files in current directory
                elif file.endswith('.pdf'):
                    pdf_docs.append(os.path.join(root, file))
        self.load_pdf_list(docx_docs)
        self.load_docx_list(pdf_docs)
        self.load_txt_list(txt_docs)


    def add_to_dlv(self, filename_divided):
        if self.dlv is None:
            self.make_dlv_base()
        done=[]
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
                    self.dlv[i] = np.concatenate((self.dlv[i], np.array([self.list_dict_value_num[i][filename_divided[i]]])))

    def make_dlv_base(self):
        self.dlv = {}
        self.list_dict_value_num = {}

    def add_dlv_level(self, i):
        if len(self.list_of_lists_of_strings)==1:
            self.dlv[i] = np.zeros(1)
            self.list_dict_value_num[i] = {}
        else:
            self.dlv[i] = np.zeros(len(self.list_of_lists_of_strings))-1
            self.list_dict_value_num[i] = {}
