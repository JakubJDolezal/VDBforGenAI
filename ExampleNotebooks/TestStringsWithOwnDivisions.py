from __future__ import annotations
import faiss
import numpy as np
import torch
import transformers as transformers
import os

from transformers import AutoTokenizer, AutoModel


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



from VDBforGenAI.VectorDatabase import VectorDatabase
from RandomGarbage.GalaChad import GalaChad

# You instantiate a database and then tell it where to load (you can decide how you wish to split the strings, I would reccomend length for now)
vdb = VectorDatabase(encoder=False, splitting_choice="paragraphs")
vdb.load_all_in_directory('./ExampleNotebooks/ExampleFolder')
# Once you have a VectorDatabase instance, you can use the get_context_from_entire_database method to retrieve the context that is most similar to a given input text.

print(vdb.get_context_from_entire_database('What does parma ham go well with?'))
# This retrieves the most similar piece of text to "What does parma ham go well with?" from your indexed directory
# You can also get the index of the document and which string in it it is
# print(vdb.get_context_indices_from_entire_database('What does parma ham go well with?'))
#
# # You can also specify which level and which directory on that level you wish to search, -1 level is always the file name level,
# # otherwise it is based on distance from where you loaded
# print(vdb.get_context_indices_from_selection('Was this made by Jakub Dolezal?', 3, 'SubfolderOfLies'))
# print(vdb.get_context_from_selection('Was this made by Jakub Dolezal?', 3, 'SubfolderOfLies'))
# The directory level and value structure is saved in
print(vdb.dlv)
name_of_model = 'GalahadEdgarRawv811'
ver = 'v40'
subver = '.3'
device = 'cpu'
model_path = '/store3/models_jakub/models_jakub/MCClauses/MCClausesVOS' + ver + subver + name_of_model + 'ep' + str(
    4)
model_dict = torch.load(model_path, map_location=device)
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
special_tokens_1 = ["<EDG>",
                    "<GEN>"]
special_tokens_2 = ['<HE>', '<ZH>', '<RU>', '<JA>', '<AR>', '<DE>', '<FR>', '<EN>', '<NL>', '<ES>', '<DA>', '<GA>',
                    '<MT>', '<HU>', '<PT>', '<BG>', '<HR>', '<ET>', '<FI>', '<SV>', '<LT>', '<IT>', '<SK>', '<LV>',
                    '<EL>', '<RO>', '<SL>',
                    '<CS>', '<PL>']
special_tokens_1 = special_tokens_1 + special_tokens_2
tokenizer.add_tokens(special_tokens_1)
temp = AutoModel.from_pretrained('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
temp.resize_token_embeddings(len(tokenizer))

model_bert = GalaChad(temp)
model_bert = model_bert
model_bert.eval()

flattened_list = [item for sublist in vdb.list_of_lists_of_strings for item in sublist]
# list_of_lists_to_flattened=[item for sublist in vdb.list_of_lists_of_strings for item in sublist]
# mapped_indices=map_indices_to_lists(list_of_lists_to_flattened)
list_of_lens = [0] + [len(vdb.list_of_lists_of_strings[i]) for i in range(len(vdb.list_of_lists_of_strings) - 1)]
list_of_lens_summed = []
for i in range(0, len(list_of_lens)):
    if i == 0:
        list_of_lens_summed.append(list_of_lens[i])
    else:
        list_of_lens_summed.append(list_of_lens_summed[i - 1] + list_of_lens[i])
list_of_context_vectors_flattened = batch_vectorise(flattened_list, encoder=model_bert, tokenizer=tokenizer)
index = reload_index(list_of_context_vectors_flattened, '/store3/models_jakub/FaissIndex')
stuff = get_context_from_selection('Was this made by Jakub Dolezal?', -1, '2105.00572.pdf', model_bert,
                                   tokenizer, vdb.dlv, vdb.list_dict_value_num,
                                   vdb.map_to_list_of_lists,
                                   vdb.map_to_list_of_lists_index, list_of_context_vectors_flattened,
                                   vdb.list_of_lists_of_strings)
stuff2 = get_relevant_contracts_above_cutoff(list_of_lens_summed, 'Was this made by Jakub Dolezal?', 0,
                                             model_bert,
                                             tokenizer, vdb.dlv, vdb.list_dict_value_num,
                                             vdb.map_to_list_of_lists,
                                             vdb.map_to_list_of_lists_index, list_of_context_vectors_flattened)
tuff2 = get_relevant_contracts_above_cutoff(list_of_lens_summed, 'Was this made by Jakub Dolezal?', 0,
                                             model_bert,
                                             tokenizer, vdb.dlv, vdb.list_dict_value_num,
                                             vdb.map_to_list_of_lists,
                                             vdb.map_to_list_of_lists_index, list_of_context_vectors_flattened, num_context=5)
tuff2 = get_relevant_contracts_above_cutoff(list_of_lens_summed, 'Was this made by Jakub Dolezal?', 0,
                                             model_bert,
                                             tokenizer, vdb.dlv, vdb.list_dict_value_num,
                                             vdb.map_to_list_of_lists,
                                             vdb.map_to_list_of_lists_index, list_of_context_vectors_flattened, num_context=1)
stuff3 = get_all_relevant_contracts(list_of_lens_summed, 'Was this made by Jakub Dolezal?', 0,
                                             model_bert,
                                             tokenizer, vdb.dlv, vdb.list_dict_value_num,
                                             vdb.map_to_list_of_lists,
                                             vdb.map_to_list_of_lists_index, list_of_context_vectors_flattened,)
print(stuff)
