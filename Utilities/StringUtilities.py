def split_string_to_dict(s):
    """
    Splits a string into a dictionary where the keys are indices and the values are the corresponding substrings
    between '/' characters in the input string s. The -1 index is always with the key -1

    Args:
    - s (str): A string that will be split into substrings separated by '/' characters.

    Returns:
    - result_dict (dict): A dictionary where each key is an integer representing the index of the substring in the
    input string, and the corresponding value is the substring itself.
    """
    split_list = s.split('/')  # Split the input string into a list of substrings separated by '/' characters
    result_dict = {}  # Initialize an empty dictionary to store the resulting key-value pairs
    for i in range(len(split_list) - 1):  # Iterate over the indices of the substrings in the split list
        result_dict[i] = split_list[i]  # Add a new key-value pair to the dictionary for each substring
    result_dict[-1] = split_list[
        -1]  # Add a key-value pair for the last substring, which doesn't have a corresponding index
    return result_dict  # Return the resulting dictionary
