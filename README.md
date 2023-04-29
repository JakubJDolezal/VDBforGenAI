# VDBforGenAI

VDBforGenAI is a Python package for building vector databases of text for use in natural language processing applications.

## Usage

To use VDBforGenAI, first install the package and its dependencies:

```
pip install git+https://github.com/JakubJDolezal/VDBforGenAI.git
```
Next, create an instance of the VectorDatabase class by passing in a list of strings, which represent the context you care about. Each string can contain multiple sentences.


## Minimal example
```
from VDBforGenAI import VectorDatabase

list_of_strings = [
    "I am a sentence. This is another sentence in the same string.",
    "This is a new string with multiple sentences. The quick brown fox jumps over the lazy dog. The end."
]

vdb = VectorDatabase(list_of_strings)
```
Once you have a VectorDatabase instance, you can use the get_context_from_entire_database method to retrieve the context that is most similar to a given input text.

```
context = vdb.get_context_from_entire_database("brown fox jumps")

print(context)
```

Output: " The quick brown fox jumps over the lazy dog."

Dependencies

VDBforGenAI has the following dependencies:
```
    Faiss
    Transformers
    Torch
    Numpy
```


Contributions are welcome! If you have any suggestions or issues, please create an issue or pull request on the GitHub repository.
License

VDBforGenAI is licensed under the MIT License.

# More Usage -
## How to add new strings



## Create a VectorDatabase object with an initial list of strings
```
from VDBforGenAI import VectorDatabase
initial_strings = [
    "This is the first sentence. This is the second sentence.",
    "This is another string with multiple sentences. Here's another sentence.",
    "And here's yet another string with multiple sentences. This one has three sentences!"
]
vdb = VectorDatabase(initial_strings)
```
Add a new string with multiple sentences using the add_string_to_context method
```
new_string = "This is a new string with three sentences. Here's the second sentence. And here's the third!"
vdb.add_string_to_context(new_string)
```
Add a new list of strings with multiple sentences using the add_list_of_strings_to_context method
```
new_list = [
    "This is a new string with two sentences. Here's the second sentence.",
    "This is another new string with multiple sentences. Here's the second sentence. And here's the third!"
]
vdb.add_list_of_strings_to_context(new_list)
```
## Passing an encoder and tokenizer from Hugging Face's Transformers library:


```
from transformers import AutoTokenizer, AutoModel
from VDBforGenAI import VectorDatabase

# Initialize the tokenizer and encoder
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
encoder = AutoModel.from_pretrained('bert-base-uncased')

# Initialize the VectorDatabase
vdb = VectorDatabase(['some text'], encoder=encoder, tokenizer=tokenizer)

```
Similarly, you can pass your own encoder as a torch model.
