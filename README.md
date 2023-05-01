# VDBforGenAI

VDBforGenAI is a Python package for building vector databases of text for use in natural language processing applications.

## Usage

To use VDBforGenAI, first install the package and its dependencies:

```commandline
pip install git+https://github.com/JakubJDolezal/VDBforGenAI.git
```
Next, create an instance of the VectorDatabase class by passing in a list of strings, which represent the context you care about. Each string can contain multiple sentences.


## Minimal example
You instantiate a database and then tell it where to load
```python
from VDBforGenAI.VectorDatabase import VectorDatabase

vdb = VectorDatabase(splitting_choice="length")
vdb.load_all_in_directory('./ExampleFolder')


```
Once you have a VectorDatabase instance, you can use the get_context_from_entire_database method to retrieve the context that is most similar to a given input text.

```python
context = vdb.get_context_from_entire_database('What does parma ham go well with?')

print(context)
```
This retrieves the most similar piece of text to "What does parma ham go well with?" from your indexed directory
You can also specify which level and which directory on that level you wish to search
```python
context_selection=vdb.get_index_and_context_from_selection('Who made this?', 2, 'SubfolderOfLies')

```
The directory level and value structure is saved in 
```python
print(vdb.dlv)
```


Dependencies

VDBforGenAI has the following dependencies:
```
        "faiss-cpu",
        "transformers",
        "torch",
        "numpy","PyPDF2",'docx','python-docx
```


Contributions are welcome! If you have any suggestions or issues, please create an issue or pull request on the GitHub repository.
License

VDBforGenAI is licensed under the MIT License.

# More Usage -
## How to add new strings



## Passing an encoder and tokenizer from Hugging Face's Transformers library:


```
from transformers import AutoTokenizer, AutoModel
from VDBforGenAI import VectorDatabase

[//]: # ( Initialize the tokenizer and encoder)
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
encoder = AutoModel.from_pretrained('bert-base-uncased')

[//]: # ( Initialize the VectorDatabase)
vdb = VectorDatabase( encoder=encoder, tokenizer=tokenizer)

```
Similarly, you can pass your own encoder as a torch model if you provide a tokenizer and the 0th output is the encoding.