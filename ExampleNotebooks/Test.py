from VDBforGenAI.VectorDatabase import VectorDatabase
# You instantiate a database and then tell it where to load (you can decide how you wish to split the strings, I would reccomend length for now)
vdb = VectorDatabase(encoder=False,splitting_choice="length")
vdb.load_all_in_directory('./ExampleFolder')
# Once you have a VectorDatabase instance, you can use the get_context_from_entire_database method to retrieve the context that is most similar to a given input text.

print(vdb.get_context_from_entire_database('What does parma ham go well with?'))
# This retrieves the most similar piece of text to "What does parma ham go well with?" from your indexed directory
# You can also get the index of the document and which string in it it is
print(vdb.get_context_indices_from_entire_database('What does parma ham go well with?'))

# You can also specify which level and which directory on that level you wish to search, -1 level is always the file name level,
# otherwise it is based on distance from where you loaded
print(vdb.get_index_and_context_from_selection('Who made this?', 2, 'SubfolderOfLies'))
# The directory level and value structure is saved in
print(vdb.dlv)

