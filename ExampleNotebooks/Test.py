from VDBforGenAI.VectorDatabase import VectorDatabase
a=VectorDatabase()
a.load_all_in_directory('./ExampleFolder')
a.list_of_lists_of_strings