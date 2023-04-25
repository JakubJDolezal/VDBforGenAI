from VDBforGenAI.VectorDatabase import VectorDatabase

a = VectorDatabase()
a.load_all_in_directory('./ExampleFolder')
print(a.get_context_from_entire_database('What does parma ham go well with?'))
print(a.get_index_and_context_from_selection('Who made this?', 2, 'SubfolderOfLies'))
print(a.list_dict_value_num)
print(a.dlv)
