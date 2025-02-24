def replace_values(lst):
    return [1 if x in {1, 2, 3} else 2 if x in {4, 5, 6} else x for x in lst]

# Example usage
my_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
new_list = replace_values(my_list)
print(new_list)  # Output: [0, 1, 1, 1, 2, 2, 2, 7, 8]
