import itertools
import json
import numpy as np

from code.graph_generation.entropy.ste import _encode_subsample

def generate_dict():
    possibl_permutations = list(set(itertools.product([1, 2, 3, 4], repeat=3)))
    ste_dict = {}
    for x in possibl_permutations:
        ste_dict[np.array2string(np.array(x), precision=1, separator=',',
                        suppress_small=True)] = _encode_subsample(x).tolist()

    with open("ste_dict.json", "w") as outfile: 
        json.dump(ste_dict, outfile)

def convert_to_numpy(item):
    if isinstance(item, list):
        return np.array(item)
    elif isinstance(item, str):
        # Check if the string is a tuple
        try:
            return np.array(eval(item))
        except (SyntaxError, NameError):
            return item
    elif isinstance(item, dict):
        return {key: convert_to_numpy(value) for key, value in item.items()}
    else:
        return item

def load_dict_as_vectorized():
    with open('ste_dict.json') as ste_dict:
        data = json.load(ste_dict)
    converted_data = convert_to_numpy(data)
    vectorized_lookup = np.vectorize(lambda x: converted_data[x], otypes=[list])

    return vectorized_lookup


# l=1
# m=3
# x = np.array([1,1,2,3,4,4,3,2])
# Y = np.empty((m, len(x) - (m - 1) * l), dtype=int)
# for i in range(m):
#     Y[i] = x[i * l:i * l + Y.shape[1]]
# subsamples = Y.T

# string_array = np.array([np.array2string(np.array(xx), precision=1, separator=',',
#                       suppress_small=True) for xx in subsamples])
# # Print the result
# print(string_array, type(string_array))

# Vectorized lookup function
# vectorized_lookup = np.vectorize(lambda x: converted_data[x], otypes=[list])
# # Apply the vectorized lookup
# result_array = vectorized_lookup(string_array)
# print(result_array)