import numba as nb


def flatten_dict(nested_dict, parent=""):
    """
    Flattens a nested dict
    """
    flat_dict = {}
    if isinstance(nested_dict, list):
        return nested_dict
    for k, v in nested_dict.items():
        if isinstance(v, dict):
            flat_dict.update(flatten_dict(v, parent + k + "_"))
        elif isinstance(v, list) and isinstance(v[0], list):
            for i in range(len(v)):
                flat_dict[parent + k + "_" + str(i)] = v[i]
        else:
            flat_dict[parent + k] = v
    return flat_dict


@nb.jit(nopython=True, cache=True)
def fill_dict_with_list(list_values, dictionary, index=0):
    """
    Fills a nested dict with a list
    """
    for k, v in dictionary.items():
        if isinstance(v, dict):
            fill_dict_with_list(list_values, v, index)
        elif isinstance(v, list):
            for i in range(len(v)):
                if isinstance(v[i], float):
                    v[i] = list_values[index]
                    index += 1
        elif isinstance(v, float):
            dictionary[k] = list_values[index]
            index += 1
    return dictionary
