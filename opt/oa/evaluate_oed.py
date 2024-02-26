import numpy as np

def enumerate_combination(pi, pn, row):
    combination = np.zeros_like(pi)
    row = row[:len(combination)]
    for i, value in enumerate(row):
        if value == 1:
            combination[i] = pi[i]
        else:
            combination[i] = pn[i]
    return combination


def evaluate_oed(func, OA, pi, pn):
    min_val = float('inf')
    min_combination = None
    values = []
    history = []  # Store combination and value history
    # Loop through each row of OA
    for row in OA:
        # Initialize combination array
        combination = enumerate_combination(pi, pn, row)

        # Calculate the value of the combination
        val = func(combination)
        values.append(val)

        # Update minimum value and combination if necessary
        if val < min_val:
            min_val = val
            min_combination = combination

        # Append combination and value to history
        history.append((combination, val))
    return min_val, min_combination, history, values

def evaluate_fa(values, func, D, pi, pn, min_val, min_combination):
    factor_analysis = np.zeros([2,D])
    fa_min_combination = [0] * D 
    # Compute factor_analysis values
    for i in range(2):
        for j in range(D):
            # Compute factor_analysis values based on the given formulae
            if i == 0:
                factor_analysis[i][j] = (values[i] + values[j+1]) / 2
            elif i == 1 and j == 0:
                factor_analysis[i][j] = (values[2] + values[3]) / 2
            elif i == 1 and j == 1:
                factor_analysis[i][j] = (values[1] + values[3]) / 2
            elif i == 1 and j == 2:
                factor_analysis[i][j] = (values[1] + values[2]) / 2
            if factor_analysis[0][j] < factor_analysis[i][j]:
                fa_min_combination[j] = 1
            else:
                fa_min_combination[j] = 2
    fa_min_combination = enumerate_combination(pi, pn, np.array(fa_min_combination))
    if np.array_equal(min_combination, fa_min_combination):
        fa_min_value = min_val
    else:
        fa_min_value = func(fa_min_combination)
    return fa_min_value, fa_min_combination