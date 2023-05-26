# Multinomial coefficient

from scipy.special import comb

def multinomial(params):
    if len(params) == 1:
        return 1
    return comb(sum(params), params[-1],exact=True) * multinomial(params[:-1])