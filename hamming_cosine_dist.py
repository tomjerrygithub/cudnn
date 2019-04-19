def hamming_distance(a, b):
    cnt=0
    for i ,j in zip(a, b):
        if i!=j:
            cnt+=1
    return cnt
    
    
import numpy as np
def chi2_distance(histA, histB, eps = 1e-10):
    d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
    for (a, b) in zip(histA, histB)])
    return d
