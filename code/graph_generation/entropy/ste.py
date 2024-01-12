import numpy as np

# possible_combinations = {
#     "1 1 1" : [1,1,1],
#     "2 2 2" : [1,1,1],
#     "3 3 3" : [1,1,1],
#     "4 4 4" : [1,1,1],
#     "1 1 2" : [1,1,2],
#     "1 1 3" : [1,1,2],
#     "1 1 4" : [1,1,2],
#     "1 2 1" : [1,2,1],
#     "1 3 1" : [1,2,1],
#     "1 4 1" : [1,2,1],
#     "1 2 2" : [0,1,1],
#     "1 3 3" : [0,1,1],
#     "1 4 4" : [0,1,1],
#     "1 2 3" : [0,1,2],
#     "1 2 4" : [0,1,2],
#     "1 3 2" : [0,2,1],
#     "1 3 4" : [0,1,2],
#     "1 4 2" : [0,2,1],
#     "1 4 3" : [0,2,1],
#     "2 1 1" : [1,0,0],
#     "2 3 3" : [1,2,2],
#     "2 4 4" : [1,2,2],
#     "2 1 2" : [1,0,1],
#     "2 1 3" : [1,0,2],
#     "2 1 4" : [1,0,2],
#     "2 2 "
# }

def _get_symbol_sequence(x,l,m):
    """ Convert a time series to a sequence of symbols

    args:
        :x (list or np.array) -- univariate time series
        :l (int) -- the time delay
        :m (int) -- the order (or symbol size)

    returns:
        :symbol sequence for time series x (numpy 2d array)
    """
    
    Y = np.empty((m, len(x) - (m - 1) * l))
    for i in range(m):
        Y[i] = x[i * l:i * l + Y.shape[1]]
    subsamples = Y.T

    return np.apply_along_axis(_encode_subsample, axis=1, arr=subsamples)

def _encode_subsample(x):
    if len(set(x)) ==1:
        return np.array([1,1,1]) 
    if _has_duplicate(x):
        return _encode_duplicated_row(x)
    return np.argsort(x, kind="quicksort")


def _has_duplicate(arr):
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if arr[i] == arr[j]:
                return True
    return False

def _encode_duplicated_row(row):
    if row[0] == row[1]:
        if row[2] > row[0]:
            return np.array([1, 1, 2])
        else:
            return np.array([1, 1, 0])

    elif row[1] == row[2]:
        if row[0] > row[1]:
            return np.array([2, 1, 1])
        else:
            return np.array([0, 1, 1])

    elif row[0] == row[2]:
        if row[1] > row[0]:
            return np.array([1, 2, 1])
        else:
            return np.array([1, 0, 1])

        
def incr_counts(key,d):
    """ Helper function for creating joint distribution over symbols.
    Joint distribution is built in a dictionary where d[key] gives the
    frequency of key.
    """
    if key in d:
        d[key] += 1
    else:
        d[key] = 1

def normalize(d):
    """ Helper function for creating joint distribution over symbols.
    Converts dictionary of raw counts to dictionary of probabilities in [0,1].
    """
    s=sum(d.values())        
    for key in d:
        d[key] /= s

def symbolic_TE(ts1,ts2,l,m):
    """
    Calculate Symbolic Transfer Entropy between two univariate time series

    args:
        :ts1 (list or np.array) -- univariate time series
        :ts2 (list or np.array) -- univariate time series
        :l (int) -- the time delay
        :m (int) -- the order (or symbol size) ... e.g m=3 gives symbols such as [0,1,2]

    returns:
        :symbolic transfer entropy from ts1 to ts2 (float)
    
    notes:
        :x_sym --  the symbol sequence obtained from ts1 ... e.g. [[0,1,2],[1,2,0],...]
        :x_sym_to_perm -- maps each symbol in x_sym to a unique int ... e.g. [0,1,2] --> 1
        :p_xyy1 -- gives the joint distribution p(x_i= , y_i= , y_{i+1}= ) stored in a dictionary
            where a key is a string ... e.g. key="1,5,2" has value p(x_i=1, y_i=5, y_{i+1}=2)
        :p_xy, p_yy1, p_y -- respective marginal distributions
    """
    
    x_sym = _get_symbol_sequence(ts1,l,m)
    y_sym = _get_symbol_sequence(ts2,l,m)
    # print(x_sym)
    # print(y_sym)

    hashmult = np.power(m, np.arange(m))
    hashval_x = (np.multiply(x_sym, hashmult)).sum(1)
    hashval_y = (np.multiply(y_sym, hashmult)).sum(1)
    
    x_sym_to_perm = hashval_x
    y_sym_to_perm = hashval_y
    # print(x_sym_to_perm)
    # print(y_sym_to_perm)
    
    p_xyy1 = {}
    p_xy = {}
    p_yy1 = {}
    p_y = {}
    for i in range(len(y_sym_to_perm)-1):
        xyy1 = str(x_sym_to_perm[i]) + "," + str(y_sym_to_perm[i]) + "," + str(y_sym_to_perm[i+1])
        xy = str(x_sym_to_perm[i]) + "," + str(y_sym_to_perm[i])
        yy1 = str(y_sym_to_perm[i]) + "," + str(y_sym_to_perm[i+1])
        y = str(y_sym_to_perm[i])
        
        incr_counts(xyy1,p_xyy1)
        incr_counts(xy,p_xy)
        incr_counts(yy1,p_yy1)
        incr_counts(y,p_y)
      
    normalize(p_xyy1)
    normalize(p_xy)
    normalize(p_yy1)
    normalize(p_y)
    
    #conditional distributions
    p_y1_given_xy = p_xyy1.copy()
    for xyy1 in p_y1_given_xy:
        xy = xyy1.split(",")[0] + "," + xyy1.split(",")[1]
        p_y1_given_xy[xyy1] /= p_xy[xy]
        
    p_y1_given_y = p_yy1.copy()
    for yy1 in p_y1_given_y:
        y = yy1.split(",")[0]
        p_y1_given_y[yy1] /= p_y[y]
        
    #entropy calculation
    final_sum = 0
    for xyy1 in p_xyy1:
        yy1 = xyy1.split(",")[1] + "," +  xyy1.split(",")[2] 
        if xyy1 in p_y1_given_xy and yy1 in p_y1_given_y:
            if float('-inf') < float(np.log2(p_y1_given_xy[xyy1]/p_y1_given_y[yy1])) < float('inf'):
                # print(p_xyy1[xyy1], p_y1_given_xy[xyy1], p_y1_given_y[yy1], np.log2(p_y1_given_xy[xyy1]/p_y1_given_y[yy1]))
                final_sum += p_xyy1[xyy1]*np.log2(p_y1_given_xy[xyy1]/p_y1_given_y[yy1])
    return final_sum