def logistic_map(x, r):
    return r*x*(1-x)

def iterate_f(it, x, r):
    iterations = []
    curr_x = x
    
    for i in range(it):
        new_x = logistic_map(curr_x, r)
        curr_x = new_x
        iterations.append(curr_x)

    return iterations
