import torch

def rowswap(M, i, j):
    M = M.clone()
    tmp = M[i].clone()
    M[i] = M[j]
    M[j] = tmp
    return M

def rowscale(M, i, factor):
    M = M.clone()
    M[i] = M[i] * factor
    return M

def rowreplacement(M, i, j, factor_i, factor_j):
    M = M.clone()
    M[j] = factor_i * M[i] + factor_j * M[j]
    return M

def rref(M):
    M = M.clone().float()
    rows, cols = M.shape
    pivot_row = 0
    
    for col in range(cols):
        pivot = None
        for r in range(pivot_row, rows):
            if abs(M[r, col]) > 1e-10:
                pivot = r
                break
        
        if pivot is None:
            continue
        
        if pivot != pivot_row:
            M = rowswap(M, pivot, pivot_row)
        
        piv_val = M[pivot_row, col]
        M = rowscale(M, pivot_row, 1.0 / piv_val)
        
        for r in range(rows):
            if r != pivot_row and abs(M[r, col]) > 1e-10:
                factor = -M[r, col]
                M = rowreplacement(M, pivot_row, r, factor, 1.0)
        
        pivot_row += 1
        if pivot_row >= rows:
            break
    
    return M

if __name__ == "__main__":
    test = torch.tensor([
        [1.0, 3.0, 0.0, 0.0, 3.0],
        [0.0, 0.0, 1.0, 0.0, 9.0],
        [0.0, 0.0, 0.0, 1.0, -4.0]
    ])
    
    print("Original:")
    print(test)
    
    step1 = rowswap(test, 0, 1)
    print("\nAfter R1<->R2:")
    print(step1)
    
    step2 = rowscale(step1, 0, 1.0/3.0)
    print("\nAfter (1/3)R1:")
    print(step2)
    
    step3 = rowreplacement(step2, 0, 2, -3.0, 1.0)
    print("\nAfter R3 = -3*R1 + R3:")
    print(step3)
