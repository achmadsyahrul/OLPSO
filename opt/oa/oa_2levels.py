import numpy as np

def construct_OA(D):
    # Step 1
    M = 2 ** int(np.ceil(np.log2(D + 1)))
    N = M - 1
    u = int(np.log2(M))

    # Step 2: Set elements in basic columns
    OA = np.zeros((M, N), dtype=int)
    for a in range(1, M + 1):
        for k in range(1, u + 1):
            b = 2 ** (k - 1)
            OA[a - 1][b - 1] = (a - 1) // (2 ** (u - k)) % 2

    # Step 3: Set elements in other columns
    for a in range(1, M + 1):
        for k in range(2, u + 1):
            for s in range(1, 2 ** (k - 1)):
                b = 2 ** (k - 1)
                OA[a - 1][(b + s) - 1] = (OA[a - 1][s - 1] + OA[a - 1][b - 1]) % 2

    # Step 4: Transform level values
    for a in range(M):
        for b in range(N):
            if OA[a][b] == 0:
                OA[a][b] = 1
            else:
                OA[a][b] = 2

    return OA