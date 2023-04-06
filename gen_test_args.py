from functools import reduce
from math import ceil
m = 768
n = [167, 183, 177, 181, 153, 139, 156, 173, 163, 150, 204, 184, 168, 156, 168, 148]
k = 4608
print(' '.join(f'-m {m} -n {nn} -k {k}' for nn in n))

num_gemm = len(n) // 2
n0, n1 = n[:num_gemm], n[num_gemm:]
r0, r1 = len(n0) / len(n), len(n1) / len(n) 
num_cu = 104
MT0, MT1 = 256, 96
numWg = ceil(sum(n[:len(n)//2]) * m / (MT0 * MT1))
bw = 13 * 4 * 4 * 64 * 4 * numWg
print(bw)
#print(f'n0: {sum(n0)}, n1: {sum(n1)}, {r0}:{r1}, {int(r0 * num_cu)}:{int(r1 * num_cu)}')