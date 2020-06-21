import numpy as np
import torch
import heapq
# time_dim = 100
# a = torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim)).float().unsqueeze(dim=0)
# b = a.repeat(10,1)
# debug = 0

heap = []
a = [3,4,5]
b = [4,5,6]
c = [7,8,9]
d = [1,4,5]

for l in a+b+c+d:
    heapq.heappush(heap, l)

m = []
while heap:
    head = heapq.heappop(heap)
    m.append(head)

debug = 0