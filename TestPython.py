import torch
from operator import itemgetter
if __name__ == "__main__":
    print("test success")
    tensor_one = torch.randn(1,3)
    print(tensor_one)
    print(tensor_one.unsqueeze(2).shape)
    instance = [[i for i in range(j, j+3)] for j in range(4)]
    print(instance)
    print( min(instance, key=itemgetter(3))[1])