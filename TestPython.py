import torch

if __name__ == "__main__":
    print("test success")
    tensor_one = torch.randn(1,3)
    print(tensor_one)
    print(tensor_one.unsqueeze(2).shape)