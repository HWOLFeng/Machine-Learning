import torch
import numpy as np

# arrange type
np_data = np.arange(4).reshape((2, 2))
# tensor type
torch_data = torch.from_numpy(np_data)
tensor2array = torch_data.numpy()

# data use function
# print(
#     '\nnumpy: ', np_data,
#     '\ntorch: ', torch_data,
#     '\nnp sin: ', np.sin(np_data),
#     '\ntorch sin: ', torch.sin(torch_data.float()),
#     '\ntorch matrix multi: ', torch.mm(torch_data.float(), torch_data.float())
# )


# BP
# should use float
tensor = torch.FloatTensor([[1, 2], [3, 4]])
tensor = torch.tensor(tensor, requires_grad=True)
t_out = torch.mean(tensor * tensor)
# calculate gradient: d(t_out)/d(tensor)
t_out.backward()
# backward result t_out will influence the input tensor
# so we print the gradient of tensor
print(tensor.grad)
