import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from models import FashionMNISTModelV1
from models import FashionMNISTModelV2
from models import FashionMNISTModelV3
from timeit import default_timer as timer
from functions import accuracy_fn
from functions import train_step
from functions import test_step
from functions import eval_model

def print_train_time(start: float, end: float, device: torch.device = None):
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time

device = "cuda" if torch.cuda.is_available() else "cpu"

train_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=None
)
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
    target_transform=None
)

BATCH_SIZE = 32
train_dataloader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)

#m = FashionMNISTModelV2(input_shape=784, hidden_units=10, output_shape=10).to(device)
m = FashionMNISTModelV3(input_shape=1, hidden_units=10, output_shape=10).to(device)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=m.parameters(), lr=0.1)

epochs = 3
train_time_start = timer()

for epoch in range(epochs):
    print(f"Epoch: {epoch}\n------")

    train_step(model=m, data_loader=train_dataloader, loss_fn=loss_fn, optimizer=optimizer, device=str(next(m.parameters()).device))

    test_step(model=m, data_loader=test_dataloader, loss_fn=loss_fn, device=str(next(m.parameters()).device))

train_time_end = timer()
total_train_time = print_train_time(start=train_time_start, end=train_time_end, device=str(next(m.parameters()).device))


m_results = eval_model(model=m,
                       data_loader=test_dataloader,
                       loss_fn=loss_fn,
                       device=str(next(m.parameters()).device))

print(m_results)

