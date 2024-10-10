import torch
from sklearn.datasets import make_circles 
from sklearn.model_selection import train_test_split
from models import CircleModelV1
from models import CircleModelV2

device = "cuda" if torch.cuda.is_available() else "cpu"

n_samples = 100
x, y = make_circles(n_samples, noise=0.03, random_state=42)
x = torch.from_numpy(x).type(torch.float)
y = torch.from_numpy(y).type(torch.float)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train, x_test, y_train, y_test = x_train.to(device), x_test.to(device), y_train.to(device), y_test.to(device)

m = CircleModelV2().to(device)

loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=m.parameters(), lr=0.1)

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


epochs = 4000

for epoch in range(epochs):
    m.train()
    y_logits = m(x_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))

    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_true=y_train, y_pred=y_pred)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    m.eval()
    with torch.inference_mode():
        test_logits = m(x_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))

        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true= y_test, y_pred=test_pred)

    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.4f}, Acc: {acc:.2f}% | Test loss: {test_loss:.4f}, Test acc: {test_acc:.2f}%")