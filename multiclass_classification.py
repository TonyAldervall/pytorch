import torch
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from models import BlobModel

device = "cuda" if torch.cuda.is_available() else "cpu"

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

NUM_CLASSES = 4
NUM_FEATURES = 2
RANDOM_SEED = 42

x_blob , y_blob = make_blobs(n_samples=1000,
                                n_features=NUM_FEATURES,
                                centers=NUM_CLASSES,
                                cluster_std=1.5,
                                random_state=RANDOM_SEED)

x_blob = torch.from_numpy(x_blob).type(torch.float).to(device)
y_blob = torch.from_numpy(y_blob).type(torch.long).to(device)

x_blob_train, x_blob_test, y_blob_train, y_blob_test = train_test_split(x_blob,
                                                                        y_blob,
                                                                        test_size=0.2,
                                                                        random_state=RANDOM_SEED)

m = BlobModel(input_features=2, output_features=4, hidden_units=8).to(device)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=m.parameters(), lr=0.1)

epochs = 100
for epoch in range(epochs):
    m.train()

    y_logits = m(x_blob_train)
    y_preds = torch.softmax(y_logits, dim=1).argmax(dim=1)

    loss = loss_fn(y_logits, y_blob_train)
    acc = accuracy_fn(y_true=y_blob_train, y_pred=y_preds)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    m.eval()
    with torch.inference_mode():
        y_test_logits = m(x_blob_test)
        y_test_preds = torch.softmax(y_test_logits, dim=1).argmax(dim=1)
        test_loss = loss_fn(y_test_logits, y_blob_test)
        test_acc = accuracy_fn(y_true=y_blob_test, y_pred=y_test_preds)

    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.4f}, Acc: {acc:.2f}% | Test loss: {test_loss:.4f}, Test acc: {test_acc:.2f}%")