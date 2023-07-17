# %%

## Graph classification example from 
## https://www.exxactcorp.com/blog/Deep-Learning/gnn-demo-using-pytorch-lightning-and-pytorch-geometric
## Using PyTorch Lightning and PyTorch Geometric.
import time
import numpy as np
import torch
from torch.nn import Dropout, Linear, ReLU
import torch_geometric
from torch_geometric.datasets import TUDataset, GNNBenchmarkDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, Sequential, global_mean_pool
import pytorch_lightning as pl
import torch.nn.functional as F

# %%
class LightningGCN(pl.LightningModule):

    def __init__(self, **kwargs):
        super(LightningGCN, self).__init__()

        self.num_features = kwargs["num_features"] \
                    if "num_features" in kwargs.keys() else 3
        self.num_classes = kwargs["num_classes"] \
                    if "num_classes" in kwargs.keys() else 2

        # hidden layer node features
        self.hidden = 256 

        self.model = Sequential("x, edge_index, batch_index", [        
                (GCNConv(self.num_features, self.hidden),
                    "x, edge_index -> x1"),
                (ReLU(), "x1 -> x1a"),                                 
                (Dropout(p=0.5), "x1a -> x1d"),                       
                (GCNConv(self.hidden, self.hidden), "x1d, edge_index -> x2"),
                (ReLU(), "x2 -> x2a"),   
                (Dropout(p=0.5), "x2a -> x2d"),                             
                (GCNConv(self.hidden, self.hidden), "x2d, edge_index -> x3"),
                (ReLU(), "x3 -> x3a"),
                (Dropout(p=0.5), "x3a -> x3d"),
                (GCNConv(self.hidden, self.hidden), "x3d, edge_index -> x4"),
                (ReLU(), "x4 -> x4a"),                                
                (Dropout(p=0.5), "x4a -> x4d"),                           
                (GCNConv(self.hidden, self.hidden), "x4d, edge_index -> x5"),
                (ReLU(), "x5 -> x5a"),   
                (Dropout(p=0.5), "x5a -> x5d"),                              
                (global_mean_pool, "x5d, batch_index -> x6"),                
                (Linear(self.hidden, self.num_classes), "x6 -> x_out")])        



    def forward(self, x, edge_index, batch_index):

        x_out = self.model(x, edge_index, batch_index)

        return x_out

    def training_step(self, batch, batch_index):

        x, edge_index = batch.x, batch.edge_index
        batch_index = batch.batch

        x_out = self.forward(x, edge_index, batch_index)

        loss = F.cross_entropy(x_out, batch.y)

        # metrics here
        pred = x_out.argmax(-1)
        label = batch.y
        accuracy = (pred == label).sum() / pred.shape[0]

        self.log("loss/train", loss)
        self.log("accuracy/train", accuracy)

        return loss

    def validation_step(self, batch, batch_index):
        x, edge_index = batch.x, batch.edge_index
        batch_index = batch.batch

        x_out = self.forward(x, edge_index, batch_index)

        loss = F.cross_entropy(x_out, batch.y)

        pred = x_out.argmax(-1)

        return x_out, pred, batch.y

    def validation_epoch_end(self, validation_step_outputs):

        val_loss = 0.0
        num_correct = 0
        num_total = 0

        for output, pred, labels in validation_step_outputs:

            val_loss += F.cross_entropy(output, labels, reduction="sum")

            num_correct += (pred == labels).sum()
            num_total += pred.shape[0]


            val_accuracy = num_correct / num_total
            val_loss = val_loss / num_total

        self.log("accuracy/val", val_accuracy)
        self.log("loss/val", val_loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr = 3e-4)

# %%
def evaluate(model, test_loader, save_results=True, tag="_default", verbose=False):

    # get test accuracy score

    num_correct = 0.
    num_total = 0.

    my_device = "cuda" if torch.cuda.is_available() else "cpu"
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    model.eval()
    total_loss = 0
    total_batches = 0

    for batch in test_loader:

        pred = model(batch.to(my_device))

        loss = criterion(pred, batch.y.to(my_device))

        num_correct += (pred.argmax(dim=1) == batch.y).sum()
        num_total += pred.shape[0]

        total_loss += loss.detach()
        total_batches += batch.batch.max()

    test_loss = total_loss / total_batches
    test_accuracy = num_correct / num_total

    if verbose:
        print(f"accuracy = {test_accuracy:.4f}")

    results = {"accuracy": test_accuracy, \
        "loss": test_loss, \
        "tag": tag }

    return results

# %%
dataset = TUDataset(root="data", name="PROTEINS")
dataset = dataset.shuffle()

num_samples = len(dataset)
batch_size = 32

num_val = num_samples // 10

val_dataset = dataset[:num_val]
test_dataset = dataset[num_val:2 * num_val]
train_dataset = dataset[2 * num_val:]

train_loader = DataLoader(train_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

it = iter(train_loader)
batch = next(it)
batch

num_features = batch.x.shape[1]
num_classes = train_dataset.num_classes

# %%
lightning_model = LightningGCN(num_features=num_features, \
                                num_classes=num_classes)

num_epochs = 100
val_check_interval = len(train_loader)

trainer = pl.Trainer(max_epochs = num_epochs, \
        val_check_interval=val_check_interval)
trainer.fit(lightning_model, train_loader, val_loader)

# %%



