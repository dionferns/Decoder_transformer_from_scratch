import torch
import lightning as L
from torch.utils.data import TensorDataset, DataLoader
from models.transformer import DecoderOnlyTransformer

# Define vocabulary mapping
token_to_id = {'what': 0, 'is': 1, 'statquest': 2, 'awesome': 3, '<EOS>': 4}
id_to_token = {v: k for k, v in token_to_id.items()}

# Training data
inputs = torch.tensor([[0, 1, 2, 4, 3], [2, 1, 0, 4, 3]])
labels = torch.tensor([[1, 2, 4, 3, 4], [1, 0, 4, 3, 4]])

# DataLoader
dataset = TensorDataset(inputs, labels)
dataloader = DataLoader(dataset)

# Initialize model
model = DecoderOnlyTransformer(num_tokens=len(token_to_id), d_model=2, max_len=6)

# Train model
trainer = L.Trainer(max_epochs=30)
trainer.fit(model, train_dataloaders=dataloader)
