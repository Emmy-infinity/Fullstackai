import pytorch_lightning as pl
import torch
from torch import nn, optim
from torch.autograd import Variable
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
print("torch version:",torch.__version__)
print("pytorch ligthening version:",pl.__version__)
import pytorch_lightning as pl
import torch
from torch import nn, optim
import torchvision
from torchvision import transforms

from torch.autograd import Variable
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
print("torch version:",torch.__version__)
train_data_path='lung_colon_image_set/train_dataset'
transforms = transforms.Compose([transforms.Resize(64),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225] )])
print("pytorch ligthening version:",pl.__version__)
train_data = torchvision.datasets.ImageFolder(root=train_data_path,transform=transforms)


target_data_path='lung_colon_image_set/val_dataset'
target_data_set = torchvision.datasets.ImageFolder(root=target_data_path,transform=transforms)




train_data = list(zip(train_data, target_data_set))
xor_data_train_loader = DataLoader(train_data, batch_size=1,num_workers=4)


class XORModel(pl.LightningModule):
    def __init__(self):
        super(XORModel,self).__init__()
        self.input_layer = nn.Linear(2, 4)
        self.output_layer = nn.Linear(4,1)
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.MSELoss()
    def forward(self, input):
        #print("INPUT:", input.shape)
        x = self.input_layer(input)
        #print("FIRST:", x.shape)
        x = self.sigmoid(x)
        #print("SECOND:", x.shape)
        output = self.output_layer(x)
        #print("THIRD:", output.shape)
        return output
    def configure_optimizers(self):
        params = self.parameters()
        optimizer = optim.Adam(params=params, lr = 0.01)
        return optimizer
    def training_step(self, batch, batch_idx):
        xor_input, xor_target = batch
        #print("XOR INPUT:", xor_input.shape)
        #print("XOR TARGET:", xor_target.shape)
        outputs = self(xor_input)
        #print("XOR OUTPUT:", outputs.shape)
        loss = self.loss(outputs, xor_target)
        return loss

from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
checkpoint_callback = ModelCheckpoint()
model = XORModel()
trainer = pl.Trainer(max_epochs=1000, callbacks=[checkpoint_callback])
trainer.fit(model, train_dataloaders=xor_data_train_loader)
train_model = model.load_from_checkpoint(checkpoint_callback.best_model_path)
test = torch.utils.data.DataLoader(xor_input, batch_size=1)
for val in xor_input:
    _ = train_model(val)
    print([int(val[0]),int(val[1])], int(_.round()))
