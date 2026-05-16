import torchvision
from torchvision import transforms,data,nn
train_data_path = "./train/"
transforms = transforms.Compose([transforms.Resize(64),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225] )])
train_data = torchvision.datasets.ImageFolder(root=train_data_path,transform=transforms)
val_data_path = "./val/"
val_data = torchvision.datasets.ImageFolder(root=val_data_path,transform=transforms)
test_data_path = "./test/"
test_data = torchvision.datasets.ImageFolder(root=test_data_path,transform=transforms)
batch_size=64
train_data_loader = data.DataLoader(train_data, batch_size=batch_size)
val_data_loader = data.DataLoader(val_data, batch_size=batch_size)
test_data_loader = data.DataLoader(test_data, batch_size=batch_size)


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(12288, 84)
        self.fc2 = nn.Linear(84, 50)
        self.fc3 = nn.Linear(50,2)
        def forward(self):
            x = x.view(-1, 12288)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.softmax(self.fc3(x))
            return x
simplenet = SimpleNet()