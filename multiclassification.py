import torch 
from torch import nn
import matplotlib.pyplot as plt 
from sklearn.datasets import make_blobs 
from sklearn.model_selection import train_test_split 
#set hyperparameter 

NUM_CLASSES=4
NUM_FEATURES=2

RANDOM_SEED=42
X_blob,y_blob=make_blobs(n_samples=1000,n_features=NUM_FEATURES,centers=NUM_CLASSES,cluster_std=2.5,random_state=RANDOM_SEED)

X_blob=torch.from_numpy(X_blob).type(torch.float)
y_blob=torch.from_numpy(y_blob).type(torch.LongTensor)

X_blob_train,X_blob_test,y_blob_train,y_blob_test=train_test_split(X_blob,y_blob,test_size=0.2,random_state=RANDOM_SEED)
import matplotlib.pyplot as plt
plt.scatter(X_blob[:,0],X_blob[:,1])
plt.show()

#A=torch.arange(-10,10,1,dtype=torch.float32)
#print(A)
#W=px.scatter(A)

#q=px.line(torch.relu(A))
#q.show()
#def sigmoid(x):
 #   return 1/(1+torch.exp(-x))
#s=px.line(sigmoid(A))
#s.show()

#device agnostic code
device='cuda' if torch.cuda.is_available() else 'cpu'

class BlobModel(nn.Module):
    def __init__(self,input_features,output_features,hidden_units=8):
        super().__init__()
        self.linear_layer_stack=nn.Sequential(
            nn.Linear(in_features=input_features,out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units,out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units,out_features=output_features)
            
            
            )
    def forward(self,x):
        return self.linear_layer_stack(x) 
model_4=BlobModel(input_features=2,output_features=4,hidden_units=12).to(device)
print(model_4)

loss_fn=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(params=model_4.parameters(),lr=0.15)
model_4.eval()
with torch.inference_mode():
    y_logits=model_4(X_blob_test.to(device))

print(y_logits[:10])
print(y_blob_test[:10])


def accuracy_fn(y_true,y_preds):
    correct=torch.eq(y_true,y_preds).sum().item()
    acc=(correct/len(y_preds))*100
    return acc

y_preds_prob=torch.softmax(y_logits,dim=1)

print(y_preds_prob[:5])
y_preds=torch.argmax(y_preds_prob,dim=1)


epoch =10000
torch.manual_seed(42)

X_blob_train,y_blob_train=X_blob_train.to(device),y_blob_train.to(device)
X_blob_test,y_blob_test=X_blob_test.to(device),y_blob_test.to(device)

for epoch in range(epoch):
    model_4.train()
    y_logits=model_4(X_blob_train)
    y_pred=torch.softmax(y_logits,dim=1).argmax(dim=1)
    loss=loss_fn(y_logits,y_blob_train)
    acc=accuracy_fn(y_true=y_blob_train ,y_preds=y_pred)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model_4.eval()
    with torch.inference_mode():
        test_logits=model_4(X_blob_test)
        test_preds=torch.softmax(test_logits,dim=1 ).argmax(dim=1)
        test_loss=loss_fn(test_logits,y_blob_test)
        test_acc=accuracy_fn(y_true=y_blob_test,y_preds=test_preds)
    if epoch%10==0:
        print(f"Epoch :{epoch} |Loss:{loss:.2f} | Acc :{acc:.2f} | Test loss:{test_loss:.4f} |Test acc:{test_acc:.2f}")
 
model_4.eval()
with torch.inference_mode():
    y_logits=model_4(X_blob_test)
print(y_logits[:10])

y_preds_prob=torch.softmax(y_logits,dim=1)
print(y_preds_prob[:10])
 
print(y_blob_test)
y_pred=torch.argmax(y_preds_prob,dim=1)
print(y_pred[:10])

from torchmetrics import Accuracy