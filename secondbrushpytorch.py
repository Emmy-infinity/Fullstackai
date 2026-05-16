import sklearn
from sklearn.datasets import make_circles
n_samples=1000000
X,y=make_circles(n_samples,noise=0.03,random_state=42)
print(len(X),len(y))
import pandas
import pandas as  pd  
circles=pd.DataFrame({"X1":X[:,0],
                      "X2":X[:,1],
                      "label":y})
circles.head(10)
print(circles.head(10))


X_sample=X[0]
y_sample=y[0]


import plotly.express as px 
e=px.scatter(x=X[:,0],y=X[:,1] )
e.show()

#import matplotlib.pyplot as plt 
#plt.scatter(x=X[:,0],y=X[:,1 ],c=y,cmap=plt.cm.RdYlBu)
#plt.show()
#print(X_sample)
#print(y_sample)
import torch 
X=torch.from_numpy(X).type(torch.float)
y=torch.from_numpy(y).type(torch.float)
print(X[:5],y[:5])

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2 ,random_state=42)
print(len(X_train),len(y_train),len(y_test))

from torch import nn 
device="cuda" if torch.cuda.is_available() else "cpu"

class CircleModelV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1=nn.Linear(in_features=2,out_features=10)
        self.layer_2=nn.Linear(in_features=10,out_features=10)
        self.layer_3=nn.Linear(in_features=10,out_features=1)
    def forward(self,x):
        z=self.layer_1(x)
        z=self.layer_2(z)
        z=self.layer_3(z)
        return self.layer_3(self.layer_2(self.layer_1(x)))

model_1=CircleModelV1()

loss_fn=nn.BCEWithLogitsLoss()
optimizer=torch.optim.SGD(params=model_1.parameters(),lr=0.1)
model_0=model_1





class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1=nn.Linear(in_features=2,out_features=5)
        self.layer_2=nn.Linear(in_features=5,out_features=1)
    def forward(self,x):
        return self.layer_2(self.layer_1(x))
model_0=CircleModelV0().to(device)

class CircleModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1=nn.Linear(in_features=2,out_features=10)
        self.layer_2=nn.Linear(in_features=10,out_features=10)
        self.layer_3=nn.Linear(in_features=10,out_features=1)
        self.relu=nn.ReLU()
        
        
    def forward(self,x):
        return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1)))).to(device)

model_0=    CircleModelV2()
#print(list(model_0.parameters()))
#print(model_0.state_dict())
#with torch.inference_mode():
 #   untrained_preds=model_0(X_test.to(device))
  #  print(len(untrained_preds))
#loss_fn=nn.BCEWithLogitsLoss()
#optimizer=torch.optim.Adam(params=model_0.parameters())


def accuracy_fn(y_true,y_preds):
    correct=torch.eq(y_true,y_preds).sum().item()
    acc=(correct/len(y_preds))*100
    return acc


#print(model_0)
with torch.inference_mode():
    y_logits=model_0(X_test.to(device))[:5]
#print(y_logits)
#print(y_test)

y_pred_probs=torch.sigmoid(y_logits)
y_pred=torch.round(y_pred_probs)

y_pred_labels=torch.round(torch.sigmoid(model_0(X_test.to(device))[:5]))

#print(torch.eq(y_pred.squeeze(),y_pred_labels.squeeze()))
#print(y_pred.squeeze())


torch.manual_seed(42)


epoch=1000
X_train,y_train=X_train.to(device),y_train.to(device)
X_test,y_test=X_test.to(device),y_test.to(device)

for epoch in range(epoch):
    model_0.train()
    y_logits=model_0(X_train).squeeze()
    y_pred=torch.round(torch.sigmoid(y_logits))
    
    loss=loss_fn(y_logits,y_train)
    acc=accuracy_fn(y_true=y_train,y_preds=y_pred)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    #print(loss_fn)
    
    model_0.eval()
    with torch.inference_mode():
        test_logits=model_0(X_test).squeeze()
        test_pred=torch.round(torch.sigmoid(test_logits))
        test_loss=loss_fn(test_logits,y_test)
        test_acc=accuracy_fn(y_true=y_test,y_preds=test_pred)
    if epoch%10==0:
        print(f"Epoch:{epoch} | Loss:{loss:.5f} Acc:{acc:.2f}  |Test lOSS:{test_loss:.5f} ,Test acc:{test_acc:.2f}")






import requests
from pathlib import Path

if Path('helper_functions.py').is_file():
    pass
else:
    pass 