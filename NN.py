import time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from utils import mean_absolute_percentage_error, theilU, PT_test
    
# RNN and PSN implementation
class RNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = 1

        self.rnn = torch.nn.RNN(self.input_size, self.hidden_size, self.n_layers)
        self.fc = torch.nn.Linear(self.hidden_size, self.output_size)
        
    def forward(self, x):
        batch_size = x.size(1)
        h0 = torch.zeros(self.n_layers, batch_size, self.hidden_size)
        out, hidden = self.rnn(x, h0)
        out = out.contiguous().view(-1, self.hidden_size)
        out = self.fc(out)
        
        return out

class PSN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PSN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fc = torch.nn.Linear(self.input_size, self.hidden_size)
        
    def forward(self, x):
        x = self.fc(x)
        x = torch.sum(x, axis=1)
        x = torch.sigmoid(x)
        return x

# General Model Class to regroup all features needed
# Usage example :
# modelMLP = Model("MLP")
# modelMLP.setup(5, 4, 1)
# modelMLP.train(trainLoader, validLoader)
# modelMLP.evaluate(testLoader)

class Model:
    def __init__(self, NNtype):
        self.NNtype = NNtype
        self.model = None
        self.optimizer = None
        
        self.input_size = None
        self.hidden_size = None
        self.output_size = None
        
        self.epochs = None        
        self.optim_type = None
        self.lr = None
        self.momentum = None
        self.train_losses = None
        self.valid_losses = None
        
        
    def setup(self, input_size, hidden_size, output_size=1, epochs=10000, optim_type="SGD", lr=0.001, momentum=0.001):
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.epochs = epochs        
        self.optim_type = optim_type
        self.lr = lr
        self.momentum = momentum
        
        self.loss_fn = torch.nn.MSELoss(reduction="mean")
        self.train_losses = None
        self.valid_losses = None
        
        if self.NNtype == "MLP":
            self.model = torch.nn.Sequential(torch.nn.Linear(self.input_size, self.hidden_size),
                                             torch.nn.Sigmoid(),
                                             torch.nn.Linear(self.hidden_size, self.output_size)
                                            )
            print(self.model)
            
        elif self.NNtype == "RNN":
            self.model = RNN(self.input_size, self.hidden_size, self.output_size)
            print(self.model)
        elif self.NNtype == "PSN":
            self.model = PSN(self.input_size, self.hidden_size, self.output_size)
            print(self.model)
        else:
            return "NN Type not implemented. Choose between ['MLP', 'RNN', 'PSN']"
        
        if self.optim_type == "SGD":
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                             lr = self.lr,
                                             momentum = self.momentum
                                            )
        else :
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                              lr = self.lr
                                             )
        print(self.optimizer)
    
    
    def train(self, trainloader, validloader, verbose=True): 
                
        self.train_losses = []
        self.valid_losses = []
        
        start_time = time.time()
        for epoch in range(self.epochs):
            self.model.train()
    
            train_loss, val_loss = [], []

            for features, target in trainloader:
                
                if self.NNtype == "RNN":
                    features = features.view(1, features.size(0), features.size(1))
                
                self.optimizer.zero_grad()

                outputs = self.model(features)
                loss = self.loss_fn(outputs, target.view(-1,1))
                loss.backward()
                self.optimizer.step()

                train_loss.append(loss.item())

            self.model.eval()
            with torch.no_grad():
                for features, target in validloader:
                    
                    if self.NNtype == "RNN":
                        features = features.view(1, features.size(0), features.size(1))
                    
                    outputs = self.model(features)
                    loss = self.loss_fn(outputs, target.view(-1,1))

                    val_loss.append(loss.item())

            self.train_losses.append(np.mean(train_loss))
            self.valid_losses.append(np.mean(val_loss))
            
            if verbose:
                if (epoch+1) % 100 == 0 or epoch+1 == 1:
                    printtext = "[{}] Epoch {}/{} - Train Loss : {:.6f} / Val Loss : {:.6f}"
                    print(printtext.format(time.strftime("%M:%S", time.gmtime(time.time()-start_time)),
                                           epoch + 1,
                                           self.epochs,
                                           np.mean(train_loss),
                                           np.mean(val_loss))
                         )
                    
        train_preds, train_targets = [], []
        valid_preds, valid_targets = [], []
        self.model.eval()
        with torch.no_grad():
            for features, target in trainloader:
                if self.NNtype == "RNN":
                    features = features.view(1, features.size(0), features.size(1))
                outputs = self.model(features)
                if self.NNtype == "PSN":
                    train_preds += outputs.numpy().tolist()
                else:
                    train_preds += outputs.numpy().T.tolist()[0]
                train_targets += target.numpy().tolist()
            for features, target in validloader:
                if self.NNtype == "RNN":
                    features = features.view(1, features.size(0), features.size(1))
                outputs = self.model(features)
                if self.NNtype == "PSN":
                    valid_preds += outputs.numpy().tolist()
                else:
                    valid_preds += outputs.numpy().T.tolist()[0]
                valid_targets += target.numpy().tolist()
        
        self.trainRMSE = mean_squared_error(train_targets, train_preds)
        self.trainMAE = mean_absolute_error(train_targets, train_preds)
        self.trainMAPE = mean_absolute_percentage_error(np.array(train_targets), np.array(train_preds))
        self.trainTheilU = theilU(np.array(train_targets), np.array(train_preds))
        
        self.validRMSE = mean_squared_error(valid_targets, valid_preds)
        self.validMAE = mean_absolute_error(valid_targets, valid_preds)
        self.validMAPE = mean_absolute_percentage_error(np.array(valid_targets), np.array(valid_preds))
        self.validTheilU = theilU(np.array(valid_targets), np.array(valid_preds))
        
        print("Train MAE : {:.4f} | Train MAPE  : {:.4f} | Train RSME : {:.4f} | Train Theil-U {:.4f}".format(self.trainMAE, self.trainMAPE, self.trainRMSE, self.trainTheilU))
        print("Valid MAE : {:.4f} | Valid MAPE  : {:.4f} | Valid RSME : {:.4f} | Valid Theil-U {:.4f}".format(self.validMAE, self.validMAPE, self.validRMSE, self.validTheilU))
        
        
    def evaluate(self, dataloader):
        
        preds, targets = [], []
        self.model.eval()
        with torch.no_grad():
            for features, target in dataloader:
                if self.NNtype == "RNN":
                    features = features.view(1, features.size(0), features.size(1))
                outputs = self.model(features)
                if self.NNtype == "PSN":
                    preds += outputs.numpy().tolist()
                else:
                    preds += outputs.numpy().T.tolist()[0]
                targets += target.numpy().tolist()
        
        self.testRMSE = mean_squared_error(targets, preds)
        self.testMAE = mean_absolute_error(targets, preds)
        self.testMAPE = mean_absolute_percentage_error(np.array(targets), np.array(preds))
        self.testTheilU = theilU(np.array(targets), np.array(preds))
        self.testPT = PT_test(np.array(targets), np.array(preds))

        print("Test MAE : {:.6f} | Test MAPE  : {:.6f} | Test RSME : {:.6f} | Test Theil-U {:.6f}".format(self.testMAE, self.testMAPE, self.testRMSE, self.testTheilU))
    
    # to retrieve the prediction and the target values as list
    def Getevaluation(self, dataloader):
        
        preds, targets = [], []
        self.model.eval()
        with torch.no_grad():
            for features, target in dataloader:
                if self.NNtype == "RNN":
                    features = features.view(1, features.size(0), features.size(1))
                outputs = self.model(features)
                if self.NNtype == "PSN":
                    preds += outputs.numpy().tolist()
                else:
                    preds += outputs.numpy().T.tolist()[0]
                targets += target.numpy().tolist()
        
        self.testRMSE = mean_squared_error(targets, preds)
        self.testMAE = mean_absolute_error(targets, preds)
        self.testMAPE = mean_absolute_percentage_error(np.array(targets), np.array(preds))
        self.testTheilU = theilU(np.array(targets), np.array(preds))
        self.testPT = PT_test(np.array(targets), np.array(preds))

        return(preds, targets)