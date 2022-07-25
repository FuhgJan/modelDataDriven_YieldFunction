import torch
import torch.nn as nn

dev = torch.device('cpu')
if torch.cuda.is_available():
    print("CUDA is available, running on GPU")
    dev = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    print("CUDA not available, running on CPU")

torch.manual_seed(2019)


class ICNN_net(torch.nn.Module):
    def __init__(self, inp, out, activation, num_hidden_units=100, num_layers=1):
        super(ICNN_net, self).__init__()
        self.fc1 = nn.Linear(inp, num_hidden_units, bias=True)
        self.fc2 = nn.ModuleList()
        for i in range(num_layers):
            self.fc2.append(nn.Linear(num_hidden_units+inp, num_hidden_units, bias=True))
        self.fc3 = nn.Linear(num_hidden_units+inp, out,bias=True)
        self.activation = activation

        self.loss_func = torch.nn.MSELoss()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-6)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5000, gamma=0.5)




    def forward(self, x_in):
        x = self.fc1(x_in)
        x = self.activation(x)
        for fc in self.fc2:
            xcat = torch.cat((x, x_in), 1)
            x = fc(xcat)
            x = self.activation(x)
        xcat = torch.cat((x, x_in), 1)
        x = self.fc3(xcat)
        return x


    def saveModel(self,PATH):

        torch.save(self.state_dict(), PATH)

    def recover_model(self, PATH):
        # Utility function for recovering the weights and biases of a network that we saved
        checkpoint = torch.load(PATH,map_location=torch.device('cpu') )
        self.load_state_dict(checkpoint)
        return 0



