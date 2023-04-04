import torch
from torch import nn


class BankNLP(nn.Module):
    def __init__(
        self, drop_prob, n_input, n_hidden1, n_hidden2, n_hidden3, n_hidden4, n_hidden5
    ):
        super().__init__()

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop_prob)
        self.fc1 = nn.Linear(n_input, n_hidden1)
        self.fc2 = nn.Linear(n_hidden1, n_hidden2)
        self.fc3 = nn.Linear(n_hidden2, n_hidden3)
        self.fc4 = nn.Linear(n_hidden3, n_hidden4)
        self.fc5 = nn.Linear(n_hidden4, n_hidden5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc4(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc5(x)

        return x


class FullModel(nn.Module):
    def __init__(
        self,
        ecb_params,
        fed_params,
        n_input_series_meta,
        n_hidden_tot_1,
        n_hidden_tot_2,
        n_hidden_tot_3,
        drop_prob,
    ):
        super().__init__()

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop_prob)
        self.ecb_fc = BankNLP(**ecb_params)
        self.fed_fc = BankNLP(**fed_params)
        self.fc_tot_1 = nn.Linear(256 + n_input_series_meta, n_hidden_tot_1)
        self.fc_tot_2 = nn.Linear(n_hidden_tot_1, n_hidden_tot_2)
        self.fc_tot_3 = nn.Linear(n_hidden_tot_2, n_hidden_tot_3)

    def forward(self, x_ecb, x_fed, x_series_meta):
        x_ecb = self.ecb_fc(x_ecb)
        x_fed = self.ecb_fc(x_fed)
        x_tot = torch.cat(
            (
                x_ecb.view(x_ecb.size(0), -1),
                x_fed.view(x_fed.size(0), -1),
                x_series_meta.view(x_series_meta.size(0), -1),
            ),
            dim=1,
        )

        x_tot = self.fc_tot_1(x_tot)
        x_tot = self.relu(x_tot)
        x_tot = self.dropout(x_tot)

        x_tot = self.fc_tot_2(x_tot)
        x_tot = self.relu(x_tot)
        x_tot = self.dropout(x_tot)

        x_tot = self.fc_tot_3(x_tot)

        return x_tot
