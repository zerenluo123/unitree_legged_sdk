from datetime import datetime
import os

import torch
from torch import nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader

from mlp import MLP
from motion_data import MotionData

def main():
    # ! define learning parameters
    device = torch.device('cuda')
    num_learning_epochs = 50
    learning_rate = 1e-3
    hidden_layers = [128, 128, 128]

    # ! logging path
    task_path = os.path.dirname(os.path.realpath(__file__))
    log_dir = os.path.join(task_path, 'runs', datetime.now().strftime('%b%d_%H-%M-%S'))
    writer = SummaryWriter(log_dir=log_dir, flush_secs=10)

    # ! divide training and test data
    custom_dataset = MotionData(root_dir='../data')
    train_size = int(len(custom_dataset) * 0.9)
    test_size = len(custom_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(custom_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=0, drop_last=False)

    # ! create an actuator network model
    actuator_model = MLP(hidden_layers,
                       nn.Tanh,
                       custom_dataset.model_in_size,
                       output_size=3,
                       init_scale=1.4).to(device)
    optimizer = optim.Adam(actuator_model.parameters(), lr=learning_rate, weight_decay=0.001)
    loss_fn = nn.MSELoss().to(device)

    # ! Training
    total_train_step = 0
    for i in range(num_learning_epochs):
        print('---------------- {} training epoch start ---------------- '.format(i+1))
        for data in train_loader:
            model_ins, labels = data
            model_ins, labels = model_ins.to(device), labels.to(device)
            model_outs = actuator_model(model_ins.float())
            loss = loss_fn(model_outs, labels.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_step += 1
            if total_train_step % 100 == 0:
                print('Train Step: {}, Train Loss: {}'.format(total_train_step, loss.item()))
                writer.add_scalar('train_loss', loss.item(), total_train_step)

            if total_train_step % 500 == 0:
                save_path = log_dir + '/snapshot{}.pt'.format(total_train_step)
                torch.save(actuator_model, save_path)

        total_test_loss = 0
        with torch.no_grad():
            for data in test_loader:
                model_ins, labels = data
                model_ins, labels = model_ins.to(device), labels.to(device)
                model_outs = actuator_model(model_ins.float())
                loss = loss_fn(model_outs, labels.float())

                total_test_loss += loss

        print('Test Loss: {}'.format(total_test_loss))
        # writer.add_scalar('test_loss', )


if __name__ == "__main__":
    main()

# model_in, label = train_data[0]
# print("model_in", model_in)
# print("label", label)
