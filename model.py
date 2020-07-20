# model = Net()
# model.load_state_dict('./final_model.pth')


class Net(nn.Module):
    # Batch shape is (3,32,32)
    def __init__(self):
        super(TrafficSignNet, self).__init__()

        # prevent overfitting by dropping randomly
        self.dropout = nn.Dropout(p=0.5)

        # convolutional 100 maps of 42x42 neurons 7x7
        self.conv1 = nn.Conv2d(3, 100, 7)

        # max pooling 100 maps of 21x21 neurons 2x2
        self.batch1 = nn.BatchNorm2d(100)
        self.pool = nn.MaxPool2d(2)

        # convolutional 150 maps of 18x18 neurons 4x4
        self.conv2 = nn.Conv2d(100, 150, 4)

        # max pooling 150 maps of 9x9 neurons 2x2
        self.batch2 = nn.BatchNorm2d(150)
        # use self.pool; Same pool kernal

        # convolutional 250 maps of 6x6 neurons 4x4
        self.conv3 = nn.Conv2d(150, 250, 4)

        # max pooling 250 maps of 3x3 neurons 2x2
        self.batch3 = nn.BatchNorm2d(250)
        # use self.pool; Same pool kernal

        # fully connected 300 neurons 1x1
        self.fc1 = nn.Linear(250 * 1 * 1, 300)
        self.batch4 = nn.BatchNorm1d(300)

        # fully connected 43 neurons 1x1
        self.fc2 = nn.Linear(300, 43)

    def forward(self, x):
        # convolutional 100 maps of 42x42 neurons 7x7
        x = self.conv1(x)

        # ELU activation function
        x = F.elu(x)

        # max pooling 100 maps of 21x21 neurons 2x2
        x = self.pool(x)
        x = self.batch1(x)

        # prevent overfitting by dropping randomly
        x = self.dropout(x)

        # convolutional 150 maps of 18x18 neurons 4x4
        x = self.conv2(x)

        # ELU activation function
        x = F.elu(x)

        # max pooling 150 maps of 9x9 neurons 2x2
        x = self.pool(x)
        x = self.batch2(x)

        # prevent overfitting by dropping randomly
        x = self.dropout(x)

        # convolutional 250 maps of 6x6 neurons 4x4
        x = self.conv3(x)

        # ELU activation function
        x = F.elu(x)

        # max pooling 250 maps of 3x3 neurons 2x2
        x = self.pool(x)
        x = self.batch3(x)

        # prevent overfitting by dropping randomly
        x = self.dropout(x)

        # fully connected 300 neurons 1x1
        x = x.view(-1, 250 * 1 * 1)
        x = self.fc1(x)

        # ELU activation function
        x = F.elu(x)
        # prevent overfitting by dropping randomly
        x = self.batch4(x)
        x = self.dropout(x)

        # fully connected 43 neurons 1x1
        x = self.fc2(x)

        return x
