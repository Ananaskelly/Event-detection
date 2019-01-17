class Config:
    def __init__(self, shape, learning_rate, n_classes):
        # shape: частоты, время, каналы
        self.shape = shape
        self.learning_rate = learning_rate
        self.n_classes = n_classes
