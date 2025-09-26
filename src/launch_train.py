from dvagen.configs import get_train_args
from dvagen.train.train import train


if __name__ == "__main__":
    train_args = get_train_args()
    train(train_args)
