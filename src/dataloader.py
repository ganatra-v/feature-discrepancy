import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

def load_two_moons(args):
    x, y = make_moons(n_samples = 5000, random_state= 42)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
    return x_train, y_train, x_test, y_test


def load_noisy_two_moons(args):
    x, y = make_moons(n_samples = 5000, noise = 0.3, random_state= 42)
    noisy_features = np.random.normal(0, 1, (x.shape[0], 5))
    x = np.hstack([x, noisy_features])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
    return x_train, y_train, x_test, y_test




def load_sonar(args):
    pass

def load_data(args):
    if args.dataset == "two-moons":
        return load_two_moons(args)
    elif args.dataset == "noisy-two-moons":
        return load_noisy_two_moons(args)
    elif args.dataset == "sonar":
        return load_sonar(args)