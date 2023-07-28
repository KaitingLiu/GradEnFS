import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='Final Thesis main program.')

    # argument for loading the dataset
    parser.add_argument('--dataset', type=str, default='artificial', help='The dataset to use (default: artificial, option: artificial, mnist, madelon, basehock, isolet, coil20, usps).')
    parser.add_argument('--training_batch_size', type=int, default=100, help='Input batch size for training (default: 100).')
    parser.add_argument('--evaluating_batch_size', type=int, default=10000, help='Input batch size for testing or validating (default: 10000).')

    # argument for model
    parser.add_argument('--hidden_dim', type=int, default=1000, help='Number of hidden neuron (default: 1000).')

    # argument for training
    parser.add_argument('--epoch', type=int, default=500, help='The number of training epoch (default: 500).')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.01).')
    parser.add_argument('--device', type=str, default='cpu', help='The device of running the program. (default:cpu, option: cup, cuda)')

    # argument for dst algorithms
    parser.add_argument('--network', type=str, default='sparse', help='Choose the network (default: sparse, option: dense, sparse).')
    parser.add_argument('--epsilon', type=int, default=1, help='Control of the sparsity level (default: 1).')
    parser.add_argument('--alpha', type=float, default=0.3, help='Prunning rate during the topology update (default: 0.3).')
    parser.add_argument('--batch_update', action='store_true', help='Choose to update topology after every batch.')
    parser.add_argument('--use_seeds', action='store_true', help='Choose to control random seeds in pytorch.')
    parser.add_argument('--seeds', type=int, action='store', nargs='*', default=[0, 1, 2, 3, 4], help='The list of seed for repeating experiment, use to control the generation of initial sparse network.')
    
    # argument for feature selection
    parser.add_argument('--k_list', type=int, action='store', nargs='*', default=[20], help='The list of the numbers of the selected features.')
    
    # repeat times
    parser.add_argument('--repeat', type=int, default=5, help='The number of experiment (default: 5).')

    # argument for generating artificial noisy data 
    parser.add_argument('--n_classes', type=int, default=2, help='The number of classes (or labels) of the classification problem (default: 2).')
    parser.add_argument('--n_samples', type=int, default=1000, help='The number of samples (default: 1000).')
    parser.add_argument('--n_features', type=int, default=500, help='The total number of features (default: 500).')
    parser.add_argument('--n_informative', type=int, default=5, help='The number of informative features (default: 5).')
    parser.add_argument('--n_redundant', type=int, default=15, help='The number of redundant features, and these features are generated as random linear combinations of the informative features (default: 15).')
    parser.add_argument('--random_state', type=int, default=0, help='Determines random number generation for dataset creation (default: 0).')
    parser.add_argument('--class_sep', type=float, default=1.0, help='The factor multiplying the hypercube size, and larger values spread out the clusters/classes and make the classification task easier (default: 1.0).')
    parser.add_argument('--flip_y', type=float, default=0.01, help='The fraction of samples whose class is assigned randomly, and larger values introduce noise in the labels and make the classification task harder (default: 0.01).')
    parser.add_argument('--n_clusters_per_class', type=int, default=1, help='The number of clusters per class (default: 1).')
    parser.add_argument('--shuffle', action='store_true', help='Shuffle the samples and the features. Without shuffling, all useful features are contained in the columns X[:, :n_informative + n_redundant + n_repeated].')

    return parser