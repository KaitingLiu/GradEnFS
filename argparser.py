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
    parser.add_argument('--epoch', type=int, default=200, help='The number of training epoch (default: 200).')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.01).')
    parser.add_argument('--device', type=str, default='cpu', help='The device of running the program. (default:cpu, option: cup, cuda)')

    # argument for dst algorithms
    parser.add_argument('--network', type=str, default='dense', help='Choose the network (default: dense, option: dense, set, fixed_er).')
    parser.add_argument('--epsilon', type=int, default=1, help='Control of the sparsity level (default: 1).')
    parser.add_argument('--alpha', type=float, default=0.3, help='Prunning rate during the topology update (default: 0.3).')
    parser.add_argument('--batch_update', action='store_true', help='Choose to update topology after every batch.')
    parser.add_argument('--use_seeds', action='store_true', help='Choose to control random seeds in pytorch.')
    parser.add_argument('--seeds', type=int, action='store', nargs='*', default=[0, 1, 2, 3, 4], help='The list of seed for repeating experiment, use to control the generation of initial sparse network.')

    # argument for neuron importance metric
    parser.add_argument('--beta', type=float, default=1, help='Hyperparameter of neuron importance metric, control how many previous importance score should be taken into account (default: 1).')
    
    # argument for feature selection
    parser.add_argument('--k_list', type=int, action='store', nargs='*', default=[25, 50, 75, 100, 150, 200], help='The list of the numbers of the selected features.')
    
    # repeat times
    parser.add_argument('--repeat', type=int, default=5, help='The number of experiment (default: 5).')

    # argument for generating artificial noisy data 
    # TODO: change the help
    parser.add_argument('--n_classes', type=int, default=10, help='Control of the sparsity level (default: 100).')
    parser.add_argument('--n_samples', type=int, default=1000, help='Control of the sparsity level (default: 100).')
    parser.add_argument('--n_features', type=int, default=500, help='Control of the sparsity level (default: 100).')
    parser.add_argument('--n_informative', type=int, default=5, help='Control of the sparsity level (default: 100).')
    parser.add_argument('--n_redundant', type=int, default=15, help='Control of the sparsity level (default: 100).')
    parser.add_argument('--random_state', type=int, default=0, help='Control of the sparsity level (default: 100).')
    parser.add_argument('--class_sep', type=int, default=2, help='Control of the sparsity level (default: 100).')
    parser.add_argument('--flip_y', type=int, default=0.01, help='Control of the sparsity level (default: 100).')
    parser.add_argument('--n_clusters_per_class', type=int, default=1, help='Control of the sparsity level (default: 100).')
    parser.add_argument('--shuffle', action='store_true', help='Choose to use probability in neuron importance addition method.')

    return parser