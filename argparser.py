import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='GradEnFS main program.')

    # argument for loading the dataset
    parser.add_argument('--dataset', type=str, default='madelon', help='The dataset to be used (default: madelon, option: mnist, usps, coil20, madelon, isolet, basehock, pcmac, prostate_ge, tox).')
    parser.add_argument('--training_batch_size', type=int, default=100, help='Input batch size for training (default: 100).')
    parser.add_argument('--evaluating_batch_size', type=int, default=10000, help='Input batch size for testing or validating (default: 10000).')

    # argument for model
    parser.add_argument('--hidden_dim', type=int, default=1000, help='Number of hidden neuron (default: 1000).')

    # hyperparameter for training
    parser.add_argument('--epoch', type=int, default=100, help='The number of training epoch (default: 100).')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.01).')
    parser.add_argument('--device', type=str, default='cpu', help='The device of running the program (default:cpu, option: cpu, cuda).')

    # hyperparameter for gradenfs algorithm 
    parser.add_argument('--network', type=str, default='sparse', help='Choose the network (default: sparse, option: dense, sparse).')
    parser.add_argument('--epsilon', type=int, default=1, help='Control of the sparsity level (default: 1).')
    parser.add_argument('--alpha', type=float, default=0.3, help='Prunning rate during the topology update (default: 0.3).')
    parser.add_argument('--beta', type=float, default=0.7, help='Hyperparameter for calculating the neuron importance metric (default: 0.7).')
    parser.add_argument('--k_list', type=int, action='store', nargs='*', default=[20], help='The list of the numbers of the selected features.')
    parser.add_argument('--batch_update', action='store_true', help='Choose to update topology after every batch.')
    
    # experiment setting
    parser.add_argument('--use_seeds', action='store_true', help='Choose to control random seeds for initializing network or not.')
    parser.add_argument('--seeds', type=int, action='store', nargs='*', default=[0, 1, 2, 3, 4], help='The list of seed for various repeating experiments, the seed is used to control the generation of network.')
    parser.add_argument('--repeat', type=int, default=5, help='The number of different experiment runs (default: 5).')

    return parser