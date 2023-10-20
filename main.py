# import lib
import os

from gradenfs import GradEnFS
from argparser import get_parser
import logging
import sys
import torch
from data_loading_util import get_dataset, Dataset
from data_saving_util import create_dir
from torch.utils.data import DataLoader

from model import MLP
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import json
import numpy as np

from evaluation_model import SVM_Model
from evaluation_model import KNN_Model
from evaluation_model import ExtraTree_Model

# function that create a logger
def setup_logger(args):
    # get logger and set level
    logger = logging.getLogger() # the effective level is warn so need to set the logger level
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(message)s')
    # file handler, if it already exists, remove it and file_handler will create a new one
    if os.path.exists(args.logs_name):
        os.remove(args.logs_name)
    file_handler = logging.FileHandler(args.logs_name)
    file_handler.setFormatter(formatter)
    # console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    # add handler
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    # record parameters
    logger.info("*************************************************arguments*************************************************")
    logger.info(args)
    # return logger
    return logger

# function to control the random seed for initializing the neural network
def seed_everything(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = True

# function which use test/validation set to evaluate the model's accuracy
def evaluate_mlp(model, data_loader, device, num_dataset):
    diff = 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            y_hat = model(x)

            # change y_hat to onehot
            _, ind = torch.topk(y_hat, dim=1, k=1) # get the largest probability's index
            y_hat.scatter_(index=ind, dim=1, value=1) # set it to 1
            y_hat[y_hat != 1] = 0 # other's are all 0

            # calculate the different items between y and y_hat
            diff += (torch.sum(torch.abs(y_hat - y))/2).detach().item()

    accuracy = 1 - (diff / num_dataset)
    return accuracy

# function for evaluating feature selection algorithm on accuracy
def evaluation_acc(evaluation_model, k_list, indexes, logger):
    accuracies = []
    for k in k_list:
        # get the indexes of the input neuron with the largest neuron importance scores
        k_indexes = indexes[:k]
        # use these k informative features to train model and get the accuracy
        accuracy = evaluation_model.train_and_test(k_indexes)
        # save the accuracy and log it
        accuracies.append(accuracy)
        info = 'The evaluation accuracy is {} for {} features'.format(accuracy, k)
        logger.info(info)
    # return the accuracies for every k features, and the longest inforamtive indexes
    return accuracies

# function shows how to train the model
def train(model, train_loader, validation_loader, optimizer, loss_function, gradenfs, args, logger):
    # variable
    losses= []
    valid_accuracies = []

    # generate the sparse network
    model.to(args.device)
    gradenfs.apply_mask()
        
    # start training
    for epoch_idx in range(1, args.epoch+1):
        model.train()
        for batch_idx, (x, y) in enumerate(train_loader, start=1):
            # get features and labels
            x, y = x.to(args.device), y.to(args.device)
            x.requires_grad = True

            # feed forward and calculate the loss
            optimizer.zero_grad()
            y_hat = model(x)
            loss = loss_function(y_hat, y)

            # backpropagate, get the gradient of loss wrt the input neuron, and update the network's weights
            loss.backward() # backpropagate
            grad = torch.mean(torch.abs(x.grad), dim=0) # get the gradient of loss with respect to every input neurons
            optimizer.step() # do the optimization

            # update the neuron importance score
            gradenfs.update_importance_scores(grad)

            # update the mask and then update the network topology after every batch 
            if args.batch_update:
                gradenfs.update_mask(batch_idx, epoch_idx)
                gradenfs.apply_mask()

        # save the loss and validation accuracy after each epoch
        losses.append(loss.detach().item())
        model.eval()
        valid_accuracy = evaluate_mlp(model, validation_loader, args.device, args.num_validation)
        valid_accuracies.append(valid_accuracy)
        # log the loss and validation accuracy after each epoch
        info = 'Epoch {} Loss: {:.6f} Validation Accuracy: {:.6f}'.format(
            epoch_idx, losses[-1], valid_accuracy)
        logger.info(info)

        # update the mask and then update the network topology after every epoch if it's not batch update 
        if not args.batch_update:
            gradenfs.update_mask(batch_idx, epoch_idx)
            gradenfs.apply_mask()

    # end training and return recorded data
    return losses, valid_accuracies

# function shows setup and start one training run
def repeat(train_loader, validation_loader, test_loader, args, logger, evaluation_model, repeat_idx):
    # get model
    mlp = MLP(args.input_dim, args.hidden_dim, args.output_dim)

    # prepare for training, get optimizer and get loss function
    optimizer = optim.Adam(mlp.parameters(), lr=args.lr)
    loss_function = nn.CrossEntropyLoss()

    # get GradEnFS algorithm
    gradenfs = GradEnFS(mlp, args, logger)

    # start training
    losses, valid_accuracies = train(mlp, train_loader, validation_loader, optimizer, loss_function, gradenfs, args, logger)

    # get the test accuracy of the final trained mlp
    test_accuracy = evaluate_mlp(mlp, test_loader, args.device, args.num_testing)
    logger.info('Test accuracy of the trained sparse neural network: {}'.format(test_accuracy))

    # save the trained model as checkpoint
    if args.save_model:
        file = '{}/{}.pt'.format(args.models_prefix, str(repeat_idx))
        torch.save({
                'neuron_importance': gradenfs.neuron_importance_scores,
                'model_state_dict': mlp.state_dict(),
            }, file)

    # select the features by GradEnFS method
    selected_feature_indexes = gradenfs.select_features()
    # use accuracies to evaluate the quality of the selected feature subset
    logger.info('Evaluation accuracies evaluated on the feature selected by GradEnFS:')
    evaluation_accuracies = evaluation_acc(evaluation_model, args.k_list, selected_feature_indexes, logger)

    # return the training's result for this run
    return losses, valid_accuracies, test_accuracy, evaluation_accuracies, selected_feature_indexes

# main function, we repeat training several times here to get average results and save it
def main():
    # get arguments
    parser = get_parser()
    args = parser.parse_args()

    # make every input to lowercase
    args.network = args.network.lower()
    args.dataset = args.dataset.lower()
    args.evaluation_model = args.evaluation_model.lower()

    # make the k in k_list be in order
    args.k_list.sort()

    # get dataset
    x_train, y_train, x_valid, y_valid, x_test, y_test = get_dataset(args)
    # make data loader
    train_loader = DataLoader(Dataset(x_train, y_train), batch_size=args.training_batch_size)
    validation_loader = DataLoader(Dataset(x_valid, y_valid), batch_size=args.evaluating_batch_size)
    test_loader = DataLoader(Dataset(x_test, y_test), batch_size=args.evaluating_batch_size)

    # get model's dimension and number of samples from data
    args.input_dim = x_train.shape[1]
    args.output_dim = y_train.shape[1]
    args.num_training = x_train.shape[0]
    args.num_validation = x_valid.shape[0]
    args.num_testing = x_test.shape[0]
    args.batch = len(train_loader)

    # make dir for logs, results and models and get prefix
    args.logs_name, args.results_name, args.models_prefix= create_dir(args)

    # get logger
    logger = setup_logger(args)

    # get the evaluation_model
    if args.evaluation_model == 'svm':
        evaluation_model = SVM_Model(x_train, y_train, x_test, y_test)
    elif args.evaluation_model == 'knn':
        evaluation_model = KNN_Model(x_train, y_train, x_test, y_test)
    elif args.evaluation_model == 'extratree':
        evaluation_model = ExtraTree_Model(x_train, y_train, x_test, y_test)
    else:
        print('can not find the corresponding evaluation model.')
        sys.exit()

    # average result for multiple seed (multiple times of training)
    # proposed dynamic method
    avr_losses = np.zeros(args.epoch)
    avr_valid_accuracies = np.zeros(args.epoch)
    avr_test_accuracy = 0
    avr_evaluation_accuracies = np.zeros(len(args.k_list))
    evaluation_accuracies_per_run = []
    features_subset_per_run = []

    
    # repeat the training process for certain times to get average results
    for repeat_idx in range(args.repeat):
        # start a new repeat
        logger.info('*************************************************Repeat {}*************************************************'.format(repeat_idx+1))
        
        # control the pytorch random seeds for network generation
        if args.use_seeds:
            seed_everything(args.seeds[repeat_idx])
        
        # start the training
        losses, valid_accuracies, test_accuracy, evaluation_accuracies, features_subset = repeat(train_loader, validation_loader, test_loader, args, logger, evaluation_model, repeat_idx)

        # record the results for every single run
        avr_losses += np.array(losses)
        avr_valid_accuracies += np.array(valid_accuracies)
        avr_test_accuracy += test_accuracy
        avr_evaluation_accuracies += np.array(evaluation_accuracies)
        evaluation_accuracies_per_run.append(evaluation_accuracies)
        features_subset_per_run.append(features_subset)

    # calculate the average results after all experimental runs end
    avr_losses = (avr_losses/args.repeat).tolist()
    avr_valid_accuracies = (avr_valid_accuracies/args.repeat).tolist()
    avr_test_accuracy = avr_test_accuracy/args.repeat
    avr_evaluation_accuracies = (avr_evaluation_accuracies/args.repeat).tolist()

    # save the results as json file
    json_data = {
        'arguments': {
            'dataset' : args.dataset,
            'epsilon' : args.epsilon,
            'alpha' : args.alpha,
            'beta' : args.beta,
        },
        'results': {
            'avr_losses': avr_losses,
            'avr_valid_accuracies': avr_valid_accuracies,
            'avr_test_accuracy': avr_test_accuracy,
            'avr_evaluation_accuracies': avr_evaluation_accuracies,
            'evaluation_accuracies_per_run': evaluation_accuracies_per_run,
            'features_subset_per_run': features_subset_per_run
        }
    }
    json_object = json.dumps(json_data)
    with open(args.results_name, "w") as outfile:
        outfile.write(json_object)

if __name__ == '__main__':
    main()