# import lib
import os

from dst import DST
from argparser import get_parser
import logging
import sys
import torch
from data_loading_util import get_artificial_data, get_dataset, Dataset
from data_saving_util import create_dir
from torch.utils.data import DataLoader

from model import MLP
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import json
import numpy as np
import math

from SVM_Model import SVM_Model

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

# function to control the random seed for generating the same initialized sparse network
def seed_everything(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = True

# function which use test set/validation set to evaluate the model's accuracy
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

# function shows how to train the model
def train(model, train_loader, validation_loader, optimizer, loss_function, dst_algorithm, args, logger, repeat_idx):
    # data saved for analysis
    # proposed dynamic method
    loss_per_epoch = []
    valid_accuracy_per_epoch = []
    dynamic_feature_indexes_per_epoch = []
    # static mehtod
    static_importance_scores = torch.zeros(model.input_dim).float().to(args.device)
    # QS method
    QS_feature_indexes_per_epoch = []

    # generate the sparse network
    model.to(args.device)
    dst_algorithm.apply_mask()

    for epoch_idx in range(1, args.epoch+1):
        # start training
        model.train()
        for batch_idx, (x, y) in enumerate(train_loader, start=1):
            x, y = x.to(args.device), y.to(args.device)
            x.requires_grad = True

            # feed forward, calculate the loss
            optimizer.zero_grad()
            y_hat = model(x)
            loss = loss_function(y_hat, y)

            # backpropagate, get the gradient info we want, and update the network's weights
            loss.backward() # backpropagate
            grad = torch.mean(torch.abs(x.grad), dim=0) # get the gradient of loss with every input neuron
            optimizer.step() # do the optimization

            # update the neuron importance score for all network
            dst_algorithm.update_importance_scores(grad)

            # check if we update mask every batch
            if args.batch_update:
                dst_algorithm.update_mask(epoch_idx, batch_idx)
            # for all dst algorithm, even we don't update the mask, after backprop we still need to apply mask
            dst_algorithm.apply_mask()
        
        # if the update frequency is not after every batch, we update it after every epoch
        if not args.batch_update:
            dst_algorithm.update_mask(epoch_idx, batch_idx)
        # for all dst algorithm, even we don't update the mask, after backprop we still need to apply mask
        dst_algorithm.apply_mask()

        # save some results after each epoch
        loss_per_epoch.append(loss.detach().item())
        model.eval()
        valid_accuracy = evaluate_mlp(model, validation_loader, args.device, args.num_validation)
        valid_accuracy_per_epoch.append(valid_accuracy)
        # also log the results and show it in console
        info = 'Epoch {} Loss: {:.6f} Validation Accuracy: {:.6f}'.format(
            epoch_idx, loss_per_epoch[-1], valid_accuracy)
        logger.info(info)

        # we get the neuron importance scores for static method
        if epoch_idx == args.epoch-1:
            static_importance_scores = dst_algorithm.importance_scores
        if epoch_idx == args.epoch:
            static_importance_scores = dst_algorithm.importance_scores - static_importance_scores

        # select k features and use svm to test it to see how neuron importance works after every epoch
        # save the detail of proposed method and QS method
        dynamic_features_indexes = dst_algorithm.select_features(dst_algorithm.importance_scores)
        dynamic_feature_indexes_per_epoch.append(dynamic_features_indexes)
        QS_importance_scores = dst_algorithm.QS_importance_scores()
        QS_features_indexes = dst_algorithm.select_features(QS_importance_scores)
        QS_feature_indexes_per_epoch.append(QS_features_indexes)
        if args.detail:
            logger.info('Statistic of the proposed dynamic method:')
            dst_algorithm.svm_acc(dynamic_features_indexes)
            logger.info('Statistic of the QS method:')
            dst_algorithm.svm_acc(QS_features_indexes)

    # return data
    return loss_per_epoch, valid_accuracy_per_epoch, static_importance_scores, dynamic_feature_indexes_per_epoch, QS_feature_indexes_per_epoch

# function shows setup and start training for one time
def repeat(train_loader, validation_loader, test_loader, args, logger, svm_model, repeat_idx):
    # get model
    mlp = MLP(args.input_dim, args.hidden_dim, args.output_dim)

    # prepare for training, get optimizer and get loss function
    optimizer = optim.Adam(mlp.parameters(), lr=args.lr)
    loss_function = nn.CrossEntropyLoss()

    # get DST algorithm
    dst_algorithm = DST(mlp, args, logger, svm_model)

    # start training
    loss_per_epoch, valid_accuracy_per_epoch, static_importance_scores, dynamic_feature_indexes_per_epoch, QS_feature_indexes_per_epoch = train(mlp, train_loader, validation_loader, optimizer, loss_function, dst_algorithm, args, logger, repeat_idx)

    # get the test accuracy of the final network, svm accuracies for all method
    test_accuracy = evaluate_mlp(mlp, test_loader, args.device, args.num_testing)
    logger.info('Test accuracy of the final network: {}'.format(test_accuracy))
    logger.info('Statistic of the proposed dynamic method:')
    dynamic_svm_accuracies = dst_algorithm.svm_acc(dynamic_feature_indexes_per_epoch[-1])

    logger.info('Statistic of the static method:')
    static_features_indexes = dst_algorithm.select_features(static_importance_scores)
    static_svm_accuracies = dst_algorithm.svm_acc(static_features_indexes)

    logger.info('Statistic of the QS method:')
    QS_svm_accuracies = dst_algorithm.svm_acc(QS_feature_indexes_per_epoch[-1])

    # return this training's result
    return loss_per_epoch, valid_accuracy_per_epoch, test_accuracy, dynamic_svm_accuracies, static_svm_accuracies, QS_svm_accuracies, dst_algorithm.importance_scores, static_importance_scores, dst_algorithm.QS_importance_scores(), dynamic_feature_indexes_per_epoch, QS_feature_indexes_per_epoch

# main function, we repeat training several times here to get average results and save it
def main():
    # get arguments
    parser = get_parser()
    args = parser.parse_args()

    # make every input to lowercase
    args.network = args.network.lower()
    args.dataset = args.dataset.lower()

    # make the k in k_list be in order
    args.k_list.sort()

    # get dataset
    if args.dataset == 'artificial':
        x_train, y_train, x_valid, y_valid, x_test, y_test = get_artificial_data(args)
    else:
        x_train, y_train, x_valid, y_valid, x_test, y_test = get_dataset(args.dataset)
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
    args.batch = math.ceil(args.num_training / args.training_batch_size)

    # make dir for logs, results and models and get prefix
    args.logs_name, args.results_name = create_dir(args)

    # get logger
    logger = setup_logger(args)

    # get the svm_model
    svm_model = SVM_Model(x_train, y_train, x_test, y_test)

    # average result for multiple seed (multiple times of training)
    # proposed dynamic method
    avr_loss_per_epoch = None
    avr_valid_accuracy_per_epoch = None
    avr_test_accuracy = 0
    avr_dynamic_svm_accuracies = None
    avr_dynamic_importance_scores = None
    # static method
    avr_static_svm_accuracies = None
    avr_static_importance_scores = None
    # QS method
    avr_QS_svm_accuracies = None
    avr_QS_importance_scores = None
    
    # repeat the training process for certain times to get average results
    for repeat_idx in range(args.repeat):
        # start a new repeat
        logger.info('*************************************************Repeat {}*************************************************'.format(repeat_idx+1))
        
        # control the pytorch random seeds for network generation
        if args.use_seeds:
            seed_everything(args.seeds[repeat_idx])
        
        # start the training
        loss_per_epoch, valid_accuracy_per_epoch, test_accuracy, dynamic_svm_accuracies, static_svm_accuracies, QS_svm_accuracies, dynamic_importance_scores, static_importance_scores, QS_importance_scores, dynamic_feature_indexes_per_epoch, QS_feature_indexes_per_epoch = repeat(train_loader, validation_loader, test_loader, args, logger, svm_model, repeat_idx)

        if repeat_idx == 0 :
            # dynamic method
            avr_loss_per_epoch = np.array(loss_per_epoch)
            avr_valid_accuracy_per_epoch = np.array(valid_accuracy_per_epoch)
            avr_dynamic_importance_scores = dynamic_importance_scores
            avr_dynamic_svm_accuracies = np.array(dynamic_svm_accuracies)
            # static method
            avr_static_importance_scores = static_importance_scores
            avr_static_svm_accuracies = np.array(static_svm_accuracies)
            # QS method
            avr_QS_importance_scores = QS_importance_scores
            avr_QS_svm_accuracies = np.array(QS_svm_accuracies)
        else:
            # dynamic method
            avr_loss_per_epoch += np.array(loss_per_epoch)
            avr_valid_accuracy_per_epoch += np.array(valid_accuracy_per_epoch)
            avr_dynamic_importance_scores += dynamic_importance_scores
            avr_dynamic_svm_accuracies += np.array(dynamic_svm_accuracies)
            # static method
            avr_static_importance_scores += static_importance_scores
            avr_static_svm_accuracies += np.array(static_svm_accuracies)
            # QS method
            avr_QS_importance_scores += QS_importance_scores
            avr_QS_svm_accuracies += np.array(QS_svm_accuracies)
        
        avr_test_accuracy += test_accuracy


    # calculate the aaverage results
    # dynamic method
    avr_loss_per_epoch = (avr_loss_per_epoch/args.repeat).tolist()
    avr_valid_accuracy_per_epoch = (avr_valid_accuracy_per_epoch/args.repeat).tolist()
    avr_dynamic_importance_scores = (avr_dynamic_importance_scores/args.repeat).tolist()
    avr_dynamic_svm_accuracies = (avr_dynamic_svm_accuracies/args.repeat).tolist()
    # static method
    avr_static_importance_scores = (avr_static_importance_scores/args.repeat).tolist()
    avr_static_svm_accuracies = (avr_static_svm_accuracies/args.repeat).tolist()
    # QS method
    avr_QS_importance_scores = (avr_QS_importance_scores/args.repeat).tolist()
    avr_QS_svm_accuracies = (avr_QS_svm_accuracies/args.repeat).tolist()
    # model
    avr_test_accuracy = avr_test_accuracy/args.repeat

    # save the results as json file
    json_data = {
        'arguments': {
            'dataset' : args.dataset,
            'epsilon' : args.epsilon,
            'network' : args.network,
            'k_list' : args.k_list,
            'num_training' : args.num_training
        },
        'results': {
            # indexes
            'dynamic_feature_indexes_per_epoch': dynamic_feature_indexes_per_epoch,
            'QS_feature_indexes_per_epoch': QS_feature_indexes_per_epoch,
            # dynamic method
            'avr_loss_per_epoch': avr_loss_per_epoch,
            'avr_valid_accuracy_per_epoch': avr_valid_accuracy_per_epoch,
            'avr_dynamic_importance_scores': avr_dynamic_importance_scores,
            # static method
            'avr_static_importance_scores': avr_static_importance_scores,
            # QS method
            'avr_QS_importance_scores': avr_QS_importance_scores,
            # scalar
            'avr_test_accuracy': avr_test_accuracy,
            'avr_dynamic_svm_accuracies': avr_dynamic_svm_accuracies,
            'avr_static_svm_accuracies': avr_static_svm_accuracies,
            'avr_QS_svm_accuracies': avr_QS_svm_accuracies

        }
    }
    json_object = json.dumps(json_data)
    with open(args.results_name, "w") as outfile:
        outfile.write(json_object)

if __name__ == '__main__':
    main()