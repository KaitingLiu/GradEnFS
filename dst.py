# import lib
import torch
import random

# Dynamic sparse training algorithm
class DST():
    def __init__(self, model, args, logger, svm_model):
        # logger and model
        self.logger = logger
        self.model = model
        self.device = args.device

        # the network we chose
        self.network = args.network

        # the training setting
        self.batch = args.batch
        self.epoch = args.epoch

        # arguments for generating sparse network, and store the setting
        self.epsilon = args.epsilon # epsilon use to control the sparsity level P(W_n) = epsilon(n+n_prv)/(n*n_prv)
        self.alpha = args.alpha # alpha is the rate of updating connection
        self.beta = args.beta # beta is the hyperparameter for calculating the neuron importance score
        self.masks = {} # 0-1 masks (torch tensor) to implement sparse topology in pytorch
        self.param_counts = {} # number of params (scalar) in each layer in sparse neural network
        self.densities = {} # number of density (scalar) in each layer in sparse neural network
        self.initialize_mask() # initialize the 0-1 masks with ER random graph

        # argument for calculating and storing neruon importance scores.
        self.importance_scores = self.initialize_importance_scores() # tensor of importance scores of every input neurons
        

        # list of number of features we want to use, and evalueate the neuron importance scores
        self.k_list = args.k_list
        self.svm = svm_model
        
    # the function for dealing with masks
    def initialize_mask(self):
        self.logger.info('Neural Network:')
        num_params_sparse = 0 # total number of parameters of sparse network
        num_params_dense = 0 # total number of parameters of dense counterpart

        # initialize masks layer by layer
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if len(param.shape) > 1:

                    # to calculate the density of each layer
                    n = param.shape[0] # number of nueron in this layer
                    n_prv = param.shape[1] # number of neuron in previous layer
                    current_layer_density = self.epsilon * (n + n_prv) / (n * n_prv) # ER random graph
                    self.densities[name] = 1 if current_layer_density > 1 else current_layer_density

                    # initialize masks of each layer
                    if self.network == 'dense':
                        self.masks[name] = (torch.ones(param.shape)).float().detach().to(self.device)
                    else:
                        self.masks[name] = (torch.rand(param.shape) < current_layer_density).float().detach().to(self.device)
                    self.param_counts[name] = torch.sum(self.masks[name]).item() # save the param count for this layer
                    
                    # calculate the total number of params for both dense and sparse network
                    num_params_dense += n * n_prv
                    num_params_sparse += self.param_counts[name]
                    
                    # log the data for this layer
                    info = '{} number of parameters(dense counterpart):{:.0f} number of parameters(network topology):{:.0f} density:{:.6f} sparsity:{:.6f}'.format(name, n * n_prv, self.param_counts[name], self.densities[name], 1 - self.densities[name])
                    self.logger.info(info)

        # log the overall sparsity and density
        overall_density = num_params_sparse/num_params_dense
        info = 'the entire network number of parameters(dense counterpart):{:.0f} number of parameters(network topology):{:.0f} overall density:{:.6f} overall sparsity:{:.6f}'.format(
            num_params_dense, num_params_sparse, overall_density, 1 - overall_density)
        self.logger.info(info)

    def update_mask(self, epoch_idx, batch_idx):
        if self.network == 'sparse':
            self.magnitude_removal()
            if epoch_idx == self.epoch and batch_idx == self.batch:
                self.logger.info('Stop random regrow in the last batch {} of the last epoch {}'.format(epoch_idx, batch_idx))
            else:
                self.random_addition()
            # self.logger.info('Updating the sparse neural network topology')
                    
            

    def apply_mask(self):
        # apply masks layer by layer to model's weight matrix
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if len(param.shape) > 1:
                    param.data = param.data * self.masks[name]

    # set
    def magnitude_removal(self):
        # SET remove the connection based on weight magnitude, so need to loop with model param(need to use weight matrix)
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if len(param.shape) > 1:
                    # if the density is 1 for this layer, do nothing and skip
                    if self.densities[name] == 1:
                        continue

                    # first calculate the number of connections need to be removed
                    removal_count = round(self.param_counts[name] * self.alpha)

                    # get the absolute value of current weight matrix and saved as temp
                    temp = torch.abs(param.data)
                    # if the exist connection (weight) is 0, then set it to -1, which means it's the smallest and will be remove first
                    temp[temp == 0] = -1 
                    # weight 0 after backpropagation will become other values (the whole weight matirx will change), so we need to times the masks again
                    temp = temp * self.masks[name]
                    # exclude all the non-exist connection
                    temp[temp == 0] = float('inf')
                    # find index of connections closest to 0 by choosing the smallest abs(connection), and 0 weight will be -1 and be remove first now
                    _, ind = torch.topk(temp.flatten(), dim=0, k=removal_count, largest=False)
                    
                    # use index to update masks
                    self.masks[name].flatten().index_fill_(dim=0, index=ind, value=0)

    def random_addition(self):
        # SET add the connections randomly layer by layer
        for name, mask in self.masks.items():
            # if the density is 1 for this layer, do nothing and skip
            if self.densities[name] == 1:
                continue

            # first calculate the number of connections need to be added, should be same value with the count of removal
            removal_count = round(self.param_counts[name] * self.alpha)

            # find the index of 0 in masks, which is find the index of non-exist connection
            ind = (mask.flatten() == 0).nonzero().flatten().tolist()

            # randomly choose some non-exist connections and set them to 1 in masks
            # use random sample so it will not be affacted by pytorch fixed random seed
            # the selected ind should be turned to tensor and put in the same device
            ind = torch.tensor(random.sample(ind, removal_count)).type(torch.int64).to(self.device)
            mask.flatten().index_fill_(dim=0, index=ind, value=1)
            
    # calculation of neuron importance
    def initialize_importance_scores(self):
        self.logger.info('successfully initialize neuron importance scores for every input neurons.')
        return torch.zeros(self.model.input_dim).float().to(self.device)
        
    def update_importance_scores(self, grad):
        self.importance_scores = self.beta * self.importance_scores + grad

    # evaluate the neuron importance scores by selecting features
    def select_features(self, importance_scores):
        _, indexes = torch.topk(importance_scores, dim=0, k=self.k_list[-1], largest=True)
        indexes = indexes.tolist()
        return indexes

    def svm_acc(self, indexes):
        svm_accuracies = []
        for k in self.k_list:
            # get the indexes of the input neuron with the largest neuron importance scores
            k_indexes = indexes[:k]
            # use these k informative features to train svm and get the accuracy
            accuracy = self.svm.train_and_test(k_indexes)
            # save the accuracy and log it
            svm_accuracies.append(accuracy)
            info = 'The SVM accuracy is {} for {} features in {} network'.format(accuracy, k, self.network)
            self.logger.info(info)
        # return the accuracies for every k features in svm, and the longest inforamtive indexes
        return svm_accuracies

    # evaluate the neuron importance scores by selecting features
    def hit_rate(self, importance_scores):
        k = self.k_list[-1]
        _, indexes = torch.topk(importance_scores, dim=0, k=k, largest=True)
        hit = len(set(indexes.tolist()).intersection(range(k)))
        rate = hit/k
        info = 'The hit rate of artificial data set is {} in {} network'.format(rate, self.network)
        self.logger.info(info)
        return rate
    
    # QS implementation
    def QS_importance_scores(self):
        # summation of the weight magnitude of the input layer as the neuron importance for QS algorithm
        QS_importance_scores = None
        for _, param in self.model.named_parameters():
            if param.requires_grad:
                if len(param.shape) > 1:
                    QS_importance_scores = torch.sum(torch.abs(param.data), dim=0)
                    break
        return QS_importance_scores