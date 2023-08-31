# import lib
import torch
import random

# Dynamic sparse training algorithm
class GradEnFS():
    def __init__(self, model, args, logger):
        self.logger = logger
        self.model = model
        self.device = args.device

        # algorithm's main hyperparameter
        self.epsilon = args.epsilon # epsilon use to control the sparsity level P(W_n) = epsilon(n+n_prv)/(n*n_prv)
        self.alpha = args.alpha # alpha is the rate of the updating connections
        self.beta = args.beta # beta is the hyperparameter for calculating the neuron importance score
        self.k_list = args.k_list # list of number of features to be selected
        
        # the training setting
        self.epoch = args.epoch
        self.batch = args.batch # how many training batch per epoch

        # the network we chose, the variable for generating a sparse neural network
        self.network = args.network # dense or sparse
        self.masks = {} # 0-1 masks (torch tensor) to implement sparse topology in pytorch
        self.param_counts = {} # number of params (scalar) in each layer in sparse neural network
        self.densities = {} # number of density (scalar) in each layer in sparse neural network
        
        # initialization
        self.initialize_mask() # initialize the 0-1 masks with ER random graph
        self.neuron_importance_scores = torch.zeros(self.model.input_dim).float().to(self.device) # tensor of importance scores of every input neurons
        
    # function for initializing masks
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

    # function for updating masks
    def update_mask(self, epoch_idx, batch_idx):
        if self.network == 'sparse':
            self.magnitude_removal()
            if epoch_idx == self.epoch and batch_idx == self.batch:
                self.logger.info('Stop random regrow in the last batch {} of the last epoch {}'.format(epoch_idx, batch_idx))
            else:
                self.random_addition()

    # function for applying masks
    def apply_mask(self):
        # apply masks layer by layer to model's weight matrix
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if len(param.shape) > 1:
                    param.data = param.data * self.masks[name]

    # function for connection removal
    def magnitude_removal(self):
        # SET remove the connection based on weight magnitude
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if len(param.shape) > 1:
                    # if the density is 1 for this layer, do nothing and skip
                    if self.densities[name] == 1:
                        continue

                    # if not, we first calculate the number of connections need to be removed
                    removal_count = round(self.param_counts[name] * self.alpha)
                    # get the absolute value of current weight matrix and saved in temporary variable
                    temp = torch.abs(param.data)
                    # if the exist connection's weight is 0, then set it to -1, which means it's the smallest and will be remove first
                    temp[temp == 0] = -1 
                    # after backpropagation the whole weight matirx will change, so we need to times the masks again
                    temp = temp * self.masks[name]
                    # exclude all the non-exist connection
                    temp[temp == 0] = float('inf')
                    # find index of connections to be removed
                    _, ind = torch.topk(temp.flatten(), dim=0, k=removal_count, largest=False)
                    
                    # use index to update masks
                    self.masks[name].flatten().index_fill_(dim=0, index=ind, value=0)

    # function for connection regrow
    def random_addition(self):
        # SET add the connections randomly layer by layer
        for name, mask in self.masks.items():
            # if the density is 1 for this layer, do nothing and skip
            if self.densities[name] == 1:
                continue

            # if not, we first calculate the number of connections need to be added, it should be same value with the count of removal
            regrow_count = round(self.param_counts[name] * self.alpha)

            # find the index of non-exist connections
            ind = (mask.flatten() == 0).nonzero().flatten().tolist()

            # randomly choose some non-exist connections and set them to 1 in masks
            # use random sample so it will not be affacted by pytorch fixed random seed
            # the selected ind should be turned to tensor and put in the same device
            ind = torch.tensor(random.sample(ind, regrow_count)).type(torch.int64).to(self.device)
            
            # use index to update masks
            mask.flatten().index_fill_(dim=0, index=ind, value=1)
            
    # function for updating neuron importances
    def update_importance_scores(self, grad):
        self.neuron_importance_scores = self.beta * self.neuron_importance_scores + grad

    # function for selecting informative features
    def select_features(self):
        _, indexes = torch.topk(self.neuron_importance_scores, dim=0, k=self.k_list[-1], largest=True)
        return indexes.tolist()