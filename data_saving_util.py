import os

# this function use to make the dir for saving data, and return the path of these dir
def create_dir(args):
    # making dir for logs, the final path will be ./conn_logs/dataset/epsilon
    logs_path = './neu_imp_logs'
    if not os.path.exists(logs_path): os.mkdir(logs_path)
    logs_path = logs_path + '/' + args.dataset
    if not os.path.exists(logs_path): os.mkdir(logs_path)
    logs_path = logs_path + '/' + str(args.epsilon)
    if not os.path.exists(logs_path): os.mkdir(logs_path)

    # making dir for results, the final path will be ./conn_results/dataset/epsilon
    results_path = './neu_imp_results'
    if not os.path.exists(results_path): os.mkdir(results_path)
    results_path = results_path + '/' + args.dataset
    if not os.path.exists(results_path): os.mkdir(results_path)
    results_path = results_path + '/' + str(args.epsilon)
    if not os.path.exists(results_path): os.mkdir(results_path)
    
    return logs_path, results_path
