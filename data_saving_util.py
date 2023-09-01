import os

# this function use to make the dir for saving data, and return the path of these dir
def create_dir(args):
    # making dir for logs
    logs_path = './gradenfs_logs'
    if not os.path.exists(logs_path): os.mkdir(logs_path)
    logs_path = logs_path + '/' + args.dataset
    if not os.path.exists(logs_path): os.mkdir(logs_path)
    logs_path = logs_path + '/' + args.network
    if not os.path.exists(logs_path): os.mkdir(logs_path)
    logs_path = '{}/{}_{}_{}'.format(logs_path, str(args.epsilon), str(args.alpha), str(args.beta))

    # making dir for results
    results_path = './gradenfs_results'
    if not os.path.exists(results_path): os.mkdir(results_path)
    results_path = results_path + '/' + args.dataset
    if not os.path.exists(results_path): os.mkdir(results_path)
    results_path = results_path + '/' + args.network
    if not os.path.exists(results_path): os.mkdir(results_path)
    results_path = '{}/{}_{}_{}'.format(results_path, str(args.epsilon), str(args.alpha), str(args.beta))

    logs_name = '{}.log'.format(logs_path)
    results_name = '{}.json'.format(results_path)

    return logs_name, results_name
