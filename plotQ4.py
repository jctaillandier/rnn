import matplotlib.pyplot as plt
import numpy as np
import os
import ast

#parser.add_argument('--save_dir', type=str, default='',
#                    help='path to save the experimental config, logs, model \
#                    This is automatically generated based on the command line \
#                    arguments you pass and only needs to be set if you want a \
#                    custom dir name')


def plot_model(dir, plot_title):

    epochs, wall_times, train_ppls, val_ppls = get_modeldata(dir)

    fig = plt.figure()
    #fig.suptitle('bold figure suptitle', fontsize=14, fontweight='bold')
    fig.suptitle(plot_title)


    #title = '{0}: {1}'.format(architecture, optimizer)
    
    ax = fig.add_subplot(121)
    #ax.set_title(plot_title)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('PPL')

    ax.plot(epochs, train_ppls,  epochs, val_ppls)
    ax.legend(['Training', 'Validation'])
    
    ax = fig.add_subplot(122)
    #ax.set_title(plot_title)

    ax.set_xlabel('Wall Time (hours)')
    ax.set_ylabel('PPL')

    ax.plot(wall_times, train_ppls,  wall_times, val_ppls)
    ax.legend(['Training', 'Validation'])


    fig.set_size_inches(8.5, 4)
    plt.savefig(plot_title.replace('.','_'))
    plt.close()
    #plt.show()


def plot_valid(experiments, plot_title):
    
    fig = plt.figure()

    #fig.suptitle('bold figure suptitle', fontsize=14, fontweight='bold')
    fig.suptitle(plot_title)
    
    ax1 = fig.add_subplot(211)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('PPL (log scale)')
    ax1.set_yscale('log')
    
    ax2 = fig.add_subplot(212)
    ax2.set_xlabel('Wall Time (hours)')
    ax2.set_ylabel('PPL (log scale)')
    ax2.set_yscale('log')

    for i in experiments.keys():
        dir = experiments[i]['dir']
        epochs, wall_times, _, val_ppls = get_modeldata(dir)
        caption = experiments[i]['title']
        caption = caption.replace(plot_title + ', ', '')
        ax1.plot(epochs, val_ppls, label = caption)
        ax1.legend()
    
        ax2.plot(wall_times, val_ppls, label = caption)
        ax2.legend()
    
    fig.set_size_inches(8.5, 11)
    plt.savefig(plot_title)
    plt.close()
    

def get_modeldata(dir):

    lc_path = os.path.join(dir, 'learning_curves.npy')
    x = np.load(lc_path)[()]
    
    train_ppls = x['train_ppls']
    val_ppls = x['val_ppls']
    epochs = np.arange(len(train_ppls))

    log_path = os.path.join(dir, 'log.txt')
    wall_times = []
    with open(log_path, 'rt') as file:
        wall_time = 0
        for line in file:
            temp = line.split()
            wall_time += float(temp[len(temp)-1])
            wall_times.append(wall_time)
    
    wall_times = np.array(wall_times, dtype=float)
    wall_times = wall_times / 3600
    return epochs, wall_times, train_ppls, val_ppls






with open("experiments.txt", "r") as data:
    experiments = ast.literal_eval(data.read())

for i in experiments.keys():
    print(i, experiments[i]['dir'], experiments[i]['title'])
    plot_model(experiments[i]['dir'],experiments[i]['title'])

optimizers = ['ADAM', 'SGD', 'SGD_SCHED']

for i in optimizers:
    with open(i + ".txt", "r") as data:
        experiments = ast.literal_eval(data.read())

    plot_valid(experiments, i)

architectures = ['RNN', 'GRU', 'TRANSFORMER']

for i in architectures:  
    with open(i + ".txt", "r") as data:
        experiments = ast.literal_eval(data.read())

    plot_valid(experiments, i)

