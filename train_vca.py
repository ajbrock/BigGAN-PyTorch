


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from models.Amy_IntermediateRoad import Amy_IntermediateRoad
from simple_utils import load_checkpoint


# Local imports
import utils
import train_fns


def run(config):
    # Update the config dict as necessary
    # This is for convenience, to add settings derived from the user-specified
    # configuration into the config-dict (e.g. inferring the number of classes
    # and size of the images from the dataset, passing in a pytorch object
    # for the activation specified as a string)
    config['resolution'] = utils.imsize_dict[config['dataset']]
    config['n_classes'] = utils.nclass_dict[config['dataset']]
    config['G_activation'] = utils.activation_dict[config['G_nl']]
    config['D_activation'] = utils.activation_dict[config['D_nl']]
    # By default, skip init if resuming training.
    if config['resume']:
        print('Skipping initialization for training resumption...')
        config['skip_init'] = True
    config = utils.update_config_roots(config)
    device = 'cuda'

    # Seed RNG
    utils.seed_rng(config['seed'])

    # Prepare root folders if necessary
    utils.prepare_root(config)

    # Setup cudnn.benchmark for free speed
    torch.backends.cudnn.benchmark = True

    # Import the model--this line allows us to dynamically select different files.
    model = __import__(config['model'])
    experiment_name = (config['experiment_name'] if config['experiment_name']
                        else utils.name_from_config(config))
    print('Experiment name is %s' % experiment_name)


    # Next, build the model
    G = model.Generator(**config).to(device)
    
    print(G)
    print('Number of params in G: {}'.format(sum([p.data.nelement() for p in G.parameters()])))

    state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num': 0,
                'best_IS': 0, 'best_FID': 999999, 'config': config}

    
    print('Loading weights...')
    utils.load_weights(G,None, state_dict, './pretrained', '138k')
    
    train_metrics_fname = '%s/%s' % (config['logs_root'], experiment_name)
    print('Training Metrics will be saved to {}'.format(train_metrics_fname))
    train_log = utils.MyLogger(train_metrics_fname, 
                             reinitialize=(not config['resume']),
                             logstyle=config['logstyle'])
    # Write metadata
    utils.write_metadata(config['logs_root'], experiment_name, config, state_dict)


    # Prepare noise and randomly sampled label arrays
    # Allow for different batch sizes in G
    G_batch_size = max(config['G_batch_size'], config['batch_size'])
    z_, y_ = utils.prepare_z_y(G_batch_size, G.dim_z, config['n_classes'],
                                device=device, fp16=config['G_fp16'])
    # Prepare a fixed z & y to see individual sample evolution throghout training
    fixed_z, fixed_y = utils.prepare_z_y(G_batch_size, G.dim_z,
                                        config['n_classes'], device=device,
                                        fp16=config['G_fp16'])  
    fixed_z.sample_()
    fixed_y.sample_()

    if config['which_train_fn'] == 'VCA_G':
        VCA = Amy_IntermediateRoad( lowfea_VGGlayer=10, highfea_VGGlayer=36, is_highroad_only=False, is_gist=False)
        VCA = load_checkpoint(VCA, config['vca_filepath'])
        VCA = VCA.to(device)

        train = train_fns.VCA_generator_training_function(G, VCA, z_, y_, config)
    # Else, assume debugging and use the dummy train fn
    else:
        train = train_fns.dummy_training_function()


    print('Beginning training at epoch %d...' % state_dict['epoch'])
    # Train for specified number of epochs, although we mostly track G iterations
    for epoch in range(state_dict['epoch'], config['num_epochs']):
        print(epoch)
        for i in range(config['iters_per_epoch']):
            state_dict['itr'] += 1

            G.train()

            metrics = train(x=None, y=None)
            train_log.log(itr=int(state_dict['itr']))


            if not (state_dict['itr'] % config['save_every']):
                if config['G_eval_mode']:
                    G.eval()
                print(metrics)
        
        train_fns.save_and_sample(G, None, None, z_, y_, fixed_z, fixed_y,
                                     state_dict, config, experiment_name)

        state_dict['epoch'] += 1
        



def main():
    parser = utils.prepare_parser()
    config = vars(parser.parse_args())

    print(config)
    run(config)

if __name__ == '__main__':
    main()