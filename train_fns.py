#train_fns.py
# Functions for the main loop of training different conditional image models
import torch
import utils
import losses
def GAN_training_function(G, D, GD, z_, y_, ema, state_dict, config):
  def train(x, y):
    G.optim.zero_grad()
    D.optim.zero_grad()
    # How many chunks to split x and y into?
    x = torch.split(x, config['batch_size'])
    y = torch.split(y, config['batch_size'])
    counter = 0
    
    # Uncomment this to optionally toggle D's "require_grad"
    # for param in D.parameters():
      # param.requires_grad = True
    
    # Uncomment this to optionally toggle G's "require_grad"
    # for param in G.parameters():
      # param.requires_grad = False
      
    for step_index in range(config['num_D_steps']):
      for accumulation_index in range(config['num_D_accumulations']):
        z_.normal_()
        y_.random_(0, utils.nclass_dict[config['dataset']])
        
        # Uncomment this to generate a new z_ and y_ at every iteration,
        # instead of overwriting them as above.
        # z_ = torch.randn(G_batch_size, G.dim_z, requires_grad=False).cuda()
        # y_ = torch.randint(low=0, high=utils.nclass_dict[config['dataset']], 
                     # size=(G_batch_size,), device='cuda', 
                     # dtype=torch.int64, requires_grad=False)
                     
        # Optionally gather some G gradients, for forward compatibility with 
        # research options
        #train_G = (config['merge_step'] and (step_index +1 == config['num_D_steps']))
        D_fake, D_real = GD(z_[:config['batch_size']], y_[:config['batch_size']], 
                            x[counter], y[counter], train_G=False, split_D=config['split_D'])# train_G)
         
        # By default, use hinge loss
        D_loss_real, D_loss_fake = losses.discriminator_loss(D_fake, D_real)
        D_loss = (D_loss_real + D_loss_fake) / float(config['num_D_accumulations'])
        D_loss.backward()#retain_graph=train_G) # Optionally retain graph for G gradients
        counter += 1
        
      # Optionally apply ortho reg in D
      if config['D_ortho'] > 0.0:
        print('using modified ortho reg in D') # Debug print to indicate we're using ortho reg in D
        utils.ortho(D, config['D_ortho'])
      
      D.optim.step()
      
    # Uncomment this to optionally toggle D's "require_grad"
    # for param in D.parameters():
      # param.requires_grad = False
    # Uncomment this to optionally toggle G's "require_grad"
    # for param in G.parameters():
      # param.requires_grad = True
      
    # Zero G's gradients by default before training G; 
    # they should already be zero, but this is just for safety atm.
    G.optim.zero_grad()
    
    for accumulation_index in range(config['num_G_accumulations']):    
      z_.normal_()
      y_.random_(0, utils.nclass_dict[config['dataset']])
      
      # Uncomment this to generate a new z_ and y_ at every iteration,
      # instead of overwriting them as above.
      # z_ = torch.randn(G_batch_size, G.dim_z, requires_grad=False).cuda()
      # y_ = torch.randint(low=0, high=utils.nclass_dict[config['dataset']], 
                     # size=(G_batch_size,), device='cuda', 
                     # dtype=torch.int64, requires_grad=False)
                     
      D_fake = GD(z_, y_, train_G=True, split_D=config['split_D'])
      G_loss = losses.generator_loss(D_fake) / float(config['num_G_accumulations'])
      G_loss.backward()
    
    # Optionally apply modified ortho reg in G
    if config['G_ortho'] > 0.0:
      print('using modified ortho reg in G') # Debug print to indicate we're using ortho reg in G
      # Don't ortho reg shared, it makes no sense. Really we should blacklist any embeddings for this
      utils.ortho(G, config['G_ortho'], 
                  blacklist=[param for param in G.shared.parameters()])
    G.optim.step()
    
    # If we have an ema, update it, regardless of if we're testing with it or not
    if config['ema']:
      ema.update(state_dict['itr'])
    
    # Return G's loss and the components of D's loss.
    return {'G_loss': float(G_loss.cpu()), 
            'D_loss_real': float(D_loss_real.cpu()),
            'D_loss_fake': float(D_loss_fake.cpu())}
  return train