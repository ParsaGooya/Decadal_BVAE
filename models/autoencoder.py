import torch
import torch.nn as nn
import numpy as np
import torch.nn.init as init
import math
import torch.distributions as dist

# functional_derivatives = {
#     torch.tanh: lambda x: 1 - torch.pow(torch.tanh(x), 2),
#     F.leaky_relu: lambda x: (x > 0).type(torch.FloatTensor) + \
#                             (x < 0).type(torch.FloatTensor) * -0.01,
#     F.elu: lambda x: (x > 0).type(torch.FloatTensor) + \
#                      (x < 0).type(torch.FloatTensor) * torch.exp(x)
# }


class Autoencoder(nn.Module):

    def __init__(self, input_dim, encoder_hidden_dims, decoder_hidden_dims=None, added_features_dim=0, append_mode=1, batch_normalization=False, dropout_rate=None, VAE = None, condition_embedding_dims = None, full_conditioning = False, condition_dependant_latent = False, fixed_posterior_variance = None, prior_flow:dict = None, condemb_to_decoder = False, device = 'cpu') -> None:
        super(Autoencoder, self).__init__()
        if VAE is None:
            assert condition_embedding_dims is None, 'condition_embedding_dims is for the cVAE model'
        self.condition_embedding_dims = condition_embedding_dims 
        self.device = device
        self.append_mode = append_mode
        self.full_conditioning = full_conditioning
        self.condition_dependant_latent = condition_dependant_latent
        self.fixed_posterior_variance = fixed_posterior_variance
        self.condemb_to_decoder = condemb_to_decoder
        latent_size = encoder_hidden_dims[-1]
        
        if prior_flow is not None:
            try: 
                prior_flow['args']['dim'] = latent_size
            except: 
                prior_flow['args'] = {}
                prior_flow['args']['dim'] = latent_size
            if self.condition_dependant_latent:
                self.flow = NormalizingFlowModel(flows = [prior_flow['type']( **prior_flow['args'], device = self.device, condition_embedding_size = condition_embedding_dims[-1]) for _ in range(prior_flow['num_layers'])], device = self.device)
            else:
                self.flow = NormalizingFlowModel(flows = [prior_flow['type']( **prior_flow['args'], device = self.device) for _ in range(prior_flow['num_layers'])], device = self.device)

        else:
            self.flow = None
        if fixed_posterior_variance is not None:
           if len(fixed_posterior_variance)>1:
            assert len(fixed_posterior_variance) == latent_size,'Provide one number or one number per latent dimension for the fixed posterior variance...'
           self.fixed_posterior_variance = torch.log(torch.from_numpy(fixed_posterior_variance)).expand(latent_size).to(device)

        if VAE is False:
            self.VAE = None
        else:
            self.VAE = VAE


        if decoder_hidden_dims is None:
            if len(encoder_hidden_dims) == 1:
                decoder_hidden_dims = []
            else:
                decoder_hidden_dims = encoder_hidden_dims[::-1][1:]

        if condition_embedding_dims is not None:
            embedding_size = condition_embedding_dims[-1]
            if condition_dependant_latent is True:
                assert latent_size == embedding_size, 'For condition dependent latent the latent_size must equal embedding_size ...'
            if condemb_to_decoder:
                latent_size = latent_size + embedding_size
            self.embedding_size = embedding_size

        if (append_mode == 2) or (append_mode == 3):
            decoder_dims = [latent_size + added_features_dim, *decoder_hidden_dims, input_dim]
        else:
            decoder_dims = [latent_size, *decoder_hidden_dims, input_dim]
            
        if condition_embedding_dims is not None:
            condition_embedding_dims = [input_dim, *condition_embedding_dims]
            layers = []
            for i in range(len(condition_embedding_dims) - 2):
                layers.append(nn.Linear(condition_embedding_dims[i], condition_embedding_dims[i + 1]))
                layers.append(nn.ReLU())
                if dropout_rate is not None:
                    layers.append(nn.Dropout(dropout_rate))
                if batch_normalization:
                    layers.append(nn.BatchNorm1d(condition_embedding_dims[i + 1]))
            if condition_dependant_latent:
                if self.flow is None:
                    self.condition_mu = nn.Linear(condition_embedding_dims[-2], embedding_size)
                    self.condition_log_var = nn.Linear(condition_embedding_dims[-2], embedding_size)
                if any([condemb_to_decoder, self.flow is not None]):
                    self.condition_embedding = nn.Linear(condition_embedding_dims[-2], embedding_size)
            else:
                layers.append(nn.Linear(condition_embedding_dims[-2], embedding_size))
            if condemb_to_decoder:
                latent_size = latent_size - embedding_size
            # layers.append(nn.ReLU())
            # if batch_normalization:
            #     layers.append(nn.BatchNorm1d(embedding_size))
            self.embedding = nn.Sequential(*layers)
            
            if full_conditioning:
                input_dim = 2 * input_dim 
            else:
                input_dim = input_dim + embedding_size

        if self.VAE is not None:
            
            if (append_mode == 1) or (append_mode == 3):
                encoder_dims = [input_dim + added_features_dim, *encoder_hidden_dims[:-1]]
            else:
                encoder_dims = [input_dim, *encoder_hidden_dims[:-1]]
        else:
            if (append_mode == 1) or (append_mode == 3):
                encoder_dims = [input_dim + added_features_dim, *encoder_hidden_dims]
            else:
                encoder_dims = [input_dim, *encoder_hidden_dims]


        layers = []
        for i in range(len(encoder_dims) - 1):
            layers.append(nn.Linear(encoder_dims[i], encoder_dims[i + 1]))
            layers.append(nn.ReLU())
            if dropout_rate is not None:
                layers.append(nn.Dropout(dropout_rate))
            if batch_normalization:
                layers.append(nn.BatchNorm1d(encoder_dims[i + 1]))
        self.encoder = nn.Sequential(*layers)

        if self.VAE is not None:
            self.mu = nn.Linear(encoder_dims[-1], latent_size)
            if self.fixed_posterior_variance is None:
                self.log_var = nn.Linear(encoder_dims[-1], latent_size)
            self.N = torch.distributions.Normal(0, 1)
            # Get sampling working on GPU
            if device.type == 'cuda':
                self.N.loc = self.N.loc.cuda()
                self.N.scale = self.N.scale.cuda()


        layers = []
        for i in range(len(decoder_dims) - 1):
            layers.append(nn.Linear(decoder_dims[i], decoder_dims[i + 1]))
            if i <= (len(decoder_dims) - 3):
                layers.append(nn.ReLU())
                if dropout_rate is not None:
                    layers.append(nn.Dropout(dropout_rate))
                if batch_normalization:
                    layers.append(nn.BatchNorm1d(decoder_dims[i + 1]))
        self.decoder = nn.Sequential(*layers)
        
        self.latent_size = latent_size
        if self.VAE is not None:
            self.encoder.apply(weights_init)
            self.decoder.apply(weights_init)
            self.mu.apply(weights_init)
            if self.fixed_posterior_variance is None:
                self.log_var.apply(weights_init)
            if condition_embedding_dims is not None:
                self.embedding.apply(weights_init)
                if condition_dependant_latent:
                    if self.flow is None:
                        self.condition_mu.apply(weights_init)
                        self.condition_log_var.apply(weights_init)
                    else:
                        self.flow.apply(weights_init)
                    try:
                        self.condition_embedding.apply(weights_init)
                    except:
                        pass

    def forward(self, x, condition = None, sample_size = 1, seed = None, nstd = 1):
        

        if (type(x) == list) or (type(x) == tuple):
            input_shape = x[0].shape
            x_in = x[0].flatten(start_dim=1)
        else:
            input_shape = x.size()
            x_in = x.flatten(start_dim=1)
        
        if self.VAE is not None:
            input_shape = (sample_size, *input_shape)
        
        if self.condition_embedding_dims is not None:
            cond_in = self.embedding(condition.flatten(start_dim=1))
            if self.condition_dependant_latent:
                if self.flow is None:
                    cond_in_mu = self.condition_mu(cond_in)
                    cond_in_log_var =self.condition_log_var(cond_in)
                if any([self.condemb_to_decoder, self.flow is not None]):
                    cond_emb = self.condition_embedding(cond_in)
                    cond_in = cond_emb.unsqueeze(-2).expand(cond_emb.shape[0], int(x_in.shape[0]/cond_emb.shape[0]), self.embedding_size)
                    cond_in = torch.flatten(cond_in, start_dim = 0, end_dim = 1)
            else:
                cond_in = cond_in.unsqueeze(-2).expand(cond_in.shape[0], int(x_in.shape[0]/cond_in.shape[0]), self.embedding_size)
                cond_in = torch.flatten(cond_in, start_dim = 0, end_dim = 1)
            if self.full_conditioning:
                full_cond = condition.expand(cond_in.shape[0], int(x_in.shape[0]/cond_in.shape[0]), x_in.shape[-1])
                full_cond = torch.flatten(full_cond, start_dim = 0, end_dim = 1)
                x_in = torch.cat([x_in, full_cond], dim=1)
            else:
                x_in = torch.cat([x_in, cond_in], dim=1)

            cond_in = cond_in.unsqueeze(0).expand(sample_size, *cond_in.shape)
            cond_in = torch.flatten(cond_in, start_dim = 0, end_dim = 1)
        
        if (type(x) == list) or (type(x) == tuple):

            if self.append_mode == 1: # append at encoder
                out = self.encoder(torch.cat([x_in, x[1]], dim=1))
                if self.VAE is not None:
                    mu = self.mu(out)
                    log_var =self.log_var(out) if self.fixed_posterior_variance is None else self.fixed_posterior_variance.unsqueeze(0).expand_as(mu).type_as(mu)
                    out = self.sample( mu, log_var, sample_size, seed, nstd = nstd)
                    out_shape = out.shape
                    if all([self.condition_embedding_dims is not None, self.condemb_to_decoder]):
                        out = torch.cat([torch.flatten(out, start_dim = 0, end_dim = 1), cond_in], dim = -1)
                    else: 
                        out = torch.flatten(out, start_dim = 0, end_dim = 1)
                    out = self.decoder(out)
                    out = torch.unflatten(out, dim = 0 , sizes = out_shape[0:2])
                    # out.view(out_shape[0] , out_shape[1], -1 )
                else:
                    out = self.decoder(out)

            elif self.append_mode == 2: # append at decoder
                out = self.encoder(x_in)
                if self.VAE is not None:
                    mu = self.mu(out)
                    log_var = self.log_var(out) if self.fixed_posterior_variance is None else self.fixed_posterior_variance.unsqueeze(0).expand_as(mu).type_as(mu)
                    out = self.sample(mu, log_var, sample_size, seed, nstd = nstd)
                    out = torch.cat([out, x[1].unsqueeze(0).expand((sample_size, *x[1].shape))], dim=2)
                    out_shape = out.shape
                    if all([self.condition_embedding_dims is not None, self.condemb_to_decoder]):
                        out = torch.cat([torch.flatten(out, start_dim = 0, end_dim = 1), cond_in], dim = -1)
                    else: 
                        out = torch.flatten(out, start_dim = 0, end_dim = 1)
                    out = self.decoder(out)
                    out = torch.unflatten(out, dim = 0 , sizes = out_shape[0:2])
                    #out.view(out_shape[0] , out_shape[1], -1 )
                else:
                    out = self.decoder(torch.cat([out, x[1]], dim=1))

            elif self.append_mode == 3: # append at encoder and decoder
                out = self.encoder(torch.cat([x_in, x[1]], dim=1))
                if self.VAE is not None:
                    mu = self.mu(out)
                    log_var = self.log_var(out) if self.fixed_posterior_variance is None else self.fixed_posterior_variance.unsqueeze(0).expand_as(mu).type_as(mu)
                    out = self.sample(mu, log_var, sample_size, seed, nstd = nstd)
                    out = torch.cat([out, x[1].unsqueeze(0).expand((sample_size, *x[1].shape))], dim=2)
                    out_shape = out.shape
                    if all([self.condition_embedding_dims is not None, self.condemb_to_decoder]):
                        out = torch.cat([torch.flatten(out, start_dim = 0, end_dim = 1), cond_in], dim = -1)
                    else: 
                        out = torch.flatten(out, start_dim = 0, end_dim = 1) 
                    out = self.decoder(out)
                    out = torch.unflatten(out, dim = 0 , sizes = out_shape[0:2])
                    #out.view(out_shape[0] , out_shape[1], -1 )
                else:
                    out = self.decoder(torch.cat([out, x[1]], dim=1))

        else:

            out = self.encoder(x_in)

            if self.VAE is not None:
                    mu = self.mu(out)
                    log_var = self.log_var(out) if self.fixed_posterior_variance is None else self.fixed_posterior_variance.unsqueeze(0).expand_as(mu).type_as(mu)
                    out = self.sample(mu, log_var, sample_size, seed, nstd = nstd)
                    out_shape = out.shape
                    if all([self.condition_embedding_dims is not None, self.condemb_to_decoder]):
                        out = torch.cat([torch.flatten(out, start_dim = 0, end_dim = 1), cond_in], dim = -1)
                    else: 
                        out = torch.flatten(out, start_dim = 0, end_dim = 1) 
                    out = self.decoder(out)
                    out = torch.unflatten(out, dim = 0 , sizes = out_shape[0:2])
                    #out.view(out_shape[0] , out_shape[1], -1 )
            else:
                out = self.decoder(out)

        if self.VAE is not None:
            if self.condition_dependant_latent:
                if self.flow is None:
                    return out.view(input_shape), mu, log_var, cond_in_mu, cond_in_log_var
                else:
                     return out.view(input_shape), mu, log_var, cond_emb
            else:
                return out.view(input_shape), mu, log_var
        else:
            return out.view(input_shape)
    
    def sample( self, mu, log_var, sample_size = 1, seed = None, nstd = 1):
        if seed is not None:
            current_rng_state = torch.random.get_rng_state()
            torch.manual_seed(seed)

        var = torch.exp(log_var) + 1e-4

        if nstd !=1:
            N = torch.distributions.Normal(0, nstd)
        # Get sampling working on GPU
            if self.device.type == 'cuda':
                N.loc = N.loc.cuda()
                N.scale = N.scale.cuda()
            out = mu + torch.sqrt(var)*N.sample((sample_size,*mu.shape))
        else:
            out = mu + torch.sqrt(var)*self.N.sample((sample_size,*mu.shape))
        
        if seed is not None:
            torch.random.set_rng_state(current_rng_state)
        
        return out
    



class Autoencoder_decoupled(nn.Module):

    def __init__(self, input_dim, encoder_hidden_dims, decoder_hidden_dims=None, added_features_dim=0, append_mode=1, batch_normalization=False, dropout_rate=None) -> None:
        super(Autoencoder_decoupled, self).__init__()
        self.append_mode = append_mode
        if (append_mode == 1) or (append_mode == 3):
            encoder_dims = [input_dim + added_features_dim, *encoder_hidden_dims]
        else:
            encoder_dims = [input_dim, *encoder_hidden_dims]

        layers = []
        for i in range(len(encoder_dims) - 1):
            layers.append(nn.Linear(encoder_dims[i], encoder_dims[i + 1]))
            layers.append(nn.ReLU())
            if dropout_rate is not None:
                layers.append(nn.Dropout(dropout_rate))
            if batch_normalization:
                layers.append(nn.BatchNorm1d(encoder_dims[i + 1]))
        self.encoder = nn.Sequential(*layers)

        if decoder_hidden_dims is None:
            if len(encoder_hidden_dims) == 1:
                decoder_hidden_dims = []
            else:
                decoder_hidden_dims = encoder_hidden_dims[::-1][1:]

        if (append_mode == 2) or (append_mode == 3):
            decoder_dims = [encoder_dims[-1] + added_features_dim, *decoder_hidden_dims, input_dim]
        else:
            decoder_dims = [encoder_dims[-1], *decoder_hidden_dims, input_dim]
        layers = []
        for i in range(len(decoder_dims) - 1):
            layers.append(nn.Linear(decoder_dims[i], decoder_dims[i + 1]))
            if i <= (len(decoder_dims) - 3):
                layers.append(nn.ReLU())
                if dropout_rate is not None:
                    layers.append(nn.Dropout(dropout_rate))
                if batch_normalization:
                    layers.append(nn.BatchNorm1d(decoder_dims[i + 1]))
        self.decoder = nn.Sequential(*layers)
        self.mean_decoder = nn.Sequential(nn.Linear(decoder_dims[0], int(np.floor(decoder_dims[0]/2))),nn.ReLU() , nn.Linear(int(np.floor(decoder_dims[0]/2)), 1) )

    def forward(self, x):
        if (type(x) == list) or (type(x) == tuple):
            input_shape = x[0].size()
            x_in = x[0].flatten(start_dim=1)
            if self.append_mode == 1: # append at encoder
                out = self.encoder(torch.cat([x_in, x[1]], dim=1))
                out_mean = self.mean_decoder(out)
                out = self.decoder(out)
                
            elif self.append_mode == 2: # append at decoder
                out = self.encoder(x_in)
                out_mean = self.mean_decoder(torch.cat([out, x[1]], dim=1))
                out = self.decoder(torch.cat([out, x[1]], dim=1))
            elif self.append_mode == 3: # append at encoder and decoder
                out = self.encoder(torch.cat([x_in, x[1]], dim=1))
                out_mean = self.mean_decoder(torch.cat([out, x[1]], dim=1))
                out = self.decoder(torch.cat([out, x[1]], dim=1))
        else:
            input_shape = x.size()
            x_in = x.flatten(start_dim=1)
            out = self.encoder(x_in)
            out_mean = self.mean_decoder(out)
            out = self.decoder(out)

        return out.view(input_shape), out_mean.unsqueeze(-1)
    

class Autoencoder_mean(nn.Module):

    def __init__(self, input_dim, encoder_hidden_dims,  added_features_dim=0,  append_mode = 1, batch_normalization=False, dropout_rate=None) -> None:
        super(Autoencoder_mean, self).__init__()

        self.append_mode = append_mode
        encoder_dims = [input_dim + added_features_dim, *encoder_hidden_dims]

        layers = []
        for i in range(len(encoder_dims) - 1):
            layers.append(nn.Linear(encoder_dims[i], encoder_dims[i + 1]))
            layers.append(nn.ReLU())
            if dropout_rate is not None:
                layers.append(nn.Dropout(dropout_rate))
            if batch_normalization:
                layers.append(nn.BatchNorm1d(encoder_dims[i + 1]))
        
        self.encoder = nn.Sequential(*layers)
        if self.append_mode in [2,3]:
            self.out = nn.Linear(encoder_dims[-1] + added_features_dim,1)
        else: 
            self.out = nn.Linear(encoder_dims[-1] ,1)


    def forward(self, x):
        if (type(x) == list) or (type(x) == tuple):
            if self.append_mode ==1:
                x_in = x[0].flatten(start_dim=1)
                out = self.encoder(torch.cat([x_in, x[1]], dim=1))
                out = self.out(out)
            elif self.append_mode == 2:
                x_in = x[0].flatten(start_dim=1)
                out = self.encoder(x_in)
                out = self.out(torch.cat([out, x[1]], dim=1))
            elif self.append_mode == 3:
                x_in = x[0].flatten(start_dim=1)
                out = self.encoder(torch.cat([x_in, x[1]], dim=1))
                out = self.out(torch.cat([out, x[1]], dim=1))

        else:
            x_in = x.flatten(start_dim=1)
            out = self.encoder(x_in)


        return out.unsqueeze(-1)
    

class cVAE(nn.Module):
    def __init__(self, input_dim, encoder_hidden_dims, decoder_hidden_dims=None,  batch_normalization=False, dropout_rate=None, device = torch.device('cpu')) -> None:
        super(cVAE, self).__init__()
        latent_dim = encoder_hidden_dims[-1]
        ### encoder 
        encoder_dims = [input_dim + latent_dim, *encoder_hidden_dims[:-1]]
        self.encoder = CondVariationalEncoder(encoder_dims, batch_normalization, dropout_rate)
        ### mean and std
        self.mu = nn.Linear(encoder_dims[-1], latent_dim)
        self.sigma = nn.Linear(encoder_dims[-1], latent_dim)
        ### embedding
        embedding_dims = [input_dim , *encoder_hidden_dims[:-1]]
        self.embedding = CondVariationalEmbedding(embedding_dims, latent_dim, batch_normalization, dropout_rate)
        ### decoder
        if decoder_hidden_dims is None:
            if len(encoder_hidden_dims) == 1:
                decoder_hidden_dims = []
            else:
                decoder_hidden_dims = encoder_hidden_dims[::-1][1:]
        decoder_dims = [latent_dim * 2 , *decoder_hidden_dims, input_dim]
        self.decoder = CondVariationalDecoder(decoder_dims, batch_normalization, dropout_rate)
        #
        self.N = torch.distributions.Normal(0, 1)
        # Get sampling working on GPU
        if device.type == 'cuda':
            self.N.loc = self.N.loc.cuda()
            self.N.scale = self.N.scale.cuda()
        #
    def forward(self, x, x_mean):
        #
            x_mean = x_mean.flatten(start_dim=1)
            if (type(x) == list) or (type(x) == tuple):
                input_shape = x[0].size()
                x_in = x[0].flatten(start_dim=1)
                add_features = x[1]
                x_in = torch.cat([x_in, add_features], dim=1) 
                x_mean = torch.cat([x_mean, add_features], dim=1)  
            else:
                input_shape = x.size()
                x_in = x.flatten(start_dim=1)
            #
            embedding = self.embedding(x_mean)    
            out = self.encoder(torch.cat([x_in, embedding], dim=1) )
            mu = self.mu(out)
            sigma = torch.exp(self.sigma(out))
            #
            z = mu + sigma*self.N.sample(mu.shape)
            out = self.decoder(torch.cat([z, embedding], dim=1) )
            return out.view(input_shape), mu, sigma
                    

class CondVariationalEncoder(nn.Module):     
        def __init__(self, encoder_dims, batch_normalization = False, dropout_rate = None):
            super(CondVariationalEncoder, self).__init__()
            layers = []
            for i in range(len(encoder_dims) - 1):
                layers.append(nn.Linear(encoder_dims[i], encoder_dims[i + 1]))
                layers.append(nn.ReLU())
                if dropout_rate is not None:
                    layers.append(nn.Dropout(dropout_rate))
                if batch_normalization:
                    layers.append(nn.BatchNorm1d(encoder_dims[i + 1]))
            self.module = nn.Sequential(*layers)
        def forward(self, x):
            return self.module(x)

class CondVariationalDecoder(nn.Module):
        def __init__(self, decoder_dims, batch_normalization = False, dropout_rate = None):
            super(CondVariationalDecoder, self).__init__()
            layers = []
            for i in range(len(decoder_dims) - 1):
                layers.append(nn.Linear(decoder_dims[i], decoder_dims[i + 1]))
                if i <= (len(decoder_dims) - 3):
                    layers.append(nn.ReLU())
                    if dropout_rate is not None:
                        layers.append(nn.Dropout(dropout_rate))
                    if batch_normalization:
                        layers.append(nn.BatchNorm1d(decoder_dims[i + 1]))
            self.module = nn.Sequential(*layers)    
        def forward(self, x):
            return self.module(x)
        
class CondVariationalEmbedding(nn.Module):    
        def __init__(self, encoder_dims, latent_dim, batch_normalization = False, dropout_rate=None):
            super(CondVariationalEmbedding, self).__init__()
            layers = []
            layers.append(nn.Linear(encoder_dims[-1], latent_dim))
            layers.append(nn.ReLU())
            if dropout_rate is not None:
                layers.append(nn.Dropout(dropout_rate))
            if batch_normalization:
                layers.append(nn.BatchNorm1d(latent_dim))
            self.encoder = CondVariationalEncoder(encoder_dims, batch_normalization = False, dropout_rate = None)
            self.module = nn.Sequential( *layers ) 
        def forward(self, x):
            x = self.encoder(x)
            return self.module(x)
        


def weights_init(m):
	if isinstance(m, nn.Linear):
		nn.init.xavier_uniform_(m.weight)
		if m.bias is not None:
			nn.init.constant_(m.bias, 0)






class Autoencoder_decoder(nn.Module):

    def __init__(self, input_dim, latent_size, decoder_hidden_dims=None, added_features_dim=0, append_mode=1, batch_normalization=False, dropout_rate=None,  device = 'cpu') -> None:
        super(Autoencoder_decoder, self).__init__()
        self.append_mode = append_mode
        self.latent_size = latent_size


        if (append_mode == 2) or (append_mode == 3):
            decoder_dims = [latent_size + added_features_dim, *decoder_hidden_dims, input_dim]
        else:
            decoder_dims = [latent_size, *decoder_hidden_dims, input_dim]
        layers = []
        for i in range(len(decoder_dims) - 1):
            layers.append(nn.Linear(decoder_dims[i], decoder_dims[i + 1]))
            if i <= (len(decoder_dims) - 3):
                layers.append(nn.ReLU())
                if dropout_rate is not None:
                    layers.append(nn.Dropout(dropout_rate))
                if batch_normalization:
                    layers.append(nn.BatchNorm1d(decoder_dims[i + 1]))
        self.decoder = nn.Sequential(*layers)
        self.decoder.apply(weights_init)


    def forward(self, x, mu, log_var, sample_size = 1, seed = None):
        if (type(x) == list) or (type(x) == tuple):
            input_shape = x[0].shape
            input_shape = (sample_size, *input_shape)

            if self.append_mode in [2,3]: # append at decoder

                    out = self.sample(mu, log_var, sample_size, seed)
                    out = torch.cat([out, x[1].unsqueeze(0).expand((sample_size, *x[1].shape))], dim=2)
                    out_shape = out.shape
                    out = self.decoder(torch.flatten(out, start_dim = 0, end_dim = 1))
                    out = torch.unflatten(out, dim = 0 , sizes = out_shape[0:2])
                    
            else: # append at encoder

                out = self.sample( mu, log_var, sample_size, seed)
                out_shape = out.shape
                out = self.decoder(torch.flatten(out, start_dim = 0, end_dim = 1))
                out = torch.unflatten(out, dim = 0 , sizes = out_shape[0:2])

        else:
            input_shape = x.size()
            input_shape = (sample_size, *input_shape)
            x_in = x.flatten(start_dim=1)

            out = self.sample(mu, log_var, sample_size, seed)
            out_shape = out.shape
            out = self.decoder(torch.flatten(out, start_dim = 0, end_dim = 1))
            out = torch.unflatten(out, dim = 0 , sizes = out_shape[0:2])


        return out.view(input_shape)

    
    def sample( self, mu, log_var, sample_size = 1, seed = None):
        if seed is not None:
            torch.manual_seed(seed)
        var = torch.exp(log_var) + 1e-4
        return Normal(mu, torch.sqrt(var)).rsample(sample_shape=(sample_size,))






class FCNN(nn.Module):
    """
    Simple fully connected neural network.
    """
    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.network(x)

class MAF(nn.Module):
    """
    Masked auto-regressive flow.

    [Papamakarios et al. 2018]
    """
    def __init__(self, dim, hidden_dim = 16, base_network=FCNN, condition_embedding_size = None,  device = 'cpu'):
        super().__init__()
        self.device = device
        self.dim = dim
        # self.layers = nn.ModuleList()
        layers = [] 
        added_features = condition_embedding_size if condition_embedding_size is not None else 0
        for i in range(1, dim):
            layers.append(base_network(i + added_features, 2, hidden_dim))
        self.layers = nn.Sequential(*layers)
        if condition_embedding_size is None:
            self.register_parameter('initial_param', param = nn.Parameter(torch.Tensor(2)))  ## make sure to register the newly defined parameters so that they are trainable!
            self.reset_parameters()
        else:
            self.initial_param = base_network(added_features, 2, hidden_dim)

    def reset_parameters(self):
        init.uniform_(self.initial_param, -math.sqrt(0.5), math.sqrt(0.5))

    def forward(self, x, condition = None):
        z = torch.zeros_like(x).to(self.device)
        log_det = torch.zeros(z.shape[0]).to(self.device)
        for i in range(self.dim):
            if i == 0:
                if condition is None:
                    mu, alpha = self.initial_param[0], self.initial_param[1]
                else:
                    out = self.initial_param(condition)
                    mu, alpha = out[:, 0], out[:, 1]
            else:
                if condition is None:
                    x_in = x[:, :i].clone()
                else:
                    x_in = torch.cat([x[:, :i].clone(), condition], dim = -1)

                out = self.layers[i - 1](x_in)
                mu, alpha = out[:, 0], out[:, 1]
            z[:, i] = (x[:, i] - mu) / torch.exp(alpha)
            log_det -= alpha
        return z.flip(dims=(1,)), log_det

    def inverse(self, z, condition = None):
        x = torch.zeros_like(z).to(self.device)
        log_det = torch.zeros(z.shape[0]).to(self.device)
        z = z.flip(dims=(1,))
        for i in range(self.dim):
            if i == 0:
                if condition is None:
                    mu, alpha = self.initial_param[0], self.initial_param[1]
                else:
                    out = self.initial_param(condition)
                    mu, alpha = out[:, 0], out[:, 1]
            else:
                if condition is None:
                    x_in = x[:, :i].clone()
                else:
                    x_in = torch.cat([x[:, :i].clone(), condition], dim = -1)
                out = self.layers[i - 1](x_in)
                mu, alpha = out[:, 0], out[:, 1]
            x[:, i] = mu + torch.exp(alpha) * z[:, i]
            # print(alpha)
            log_det += alpha # torch.clamp(-1*alpha, max=10)
        return x, log_det
    
class NormalizingFlowModel(nn.Module):

    def __init__(self, flows, device = 'cpu'):
        super().__init__()
        # self.prior = MultivariateNormal(torch.zeros(dim), torch.eye(dim))
        self.flows = nn.ModuleList(flows)
        self.device = device
    def forward(self, x, condition = None):
        bsz, _ = x.shape
        log_det = torch.zeros(bsz).to(self.device)
        for flow in self.flows:
            x, ld = flow.forward(x, condition = condition)
            log_det = log_det+  ld
        # z, prior_logprob = x, self.prior.log_prob(x)
        return x, log_det #prior_logprob

    def inverse(self, z, condition = None):
        bsz, _ = z.shape
        log_det = torch.zeros(bsz).to(self.device)
        for flow in self.flows[::-1]:
            z, ld = flow.inverse(z, condition = condition)
            log_det = log_det + ld
        x = z
        return x, log_det
    