using Flux
using Zygote
using MLUtils: DataLoader
using MLDatasets: MNIST
using CairoMakie


# Get some training data
data_train = MNIST(:train);
all_X_train, all_Y_train = data_train[:];
all_X_train = Flux.unsqueeze(all_X_train, dims=3);

struct Reshape
    shape
end
Reshape(args...) = Reshape(args)
(r::Reshape)(x) = reshape(x, r.shape)
Flux.@functor Reshape ()

# The encoder network  will learn the variational parameters μ, σ
# for a given observation x. That is, for a given sample xⁱ, the encoder
# will yield μ(xⁱ), σ(xⁱ).
# Written as a distribution, the encoder represents the mapping
# q_ϕ(zⁱ|xⁱ) = Normal(zⁱ|μ(xⁱ), diag(σ²(xⁱ))).
# Here diag(σ²) refers to the fact that this is a multi-dimensional normal distribution
#
# During learning, the weights of the encoder will be adjusted as to yield the
# optimum paramters ϕ for this mapping.
encoder_features = Chain(
    Conv((5, 5), 1=>32, relu; stride=1),
    Conv((5, 5), 32=>32, relu; stride=1),
    Conv((7, 7), 32=>32, relu; stride=1),
    Conv((7, 7), 32=>32, relu, stride=1),
    Flux.flatten,
    Dense(32 * 8 * 8, 256, relu),
    Dense(256, 256, relu)
)
encoder_μ = Chain(encoder_features, Dense(256, 10))

# The encoder network will be tasked to learn the log of the variance.
# If we would only learn σ², we would have to enforce it to be positive definite.
# This constraint is removed by asking it to learn the logarithm
encoder_logvar = Chain(encoder_features, Dense(256, 10))


# The decoder describes the likelihood p(x|z)_θ that maps
# a vector from the latent space onto an observation x:
# Or said otherwise: given a hidden representation z, the decoder
# decodes this into a distribution over the observations x.
decoder = Chain(
    Dense(10, 256, relu),
    Dense(256, 256, relu),
    Dense(256, 32 * 8 * 8, relu),
    Reshape(8, 8, 32, :),
    ConvTranspose((7, 7), 32=>32, relu),
    ConvTranspose((7, 7), 32 => 32, relu),
    ConvTranspose((5, 5), 32 => 32, relu),
    ConvTranspose((5, 5), 32 => 1)
)


batch_size=128;
# Set up data loaders
loader_train = DataLoader((all_X_train, all_Y_train), batchsize=batch_size, shuffle=true);

params = Flux.params(encoder_μ, encoder_logvar, decoder);
opt = Flux.Optimise.Adam(1e-3);

# Inner training loop
(x,y) = first(loader_train)

# Hyperparameters that weigh the terms in the loss function
β = 1.0
γ = 1.0

for (x,y) in loader_train
    acc_loss = 0.0
    loss, back = Zygote.pullback(params) do
        μ = encoder_μ(x);
        logvar = encoder_logvar(x);
        # Apply reparameterization trick to latent sample
        z = μ + randn(Float32, size(logvar)) .* exp.(5f-1 * logvar)
        # Reconstruct latent sample
        x̂ = decoder(z);
        # Negative reconstruction loss Ε_q[logp_x_z]
        logp_x_z = -sum(Flux.Losses.logitbinarycrossentropy.(x̂, x)) / batch_size
        # KL(qᵩ(z|x)||p(z)) where p(z)=N(0,1) and qᵩ(z|x) models the encoder
        # The @. macro makes sure that all operates are elementwise
        kl_q_p = 0.5f0 * sum(@. (exp(logvar) + μ^2 - logvar - 1f0)) / batch_size
        # Weight decay regularisation term
        reg = γ * sum(x->sum(x.^2), Flux.params(encoder_μ, encoder_logvar, decoder))
        # We want to maximize the evidence lower bound
        elbo = logp_x_z - β .* kl_q_p
        loss = -elbo + reg
    end
    grads = back(1f0);
    Flux.Optimise.update!(opt, params, grads);
    acc_loss += loss;
    @show loss
end


