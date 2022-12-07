using Flux
using Zygote
using MLUtils: DataLoader
using MLDatasets: MNIST
using CUDA
using CairoMakie

using vae

# Get some training data
data_train = MNIST(:train);
all_X_train, all_Y_train = data_train[:];
# scale to -0.5:0.5 in order to use tanh
#all_X_train = (all_X_train .- 5f-1) .* 2f0;
all_X_train = Flux.unsqueeze(all_X_train, dims=3) |> gpu;

all_X_test, _ = MNIST(:test)[:];
#all_X_test = (all_X_test .- 5f-1) .* 2f0;
all_X_test = Flux.unsqueeze(all_X_test, dims=3) |> gpu;

struct Reshape
    shape
end
Reshape(args...) = Reshape(args)
(r::Reshape)(x) = reshape(x, r.shape)
Flux.@functor Reshape ()

latent_dim = 2
batch_size = 128
num_epochs = 10


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
    Conv((3, 3), 1=>32, relu; stride=2, ), # 13x13 
    Conv((3, 3), 32=>64, relu; stride=2), # 6x6
    Flux.flatten,
    Dense(64 * 6 * 6, 16, relu),
);
encoder_μ = Chain(encoder_features, Dense(16, latent_dim)) |> gpu;

# The encoder network will be tasked to learn the log of the variance.
# If we would only learn σ², we would have to enforce it to be positive definite.
# This constraint is removed by asking it to learn the logarithm
encoder_logvar = Chain(encoder_features, Dense(16, latent_dim)) |> gpu;


# The decoder describes the likelihood p(x|z)_θ that maps
# a vector from the latent space onto an observation x:
# Or said otherwise: given a hidden representation z, the decoder
# decodes this into a distribution over the observations x.
decoder = Chain(
    Dense(latent_dim, 64 * 7 * 7, relu),
    Reshape(7, 7, 64, :),
    ConvTranspose((3, 3), 64=>64, relu, stride=2, pad=SamePad()),
    ConvTranspose((3, 3), 64=>32, relu, stride=2, pad=SamePad()),
    ConvTranspose((3, 3), 32 => 1, stride=1, pad=SamePad())
) |> gpu;

# No \sigma


# Set up data loaders
loader_train = DataLoader(all_X_train, batchsize=batch_size, shuffle=true);
loader_test = DataLoader(all_X_test, batchsize=10, shuffle=true);
length(loader_train)


# Hyperparameters that weigh the terms in the loss function
# Note: it's best to set γ=0
β = 1.0
γ = 1f-4

# Inner training loop
x = first(loader_train);
# size(x)
μ = encoder_μ(x);
logvar = encoder_logvar(x);
# Apply reparameterization trick to latent sample
z = μ + CuArray(randn(Float32, size(logvar))) .* exp.(5f-1 * logvar)
# Reconstruct latent sample
x̂ = decoder(z);
# Negative reconstruction loss Ε_q[logp_x_z]
logp_x_z = -sum(Flux.Losses.logitbinarycrossentropy.(x̂, x)) / batch_size
# KL(qᵩ(z|x)||p(z)) where p(z)=N(0,1) and qᵩ(z|x) models the encoder
# The @. macro makes sure that all operates are elementwise
kl_q_p = 0.5f0 * sum(@. (exp(logvar) + μ^2 - logvar - 1f0)) / batch_size
# Weight decay regularisation term
reg = sum(x->sum(x.^2), Flux.params(encoder_μ, encoder_logvar, decoder))
# We want to maximize the evidence lower bound
elbo = logp_x_z - β .* kl_q_p
loss = -elbo + γ * reg

params = Flux.params(encoder_μ, encoder_logvar, decoder);
opt = Flux.Optimise.Adam(1e-3);

lossvec = zeros(num_epochs)
for epoch ∈ 1:num_epochs
    batch_idx = 1
    acc_loss = 0.0
    for x ∈ loader_train
        loss, back = Zygote.pullback(params) do
            μ = encoder_μ(x);
            logvar = encoder_logvar(x);
            # Apply reparameterization trick to latent sample
            z = μ + CuArray(randn(Float32, size(logvar))) .* exp.(5f-1 * logvar);
            # Reconstruct latent sample
            x̂ = decoder(z);
            # Negative reconstruction loss Ε_q[logp_x_z]
            logp_x_z = -sum(Flux.Losses.logitbinarycrossentropy.(x̂, x)) / batch_size;
            # KL(qᵩ(z|x)||p(z)) where p(z)=N(0,1) and qᵩ(z|x) models the encoder
            # The @. macro makes sure that all operates are elementwise
            kl_q_p = 0.5f0 * sum(@. (exp(logvar) + μ^2 - logvar - 1f0)) / batch_size;
            # Weight decay regularisation term
            reg = sum(x->sum(x.^2), Flux.params(encoder_μ, encoder_logvar, decoder));
            # We want to maximize the evidence lower bound
            elbo = logp_x_z - β .* kl_q_p;
            Zygote.ignore() do 
                @show logp_x_z, kl_q_p, reg
            end
            loss = -elbo + γ * reg;
        end
        @show loss
        grads = back(1f0);
        Flux.Optimise.update!(opt, params, grads);
        acc_loss += loss;
        batch_idx +=1 
    end
    lossvec[epoch] = acc_loss
    @show epoch, acc_loss

    img_array = build_img_array(encoder_μ, encoder_logvar, decoder, loader_test);
    f = Figure()
    ax = Axis(f[1, 1], title="epoch $(epoch)")
    contourf!(ax, img_array, colormap=:grays)
    save("vae_epoch$(epoch).png", f)
end

