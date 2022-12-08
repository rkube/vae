using Flux
using CUDA
using CairoMakie
using ColorSchemes

"""
    Reconstruct a batch of images and 
"""
function build_img_array(encoder_μ, encoder_logvar, decoder, loader)
    # Get one batch from the loader
    x = first(loader);
    num_img = max(size(x)[end], 10);

    # Reconstruct an image
    μ = encoder_μ(x);
    logvar = encoder_logvar(x);
    z = μ + CuArray(randn(Float32, size(logvar))) .* exp.(0.5f0 * logvar);
    x̂ = sigmoid.(decoder(z));

    img_array = zeros(Float32, 28 * num_img, 28 * 2);
    for ix ∈ 1:num_img
        ix_start = (ix - 1) * 28 + 1
        ix_end = ix * 28
        img_array[ix_start:ix_end, 29:56] = x[:, 28:-1:1, 1, ix];
        img_array[ix_start:ix_end, 1:28] = x̂[:, 28:-1:1, 1, ix];
    end
    img_array
end

"""
    plot_latent_space(encoder_μ, encoder_logvar, X, labels)

Plot point cloud in latent space

* encoder_μ - Mapping of training data to mean of distribution
* encoder_logvar - Maps input space to log var of distribution
* X - images
* labels - labels

"""
function plot_latent_space(encoder_μ, encoder_logvar, X_test, labels_test, epoch)
    μ_test = encoder_μ(X_test);
    logvar_test = encoder_logvar(X_test);

    fig = Figure()
    ax = Axis(fig[1, 1], title="epoch $(epoch)")
    for l ∈ sort(unique(labels_test))
        l_ix = labels_test .== l
        scatter!(ax, μ_test[1, l_ix], μ_test[2, l_ix], color=ColorSchemes.:tab10[l+1], label="$(l)")
    end
    axislegend(ax, position=:rt)
    fig
end

function plot_latent_dens(encoder_μ, encoder_logvar, X_test, labels_test, epoch)
    μ_test = encoder_μ(X_test);
    logvar_test = encoder_logvar(X_test);

    fig = Figure()
    ax = Axis(fig[1, 1], title="epoch $(epoch)")
    for l ∈ sort(unique(labels_test))
        l_ix = labels_test .== l
        dens = kde(μ_test[:, l_ix]')
        #TODO: Add appropriate colorbars. One color per label.
        contourf!(ax, dens.x, dens.y, dens.density)
        #scatter!(ax, μ_test[1, l_ix], μ_test[2, l_ix], color=ColorSchemes.:tab10[l+1], label="$(l)")
    end
    axislegend(ax, position=:rt)
    fig
end



export build_img_array, plot_latent_space