using Flux
using CUDA

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


export build_img_array