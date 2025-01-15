using AutoStructs: @structdef # for conveniently prototyping struct
using Flux: Flux, Embedding, MultiHeadAttention, LayerNorm, Dense, Chain, gelu
using Tsunami
using LinearAlgebra, Random, Statistics
using NNlib: trues_like

# Positional Embedding code from https://github.com/mashu/PositionalEmbeddings.jl

struct AbsolutePE{T<:AbstractArray}
    embedding_size::Int
    max_length::Int
    embeddings::T
end

Flux.@layer AbsolutePE trainable=()

function AbsolutePE(embedding_size::Int; max_length::Int=1024, base::Int=10_000)
    freqs = compute_frequencies(embedding_size, max_length, base)
    embeddings = zeros(Float32, embedding_size, max_length)
    embeddings[1:2:end, :] .= sin.(freqs)
    embeddings[2:2:end, :] .= cos.(freqs)
    return AbsolutePE(embedding_size, max_length, embeddings)
end

function compute_frequencies(dim::Int, seq_len::Int, base::Number=10_000)
    θ = 1 ./ (base .^ (collect(0:2:dim-1) ./ dim))
    positions = collect(0:seq_len-1)
    return θ * positions'
end

function (layer::AbsolutePE)(x::AbstractArray)
    channels, seq_len, batch_size = size(x)
    @assert channels == layer.embedding_size "Channel dimension must match embedding size"
    @assert seq_len <= layer.max_length "Sequence length exceeds maximum length"
    pos_embeddings = view(layer.embeddings, :, 1:seq_len)
    embeddings_broadcast = reshape(pos_embeddings, (channels, seq_len, 1))
    return x .+ embeddings_broadcast
end


@structdef function TransformerBlock(; dim_model=128, num_heads=8)
    ffwd_norm = LayerNorm(dim_model)
    mha_norm = LayerNorm(dim_model)
    mha = MultiHeadAttention(dim_model; nheads=num_heads)
    ffwd = Chain(Dense(dim_model => 4dim_model, gelu), Dense(4dim_model => dim_model))
    return TransformerBlock(mha_norm, ffwd_norm, mha, ffwd)
end

Flux.@layer TransformerBlock

function (block::TransformerBlock)(x, mask)
    x = x .+ block.mha(block.mha_norm(x); mask)[1]
    x = x .+ block.ffwd(block.ffwd_norm(x))
    return x
end

@structdef function Transformer(; dim_model=512, vocab_size=33278, num_layers=6, num_heads=8, max_length=512)
    embedding = Embedding(vocab_size => dim_model)
    pos_encoder = AbsolutePE(dim_model; max_length)
    blocks = [TransformerBlock(; dim_model, num_heads) for _ in 1:num_layers]
    # lay[MultiHeadAttention(d_model, d_model, d_model, d_model) for _ in 1:6]
    head = Dense(dim_model => vocab_size)
    return Transformer(
            max_length, dim_model, vocab_size, num_layers, num_heads,
            embedding, pos_encoder, blocks, head)
end

Flux.@layer Transformer

function (model::Transformer)(x)
    x = model.embedding(x)
    dim, len, batch_size = size(x)
    causal_mask = triu(trues_like(x, (len, len)))
    x = model.pos_encoder(x)
    for block in model.blocks
        x = block(x, causal_mask)
    end
    return model.head(x)
end


# dim, max_length, batch_size = 64, 10, 2
# vocab_size = 100
# x = rand(1:vocab_size, max_length, batch_size)
# model = Transformer(; dim_model=dim, vocab_size, max_length)
# h = model.embedding(x)
# @assert size(h) == (dim, max_length, batch_size)
# b1 = model.blocks[1]
# causal_mask = triu(trues_like(h, (max_length, max_length)))
# y = b1(h, causal_mask)
# @assert size(y) == size(h)
# i = 3
# Flux.gradient(h -> mean(b1(h, causal_mask)[:,i,1]), h)[1]
# y = model(x)
# @assert size(y) == (vocab_size, max_length, batch_size)