# using Pkg
# Pkg.activate(@__DIR__)

using Transformers
using Transformers.TextEncoders
using Transformers.HuggingFace
using Transformers.Datasets
using Transformers.Datasets: GLUE
# using HuggingFaceDatasets

using Flux, Tsunami
using  Optimisers: Optimisers, Adam

#### MODEL #########

mutable struct Bert{B,E,L} <: FluxModule
    net::B
    tokenizer::E
    labels::L
end

Flux.trainable(b::Bert) = (net = b.net,)

function Bert(labels)
    labels = Vocab([labels...])
    num_labels = length(labels)
    config = HuggingFace.HGFConfig(hgf"bert-base-uncased:config"; num_labels)
    net = load_model("bert-base-uncased", :ForSequenceClassification; config)
    tokenizer = load_tokenizer("bert-base-uncased"; config)
    return Bert(net, tokenizer, labels)
end

function Tsunami.train_step(model::Bert, trainer, batch, batch_idx)
    input = preprocess(model, batch)
    l, p = loss(model, input)
    acc = acc(p, input.label)
    Tsunami.log(trainer, "train/loss", l)
    Tsunami.log(trainer, "train/acc", acc)
    return l
end

function configure_optimizers(model, trainer)
    return Adam(model.net, 1e-5)
end

function preprocess(model, batch)
    data = encode(model.encoder, map(collect, zip(batch[1], batch[2])))
    label = lookup(OneHot, model.labels, batch[3])
    return merge(data, (; label))
end

function acc(p, label)
    pred = Flux.onecold(p)
    truth = Flux.onecold(label)
    sum(pred .== truth) / length(truth)
end

function loss(model, input)
    nt = model(input)
    p = nt.logit
    l = logitcrossentropy(p, input.label)
    return l, p
end

### DATASET ######
mutable struct Dataset
    mnli
    batchsize
    split
end

Dataset(; split = :train, batchsize = 4) = Dataset(GLUE.MNLI(false), batchsize, split)

function Base.iterate(d::Dataset, datas = nothing)
    if datas === nothing
        if d.split == :train
            datas = dataset(Train, d.mnli)
        elseif d.split == :dev
            datas = dataset(Dev, d.mnli)
        end
    end
    res = get_batch(datas, d.batchsize)
    res === nothing && return nothing
    return res, datas
end

get_labels(d::Dataset) = Transformers.get_labels(d.mnli)
Base.length(d::Dataset) = nothing # unknown length

########## TRAINING #########

train_loader = Dataset(; split = :train, batchsize = 4)
val_loader = Dataset(; split = :dev, batchsize = 4)

labels = get_labels(train_loader)
model = Bert(labels)

Trainer(max_steps = 10)

batch = first(train_loader)

