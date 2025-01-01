# Load a pretrained Bert model and fine-tune it on the 
# glue-mnli dataset.

# Ported to Tsunami from Transformers.jl:
# https://github.com/chengchingwen/Transformers.jl/blob/master/example/BERT/mnli/train.jl

using Transformers
using Transformers.TextEncoders
using Transformers.HuggingFace
using Transformers.Datasets
using Transformers.Datasets: GLUE
# using HuggingFaceDatasets

using Flux, Tsunami
using  Optimisers: Optimisers, Adam
using ChainRulesCore

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

(b::Bert)(x) = b.net(x)

function Tsunami.train_step(model::Bert, trainer, batch, batch_idx)
    input = preprocess(model, batch)
    l, p = loss(model, input)
    acc = accuracy(p, input.label)
    Tsunami.log(trainer, "train/loss", l)
    Tsunami.log(trainer, "train/acc", acc)
    return l
end

function Tsunami.configure_optimisers(model::Bert, trainer)
    return Optimisers.setup(Optimisers.Adam(1e-6), model)
end

function preprocess(model, batch)
    data = encode(model.tokenizer, map(collect, zip(batch[1], batch[2])))
    label = lookup(OneHot, model.labels, batch[3])
    return merge(data, (; label))
end

@non_differentiable preprocess(::Any...)

function accuracy(p, label)
    pred = Flux.onecold(p)
    truth = Flux.onecold(label)
    sum(pred .== truth) / length(truth)
end

function loss(model, input)
    nt = model(input)
    p = nt.logit
    l = Flux.logitcrossentropy(p, input.label)
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
    return (res...,), datas
end

get_labels(d::Dataset) = Transformers.get_labels(d.mnli)

########## TRAINING #########

train_loader = Dataset(; split = :train, batchsize = 4)
val_loader = Dataset(; split = :dev, batchsize = 4)

# batch = first(train_loader)

labels = get_labels(train_loader)
model = Bert(labels)

# :gpu not working yet due to custom gpu movement rules in Transformers.jl
trainer = Trainer(accelerator=:cpu, max_steps=2, fast_dev_run=false, checkpointer=false, log_every_n_steps=1)
fit_state = Tsunami.fit!(model, trainer, train_loader)
