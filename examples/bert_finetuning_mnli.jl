using Transformers
using Transformers.TextEncoders
using Transformers.HuggingFace
using Transformers.Datasets
using Transformers.Datasets: GLUE
using HuggingFaceDatasets

using Flux
using  Optimisers: Optimisers, Adam

function preprocess(batch)
    global bertenc, labels
    data = encode(bertenc, map(collect, zip(batch[1], batch[2])))
    label = lookup(OneHot, labels, batch[3])
    return merge(data, (label = label,))
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

dtrain = load_dataset("glue", "mnli", split = "train")


mnli = GLUE.MNLI(false)
labels = Vocab([get_labels(mnli)...])
# load the old config file and update some value
bert_config = HuggingFace.HGFConfig(hgf"bert-base-uncased:config"; num_labels = length(labels))

# load the model / tokenizer with new config
bert_model = load_model("bert-base-uncased", :ForSequenceClassification; config = bert_config)
bertenc = load_tokenizer("bert-base-uncased"; config = bert_config)

opt = Optimisers.setup(Adam(1e-6), bert_model)
    
for e = 1:2
    @info "epoch: $e"
    datas = dataset(Train, mnli)
    i = 1
    al = zero(Float64)
    while (batch = get_batch(datas, Batch)) !== nothing
        input = preprocess(batch::Vector{Vector{String}})
        (l, p), back = Zygote.pullback(bert_model) do model
            loss(model, input)
        end
        a = acc(p, input.label)
        al += a
        (grad,) = back((Zygote.sensitivity(l), nothing))
        i += 1
        Optimisers.update!(opt, bert_model, grad)
        mod1(i, 16) == 1 && @info "training" loss=l accuracy=al/i
    end

    test()
end

function test()
    @info "testing"
    i = 1
    al = zero(Float64)
    datas = dataset(Dev, mnli)
    while (batch = get_batch(datas, Batch)) !== nothing
        input = preprocess(batch)
        p = bert_model(input).logit
        a = acc(p, input.label)
        al += a
        i+=1
    end
    al /= i
    @info "testing" accuracy = al
    return al
end