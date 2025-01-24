# # Auto-Regressive Transformer trained on the WikiText-2 dataset

# References:   
# - https://huggingface.co/docs/tokenizers/quicktour

using Tsunami, Flux
using Flux: DataLoader
import HuggingFaceTokenizers as HFT
import HuggingFaceDatasets as HFD
using PythonCall: PyList, pyconvert

include("model.jl") # Transformer

function get_dataset()
    dataset = HFD.load_dataset("wikitext", "wikitext-2-raw-v1")
    train_dataset = dataset["train"]   
    val_dataset = dataset["validation"]
    test_dataset = dataset["test"]
    return train_dataset, val_dataset, test_dataset
end

function get_tokenizer(train_dataset)
    # we follow https://huggingface.co/docs/tokenizers/quicktour
    bpe = HFT.tokenizers.models.BPE(unk_token="[UNK]")
    tokenizer = HFT.tokenizers.Tokenizer(bpe) # python tokenizer
    pre_tok = HFT.tokenizers.pre_tokenizers.Whitespace()
    tokenizer.pre_tokenizer = pre_tok
    trainer = HFT.tokenizers.trainers.BpeTrainer(special_tokens=PyList(["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]))
    tokenizer.train_from_iterator(train_dataset["text"], trainer=trainer)
    tokenizer.enable_padding(pad_id=3, pad_token="[PAD]")
    vocab_size = pyconvert(Int, tokenizer.get_vocab_size())
    return tokenizer, vocab_size
    # return HFT.Tokenizer(tokenizer) # return the julia wrapper
end

function encode_batch(tokenizer, batch)
    to_ids(s) = pyconvert(Vector{Int}, tokenizer.encode(s).ids)
    return [to_ids(batch[i]["text"]) for i in 1:length(batch)]
end

function transform(batch)
    encodings = tokenizer.encode_batch(batch["text"])
    # Get ids and convert to julia.
    # Also go from python 0-indexing to julia 1-indexing
    batch_ids = map(x -> pyconvert(Vector{Int}, x.ids) .+ 1, encodings) 
    return MLUtils.batch(batch_ids) # Padded token sequences are batched in 2D array
end

train_dataset, val_dataset, test_dataset = get_dataset()
tokenizer, vocab_size = get_tokenizer(train_dataset);
model = Transformer(; vocab_size)
x = transform(train_dataset[1:10])
x = Flux.batch(x)
model(x)

train_loader = DataLoader(train_dataset, batchsize=512, shuffle=true, collate=false)
# train_loader = mapobs(transform, train_loader)
