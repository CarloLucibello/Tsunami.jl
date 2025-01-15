# Needs 
# pkg> add https://github.com/MurrellGroup/HuggingFaceTokenizers.jl
using Tsunami, Flux
import HuggingFaceTokenizers as HFT
import HuggingFaceDatasets as HFD
using PythonCall: PyList

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
    return tokenizer
    # return HFT.Tokenizer(tokenizer) # return the julia wrapper
end

model = Transformer()
train_dataset, val_dataset, test_dataset = get_dataset()
tokenizer = get_tokenizer(train_dataset)
