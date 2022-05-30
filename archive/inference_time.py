# @title Test Function { run: "auto", vertical-output: true }
import torch
import random
from transformers import (
    BertConfig,
    BertForSequenceClassification
)


def inference_time_test(model):
    def generate_rand_ids(ids, seq_len):
        rand_input_ids = [101] + [random.choices(ids)[0] for _ in range(seq_len - 2)] + [102]
        return torch.tensor(rand_input_ids)

    def generate_fake_input(bsz, seq_len, vocab_size):

        ids = [i for i in range(0, vocab_size)]
        ids.remove(101)
        ids.remove(102)

        input_ids = torch.stack([generate_rand_ids(ids, seq_len) for _ in range(bsz)], dim=0).to(device)
        token_type_ids = torch.zeros([bsz, seq_len]).to(device)
        attention_mask = torch.ones([bsz, seq_len]).to(device)

        return {
            'input_ids': input_ids.int(),
            'token_type_ids': token_type_ids.int(),
            'attention_mask': attention_mask.int()
        }

    batch_size = 32  # @param {type:"slider", min:0, max:100, step:32}
    seq_len = 384  # @param {type:"slider", min:0, max:512, step:128}
    trials = 10  # @param {type:"slider", min:5, max:20, step:1}

    fake_x = generate_fake_input(batch_size, seq_len, config.vocab_size)

    import time

    # Warmup
    for _ in range(3):
        model(**fake_x)

    for _ in range(trials):
        s = time.time()
        model(**fake_x)
        t = time.time()

    del fake_x

    return (t - s) / trials


if __name__ == '__main__':
    #@title  { run: "auto" }
    use_gpu = True #@param {type:"boolean"}
    if use_gpu:
      device = 'cuda:0'
    else:
      device = 'cpu'


    hidden_size =  768#@param {type:"integer"}
    num_hidden_layers =  12#@param {type:"integer"}
    num_attention_heads = 12 #@param {type:"integer"}
    intermediate_size = 3072 #@param {type:"integer"}


    config = BertConfig(
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
    )

    with torch.no_grad():
         model = BertForSequenceClassification(config).to(device)
         baseline = inference_time_test(model)
         print("Inference time in secs: ", baseline)

    del model
    torch.cuda.empty_cache()