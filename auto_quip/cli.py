import click

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from auto_quip.quantizer import QuipQuantizer

@click.command()
@click.option('--model_name', required=True,
              help='Name or path of the model to be quantized.')
@click.option('--quant_dir', required=True,
              help='Directory to save the quantized model.')
@click.option('--codebook', default="E8P12",
              type=click.Choice(["E8P12", "E8P12RVQ3B", "E8P12RVQ4B",
                                 "D4", "HI"], case_sensitive=False),
              help='Codebook to be used for quantization. Choices are '
                   'E8P12 (2bit), E8P12RVQ3B (3bit), E8P12RVQ4B (4bit), '
                   'D4 (2bit), HI (4bit). Default: E8P12.')
@click.option('--dataset', default="wikitext2",
              type=click.Choice(["wikitext2", "c4", "ptb"],
                                case_sensitive=False),
              help='Dataset to be used for quantization. Choices are '
                   'wikitext2, c4, ptb. Default: wikitext2.')
@click.option('--nsamples', required=True, default=4096,
              type=click.IntRange(1, 65536),
              help='Number of samples to be used for quantization. '
                   'Default: 4096.')
@click.option('--quip_tune_iters', default=10,
              type=click.IntRange(1, 1024),
              help='Greedy update passes of the algorithm. '
                   'Higher is slower but yields slightly better results. '
                   'Default: 10.')
@click.option('--modules_to_not_convert', type=str,
              help="Comma separated list of modules to not convert to "
              "quantized modules. Useful for MoE models where the gate "
              "layer is not quantized.")
@click.option('--merge_suv', is_flag=True, default=False,
              help="A trick to cancel out some vectors to reduce "
              "calculation. Only supports Llama, Mixtral, and Qwen "
              "models. Default: False.")
def quantize_model(model_name,
                   quant_dir,
                   codebook,
                   dataset,
                   nsamples,
                   quip_tune_iters,
                   modules_to_not_convert,
                   merge_suv):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

    quant = QuipQuantizer(codebook=codebook,
                          dataset=dataset,
                          nsamples=nsamples,
                          quip_tune_iters=quip_tune_iters,
                          modules_to_not_convert=modules_to_not_convert,
                          merge_suv=merge_suv)
    quant_model = quant.quantize_model(model, tokenizer)
    quant.save(quant_model, quant_dir)
    tokenizer.save_pretrained(quant_dir)

def main():
    quantize_model()

if __name__ == "__main__":
    main()
