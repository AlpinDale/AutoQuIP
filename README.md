# QuIP

Based on the great works of [Cornell-RelaxML/quip-sharp](https://github.com/Cornell-RelaxML/quip-sharp) (the original implementation) and [chu-tianxiang/QuIP-for-all](https://github.com/chu-tianxiang/QuIP-for-all) (majority of this repo).

AutoQuIP is designed to provide an easy API for quantizing and loading QuIP# models.

The checkpoints provided here are incompatible with the original team's, due to a few factors:

* Every linear layer is quantized seperately without fusions (i.e. concatenating QKV layers)
* The packing format is slightly different, weights are simply packed as `(outdim, indim / codesz)` shape without complex permuting.

## Usage

Clone the repository, and install via pip:
```
git clone https://github.com/AlpinDale/AutoQuIP

pip install -e .
```

### Quantize

Please refer to the [author's blog](https://cornell-relaxml.github.io/quip-sharp/) for a thorough explanation of the QuIP# algorithm.

After installation, you can use the CLI app `auto_quip` to quantize a model. Simply run 
`auto_quip --help` for a list of options.

If you wish to do this programmatically via the API, you can use the following example:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_quip import QuipQuantizer

model_name = "meta-llama/Llama-2-70b-hf"
quant_dir = "llama-70b_2bit_quip"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

quant = QuipQuantizer(codebook="E8P12", dataset="c4", nsamples=4096)
quant_model = quant.quantize_model(model, tokenizer)
quant.save(quant_model, quant_dir)
tokenizer.save_pretrained(quant_dir)
```

Arguments for `QuipQuantizer` includes:
* `codebook`: the algorithm for quantization, including `E8P12`(2-bit), `E8P12RVQ3B`(3-bit), `E8P12RVQ4B`(4-bit), `D4`(2-bit), `HI`(4-bit). `D4` and `HI` are relatively inferior to E8P12-based methods.
* `dataset`: the data used for calibration, supports `c4`, `ptb`, `wikitext2`.
* `nsamples`: number of samples used for calibration, larger is slower. By default 4096 samples of calibration data will be used as [suggested by the author](https://github.com/Cornell-RelaxML/quip-sharp/issues/13#issuecomment-1848867522), which is very time consuming.
* `quip_tune_iters`: Greedy update passes of the algorithm, higher is slower but yields slightly better quanlity. Default to 10.
* `use_rand`: when the dim is not powers of 2, say `dim = 2^n * base`, use_rand will decompose it into `2^n` Hadamard matrix and `base x base` random orthogonal matrices, instead of decomposing to two Hadamard matrix in the original implementation which is not always feasible. Default to true.
* `modules_to_not_convert`: the name of layers not to quantize, useful for MOE models where gate layer is often unquantized.
* `merge_suv`: trick to cancel out some vectors to reduce calculation. Only support llama, mixtral and qwen. Default to false.


### Inference
```python
from transformers import AutoTokenizer
from auto_quip import load_quantized_model

quant_dir = "llama-70b_2bit_quip"

quant_model = load_quantized_model(quant_dir).cuda()
tokenizer = AutoTokenizer.from_pretrained(quant_dir)

input_ids = tokenizer.encode("The capital of France is", return_tensors="pt").cuda()
print(tokenizer.decode(quant_model.generate(input_ids, do_sample=True)[0]))
```