# TAROT: TEST-DRIVEN AND CAPABILITY-ADAPTIVE CURRICULUM REINFORCEMENT FINE-TUNING FOR CODE GENERATION

This the official repository of "TAROT: TEST-DRIVEN AND CAPABILITY-ADAPTIVE CURRICULUM REINFORCEMENT FINE-TUNING FOR CODE GENERATION". We provide scripts that are used to generate 4-tiered test suite of dataset, fine-tune models and evlaluate the models from the paper. Also, all the fine-tuned models and dataset can be found in Hugging Face.

## Generating 4-tiered test suite of dataset

The base dataset comes from Hugging Face's [open-r1/verifiable-coding-problems-python-v2](https://huggingface.co/datasets/open-r1/verifiable-coding-problems-python-v2). We used OpenAI's `o1-pro`, `o3`, and `o4-mini` to generate 4-tiered test suite for each data point. Final dataset contains 15.5K data points which has 4-tiered test suite per each (~ 62K test cases are included).

## Fine-tuning models

We experiment to verify the effectiveness of TAROT on 1.5B, 3B, and 7B variant of `Qwen2.5-Instruct` and `Qwen2.5-Coder-Instruct`, `Qwen3-4B-Instruct-2507`, and 2B and 9B variant of `Gemma2` models using GRPO algorithm.

To streamline the Reinfocement Fine-Tuning (RFT) and to make sure the consistency of the fine-tuning process across different model families, we used [dstack.ai](https://dstack.ai/) for provisioning cloud VMs, the Hugging Face's [`Open-R1`](https://github.com/huggingface/open-r1) to define high-level fine-tuning recipe, [`trl`](https://github.com/huggingface/trl) to to define custom rewarding system, [`vllm`](https://github.com/vllm-project/vllm) for faster generation for GRPO, and [`E2B`]() for safe code execution in the cloud sandbox.

Exact fine-tuning process is specified in the dstack YAML file, and you can it [here](./8x80GB.task.dstack.yml). To run such YAML, you need to run the following commands after setting up dstack which you can find the how-to guide [here](https://dstack.ai/docs/installation/).

```bash
$ export HF_TOKEN=[YOUR-HUGGING-FACE-TOKEN]
$ export HUGGINGFACE_TOKEN=[YOUR-HUGGING-FACE-TOKEN]
$ export WANDB_API_KEY=[YOUR-WANDB-API-KEY]
$ export E2B_API_KEY=[YOUR-E2B-API-KEY]

$ TARGET_BASE_MODEL=.... \ # base model (i.e., Qwen/Qwen2.5-1.5B-Instruct)
  TARGET_YAML=.... \ # RFT recipe (i.e., qwen2.5-1.5b/basic_only.yaml)
  TARGET_REWARD=.... \ # reward function (i.e., custom_rewards/basic_only.py)
  MAX_MODEL_LEN=.... \ # max length to gen (i.e., 4096)
  dstack apply -f 8x80GB.task.dstack.yml -d
```

The list of the `TARGET_YAML` can be found under `gemma2-2b-it`, `gemma2-9b-it`, `qwen2.5-1.5b`, `qwen2.5-3b`, `qwen2.5-7b`, `qwen2.5-coder-1.5b`, `qwen2.5-coder-3b`, `qwen2.5-coder-7b`, and `qwen3-4b-it` folders. And the list of custom reward functions can be found under `custom_rewards` folder as well.

## Evaluating models

All the fine-tuned models were evaluated on [HumanEval](https://github.com/openai/human-eval), [HumanEval+](https://github.com/evalplus/evalplus), [MBPP](https://arxiv.org/abs/2108.07732), [MBPP+](https://github.com/evalplus/evalplus), [LiveCodeBench_v5](https://livecodebench.github.io/), [CodeForces](https://arxiv.org/abs/2501.01257v1), and [CruxEval](https://arxiv.org/abs/2401.03065) using [EvalChemy](https://github.com/mlfoundations/evalchemy) framework to ensure the evaluation consistency.

Evaluation steps can be found in detailed at the EvalChemy official REAMDE, but here are the briefs for any convenient use.

```bash
# Create and activate conda environment
$ conda create --name evalchemy python=3.10
$ conda activate evalchemy

# Clone the repo
$ git clone https://github.com/mlfoundations/evalchemy.git
$ cd evalchemy

# Install dependencies
$ pip install -e .
$ huggingface-cli login

$ python -m eval.eval \
    --model vllm \
    --tasks HumanEval,HumanEvalPlus,MBPP,MBPPPlus,LiveCodeBenchv5_official,CodeForces,CruxEval \
    --model_args "pretrained=[TARGET-MODEL],tensor_parallele_size=[NUM-GPUs]" \
    --batch_size 16 \
    --output_path logs

```