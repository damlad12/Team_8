import sys

import re
import jsonlines
import pandas as pd
import fire
import gradio as gr
import torch
torch.set_num_threads(1)
import transformers
import json
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

MAX_INT = sys.maxsize

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass

def preprocess_data(example):
    return {
        "instruction": example.get("user", ""),  # Use the 'user' column as instruction
        "input": example.get("system", ""),      # Use the 'system' column as input
        "output": example.get("assistant", "")   # Use the 'assistant' column as output
    }

def batch_data(data_list, batch_size=1):
    n = len(data_list) // batch_size
    batch_data = []
    for i in range(n - 1):
        start = i * batch_size
        end = (i + 1) * batch_size
        batch_data.append(data_list[start:end])

    last_start = (n - 1) * batch_size
    last_end = MAX_INT
    batch_data.append(data_list[last_start:last_end])
    return batch_data


def main(
    load_8bit: bool = False,
    base_model: str = "",
    lora_weights: str = "tloen/alpaca-lora-7b",
    batch_size: int = 1,
    share_gradio: bool = False,
    ranking_method = "",
):
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"

    # model_type = lora_weights.split('/')[-1]
    # model_name = '_'.join(model_type.split('_')[:2])

    # if model_type.find('book') > -1:
    #     train_sce = 'book'
    # else:
    #     train_sce = 'movie'
    
    # if test_data_path.find('book') > -1:
    #     test_sce = 'book'
    # else:
    #     test_sce = 'movie'
    
    # temp_list = model_type.split('_')
    # seed = temp_list[-2]
    # sample = temp_list[-1]
    start = 0
    end = MAX_INT
    if os.path.exists(result_json_data):
        f = open(result_json_data, 'r')
        data = json.load(f)
        f.close()
    else:
        data = dict()

    if not data.__contains__(train_sce):
        data[train_sce] = {}
    if not data[train_sce].__contains__(test_sce):
        data[train_sce][test_sce] = {}
    if not data[train_sce][test_sce].__contains__(model_name):
        data[train_sce][test_sce][model_name] = {}
    if not data[train_sce][test_sce][model_name].__contains__(seed):
        data[train_sce][test_sce][model_name][seed] = {}
    if data[train_sce][test_sce][model_name][seed].__contains__(sample):
        exit(0)
        # data[train_sce][test_sce][model_name][seed][sample] = 

    generation_config = GenerationConfig(
                temperature=0,
                top_p=0.1,
                top_k=10,
                num_beams=1,
            )

    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
            device_map={'': 0}
        )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

    tokenizer.padding_side = "left"
    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    generation_config = GenerationConfig(
            temperature=0.1,
            top_p=0.1,
            top_k=10,
            num_beams=1,
            max_new_tokens=300,
        )

    root_path = '/content/drive/MyDrive/EC523DL/RecRanker/RecRanker/conventional recommender/dataset/ml-100k/'
    # data_path = '/content/drive/MyDrive/EC523DL/RecRanker/RecRanker/conventional recommender/dataset/ml-100k/test_ml-100k_MF_pointwise.jsonl'
    # data_path = f'/content/drive/MyDrive/EC523DL/RecRanker/RecRanker/conventional recommender/dataset/{ranking_method}.jsonl'
    data_path = '/content/drive/MyDrive/EC523DL/RecRanker/RecRanker/conventional recommender/dataset/ml-100k/test_ml-100k_MF_pairwise.jsonl'
    INVALID_ANS = "[invalid]"
    res_ins = []
    res_answers = []
    problem_prompt = (
        "{instruction}"
    )
    with open(data_path, "r+", encoding="utf8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            temp_instr = problem_prompt.format(instruction=item["inst"])
            print(temp_instr)
            temp_instr = preprocess_data(temp_instr)
            res_ins.append(temp_instr)
    # print('res_ins', res_ins)
    res_ins = res_ins[start:end]
    res_answers = res_answers[start:end]
    print('lenght ====', len(res_ins))
    batch_res_ins = batch_data(res_ins, batch_size=batch_size)
    results = []
    res_completions = []
    idx = 0
    with torch.no_grad():
        for prompt in batch_res_ins:
            # print(f'prompt is: {prompt}')
            if isinstance(prompt, list):
                pass
            else:
                prompt = [prompt]

            completions = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
            output_ids = model.generate(
                input_ids=completions["input_ids"],
                attention_mask=completions["attention_mask"],
                max_new_tokens=300,
                temperature=0.1,     # Randomness in generation
                top_k=10,            # Limits the number of highest-probability tokens considered
                top_p=0.1,           # Limits token sampling to cumulative probability <= 0.1
                num_beams=1,         # Beam search disabled (uses sampling)
            )
            generated_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            for idx, text in enumerate(generated_text):
                local_idx = 'INDEX ' + str(idx) + ':'
                text = text.replace('\n', '').replace('    ', '')  # Clean up whitespace
                formatted_text = local_idx + text
                res_completions.append(formatted_text)
                # print(formatted_text)
                idx = idx + 1
    # print('res_completions', res_completions[:])
    def write_list_to_file(string_list, output_file):
        with open(output_file, 'w') as file:
            for item in string_list:
                file.write(item + '\n')
    df = pd.DataFrame(res_completions)
    df.to_csv(f'{root_path}/inference_results_for_{ranking_method}.txt', index=None, header=None)

            # completions = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
            # generation_output = model.generate(
            #     **completions,
            #     generation_config=generation_config,
            #     return_dict_in_generate=True,
            #     output_scores=True,
            #     max_new_tokens=128,
            #     # batch_size=batch_size,
            # )
            # s = generation_output.sequences
            # output = tokenizer.batch_decode(s, skip_special_tokens=True)
            # match = re.search(r"(?:Answer:|The answer is)\s*(?:The answer is\s*)?(-?\d+\.?\d*)", output[0], re.IGNORECASE)
            # if match:
            #     # Convert the matched number to float and return
            #     results.append(float(match.group(1)))
            # else:
            #     # Return None if no valid match is found
            #     print(f"Could not extract prediction from: {output}")
            #     results.append(None)
    # # for prompt in batch_res_ins:
    # #     print(f'prompt is: {prompt}')
    # #     if isinstance(prompt, list):
    # #         pass
    # #     else:
    # #         prompt = [prompt]
    # #     completions = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
    # #     print(f"completion is: {completions}")
    # #     for output in completions:
    # #         print(f"output is: {output}")
    # #         local_idx = 'INDEX ' + str(idx) + ':'
    # #         prompt = output.prompt
    # #         generated_text = output.outputs[0].text
    # #         print(generated_text)
    # #         generated_text = generated_text.replace('\n', '').replace('    ', '')
    # #         generated_text = local_idx + generated_text
    # #         res_completions.append(generated_text)
    # #         idx += 1
    # print('res_completions', res_completions[:])
    # def write_list_to_file(string_list, output_file):
    #     with open(output_file, 'w') as file:
    #         for item in string_list:
    #             file.write(item + '\n')
    # df = pd.DataFrame(results)
    # df.to_csv(f'{root_path}/inference_results_for_pointwise.txt', index=None, header=None)

if __name__ == "__main__":
    fire.Fire(main)