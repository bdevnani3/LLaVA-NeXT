import argparse
import torch
from tqdm import tqdm

from llava.model.builder import load_pretrained_model

from PIL import Image
import math
import numpy as np
import time
import csv



def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def eval_model(args):
    
    # Model
    pretrained = 'liuhaotian/llava-v1.6-mistral-7b'
    model_name = 'llava-v1.6-mistral-7b'
    tokenizer, \
        model, \
            image_processor, \
                context_len = load_pretrained_model(pretrained,
                                                    None,
                                                    model_name=model_name,
                                                    device_map="auto",
                                                    multimodal=True,
                                                    use_flash_attention_2=False,
                                                    attn_implementation='sdpa')
    

    batch_size = 1
    runs = 50
    vis_tokens = 2048
    system_tokens = 34
    text_tokens = 100
    total_tokens = vis_tokens + system_tokens + text_tokens
    print(f"******Total tokens: {total_tokens}")

    attention_mask = torch.ones((batch_size, total_tokens), dtype=torch.bool).to('cuda:0')
    inputs_embeds = torch.empty((batch_size, total_tokens, 4096)).uniform_(-30.0, 30.0).to(torch.half).to('cuda:0')
    token_indices = [[system_tokens, vis_tokens, text_tokens] for _ in range(batch_size)]

    kwargs = {'pad_token_id': 0, 'do_sample': False, 'temperature': 0, 'top_p': None, 'num_beams': 1, 'max_new_tokens': 1, 'use_cache': True}

    time_dict = {}

    with torch.inference_mode():
        for r in tqdm([0,1,2,4,8,16,32,64,128,256,512, 1024, 2048]):
        # for r in tqdm([64]):
            time_dict[r] = []
            for i in range(runs+30):
                # import pdb; pdb.set_trace()
                if i < 30: # Throw the first n runs away
                    _ = super(model.__class__, model).generate(position_ids=None, 
                                                            attention_mask=attention_mask, 
                                                            inputs_embeds=inputs_embeds, 
                                                            token_indices=token_indices,
                                                            r=r,
                                                            tome_consistent_sizing=args.tcs,
                                                            **kwargs)
                    continue
                start = time.time()
                _ = super(model.__class__, model).generate(position_ids=None, 
                                                        attention_mask=attention_mask, 
                                                        inputs_embeds=inputs_embeds, 
                                                        token_indices=token_indices,
                                                        r=r,
                                                        tome_consistent_sizing=args.tcs,
                                                        **kwargs)
                end = time.time()
                time_taken = np.round(end - start, 4)
                time_dict[r].append(time_taken)
                # print(f"Time taken for r={r}: {time_taken}")
            print(f"--------Average time taken for r={r}: {np.mean(time_dict[r])}--------")
            print(f"--------Standard deviation for r={r}: {np.std(time_dict[r])}--------")
            print()
            #save the time_dict to csv
            # all_averages = [np.round(np.mean(time_dict[r]), 4) for r in time_dict.keys()]
            # print(f"--------All averages: {all_averages}--------")

        print("Saving to csv")
        with open(f"mistral_time_benchmarking_r_tcs{args.tcs}_{vis_tokens}.csv", "w") as f:
            for r in time_dict.keys():
                writer = csv.writer(f)
                writer.writerow([r]+time_dict[r])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-tcs", action="store_true", help="Use Tome Consistent Sizing")
    args = parser.parse_args()

    eval_model(args)