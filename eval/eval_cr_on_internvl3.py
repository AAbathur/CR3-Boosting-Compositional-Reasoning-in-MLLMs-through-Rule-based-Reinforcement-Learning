import os
import re
import json
from tqdm import tqdm

import pandas as pd
import torch
import torchvision.transforms as T
#from decord import VideoReader, cpu

from PIL import Image
from sklearn.utils import shuffle
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer


import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--model-path", type=str, default=None)
parser.add_argument("--task", choices=['winoground','mmvp','cola', 'vsr', 'tally-qa-simple', 'tally-qa-complex'], )
parser.add_argument("--device", type=int)
parser.add_argument("--reason", type=str)
args = parser.parse_args()


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values



def load_model(model_path, device):
    
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        load_in_8bit=False,
        #low_cpu_mem_usage=True,
        use_flash_attn=True,
        device_map="cuda:{}".format(device),
        trust_remote_code=True,).eval()
    
    model = model.to("cuda:{}".format(device))

    generation_config = dict(max_new_tokens=1024, do_sample=False)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    return model, tokenizer, generation_config


def get_text2img_prompt(img_path1, img_path2, cap1, reasoning, vpath2pixel=None):
    if reasoning:
        QUESTION_TEMPLATE = "{Question}\nFirst output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer with First or Second."
    else:
        QUESTION_TEMPLATE = "{Question}\nOutput the final answer with First or Second."
    
    question = "Which image best match the below caption?\nCaption:{}".format(cap1)
    prompt = QUESTION_TEMPLATE.format(Question=question)
    prompt = "First: <image>/nSecond: <image>/n" + prompt

    if vpath2pixel:
        pixel_values1 = vpath2pixel[img_path1]
        pixel_values2 = vpath2pixel[img_path2]
    else:
        pixel_values1 = load_image(img_path1, max_num=12).to(torch.bfloat16)
        pixel_values2 = load_image(img_path2, max_num=12).to(torch.bfloat16)
    num_patches_list = [pixel_values1.size(0), pixel_values2.size(0)]
    pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)

    return prompt, pixel_values, num_patches_list

def get_img2text_prompt(img_path1, cap1, cap2, reasoning):
    if reasoning:
        QUESTION_TEMPLATE = "{Question}\nFirst output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer with the option's letter A or B."
    else:
        ## W/O reasoning, directly prompt
        QUESTION_TEMPLATE = "{Question}\nOutput the final answer with the option's letter A or B."
    question = "Which caption best describe the given image?\nA.{}\nB.{}".format(cap1, cap2)
    prompt = QUESTION_TEMPLATE.format(Question=question)
    prompt = "<image>/n"+prompt

    pixel_values1 = load_image(img_path1, max_num=12).to(torch.bfloat16)
    num_patches_list = [pixel_values1.size(0)]
    return prompt, pixel_values1, num_patches_list




def get_itm_prompt(img_path, text1, reasoning):
    if reasoning:
        QUESTION_TEMPLATE = "{Question}\nFirst output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer with Yes or No."
        question = "Does the below caption precisely describe the given image?\nCaption: {}".format(text1)
    else:
        QUESTION_TEMPLATE = "{Question}\nOutput the final answer with Yes or No."
        question = "Does the below caption precisely describe the given image?\nCaption: {}".format(text1)
    prompt = QUESTION_TEMPLATE.format(Question=question)
    prompt = "<image>/n" + prompt

    pixel_values1 = load_image(img_path, max_num=12).to(torch.bfloat16)
    num_patches_list = [pixel_values1.size(0)]
    return prompt, pixel_values1, num_patches_list





def get_vqa_prompt(image_path1, ques, reasoning):
    if reasoning:
        QUESTION_TEMPLATE = "{Question}\nFirst output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Please try to answer the question with short words or phrases if possible."
        question = ques
    ## W/O reasoning, directly prompt
    else:
        QUESTION_TEMPLATE = "{Question}\nPlease try to answer the question with short words or phrases if possible."
        question = ques
    prompt = QUESTION_TEMPLATE.format(Question=question)
    prompt = "<image>/n" + prompt
    
    pixel_values1 = load_image(image_path1, max_num=12).to(torch.bfloat16)
    num_patches_list = [pixel_values1.size(0)]
    return prompt, pixel_values1, num_patches_list



def load_winoground():
    inpath = "/your/local/data/path//Winoground-data/data_examples.jsonl"
    img_path_temp = "/your/local/data/path//Winoground-data/images/{}.png" 
    all_data = []
    with open(inpath, "r", encoding="utf-8") as f1:
        for i, line in enumerate(f1):
            cont = json.loads(line.rstrip())
            cap0 = cont['caption_0']
            cap1 = cont['caption_1']
            img0_path = img_path_temp.format(cont['image_0'])
            img1_path = img_path_temp.format(cont['image_1'])
            #{"id": cont['id'], "cap0": cap0, "cap1": cap1, "img0_path": img0_path, "img1_path": img1_path}
            #yield cap0, cap1, img0_path, img1_path 
            all_data.append([cap0, cap1, img0_path, img1_path])
    return all_data
            

def load_cola():
    def get_img_path(img_url, img_dir):
        img_name = img_url.split("/")[-1]
        return os.path.join(img_dir, img_name)
    path1 = "/your/local/data/path//Cola/COLA_multiobjects_matching_benchmark.json"
    img_dir = "/your/local/data/path//Cola/cola-multi-obj-images"
    
    all_data = []
    with open(path1, "r", encoding="utf-8") as f1:
        cont = json.load(f1)
        for i,ct in enumerate(cont):
            #if i>3: break
            url0, cap0, url1, cap1 = ct
            img0_path = get_img_path(url0, img_dir)
            img1_path = get_img_path(url1, img_dir)
            all_data.append([cap0, cap1, img0_path, img1_path])
            
    return all_data


def extract_answer(content):
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
    if content_answer_match:
        content_answer = content_answer_match.group(1).strip()
        return content_answer
    return None


##########################################################################################################################################

def process_winoground_for_internvl3(ds, REASON=True):
    if ds == "winoground":
        print('load winoground')
        all_data = load_winoground()
        nhalf = 200
        vpath2pixel = None

    elif ds == "cola":
        all_data = load_cola()
        nhalf = len(all_data) // 2
        vpath2pixel = None

    all_sample, all_label = [], []

    all_i2t_mark = [0 for _ in range(nhalf)] + [1 for _ in range(nhalf)]
    all_i2t_mark = shuffle(all_i2t_mark, random_state=1123)
    all_t2i_mark = [1 for _ in range(nhalf)] + [0 for _ in range(nhalf)]
    all_t2i_mark = shuffle(all_t2i_mark, random_state=1234)

    all_new_data = []

    for i, data in tqdm(enumerate(all_data), total=len(all_data)):
        cap0, cap1, img0_path, img1_path = data
        all_new_data.append([cap0, cap1, img0_path, img1_path])
        i2t_mark = all_i2t_mark[i]
        t2i_mark = all_t2i_mark[i]
        #### pos-first
        if i2t_mark == 1:
            img0_cap = get_img2text_prompt(img0_path, cap0, cap1, reasoning=REASON, vpath2pixel=vpath2pixel)
            img1_cap = get_img2text_prompt(img1_path, cap0, cap1, reasoning=REASON, vpath2pixel=vpath2pixel)
            label_i0 = "A"
            label_i1 = "B"
        else:
            img0_cap = get_img2text_prompt(img0_path, cap1, cap0, reasoning=REASON, vpath2pixel=vpath2pixel)
            img1_cap = get_img2text_prompt(img1_path, cap1, cap0, reasoning=REASON, vpath2pixel=vpath2pixel)
            label_i0 = "B"
            label_i1 = "A"

        if t2i_mark == 1:
            cap0_img = get_text2img_prompt(img0_path, img1_path, cap0, reasoning=REASON, vpath2pixel=vpath2pixel)
            cap1_img = get_text2img_prompt(img0_path, img1_path, cap1, reasoning=REASON, vpath2pixel=vpath2pixel)
            label_c0 = "first"
            label_c1 = "second"
        else:
            cap0_img = get_text2img_prompt(img1_path, img0_path, cap0, reasoning=REASON, vpath2pixel=vpath2pixel)
            cap1_img = get_text2img_prompt(img1_path, img0_path, cap1, reasoning=REASON, vpath2pixel=vpath2pixel)
            label_c0 = "second"
            label_c1 = "first"
        all_sample.append([img0_cap, img1_cap, cap0_img, cap1_img])
        all_label.append([label_i0, label_i1, label_c0, label_c1])


    return all_sample, all_label, all_new_data



def format_check(content):
    pattern = r"<think>.*?</think>\s*<answer>\b(A|B|First|Second|Yes|No)\b</answer>"
    match = re.fullmatch(pattern, content, re.DOTALL)
    return match

def accuracy_on_winoground_v2(all_preds, all_label, all_sample, extract_ans=True):

    nbad, total = 0, 0
    i2t_count, t2i_count, group_count = 0, 0, 0
    for i, (img0_i2t, img1_i2t, cap0_t2i, cap1_t2i) in enumerate(all_preds):
        q1, q2, q3, q4 = all_sample[i]
        
        total += 1
        img0_i2t_label, img1_i2t_label, cap0_t2i_label, cap1_t2i_label = all_label[i]
        

        if extract_ans:
            
            img0_i2t_ans = extract_answer(img0_i2t)
            check1 = format_check(img0_i2t)

            img1_i2t_ans = extract_answer(img1_i2t)
            check2 = format_check(img1_i2t)

            cap0_t2i_ans = extract_answer(cap0_t2i)
            check3 = format_check(cap0_t2i)

            cap1_t2i_ans = extract_answer(cap1_t2i)
            check4 = format_check(cap1_t2i)

            print(i)
            print("img0_i2t", repr(img0_i2t))
            print("extract ans:", img0_i2t_ans, "format check:", check1)
            print("img1_i2t", repr(img1_i2t))
            print("extract ans:", img1_i2t_ans, "format check:", check2)
            print("cap0_t2i", repr(cap0_t2i))
            print("extract ans:", cap0_t2i_ans, "format check: ", check3)
            print("cap1_t2i", repr(cap1_t2i))
            print("extract ans:", cap1_t2i_ans, "format check:", check4)
            print("***********************************")
        else:
            img0_i2t_ans = img0_i2t
            img1_i2t_ans = img1_i2t
            cap0_t2i_ans = cap0_t2i
            cap1_t2i_ans = cap1_t2i

        if (not img0_i2t_ans) or (not img1_i2t_ans) or (not cap0_t2i_ans) or (not cap1_t2i_ans):
            nbad += 1
            continue
        flag0, flag1 = False, False
        if (img0_i2t_ans == img0_i2t_label) and (img1_i2t_ans == img1_i2t_label):
            i2t_count += 1
            flag0 = True
        if (cap0_t2i_ans.lower() == cap0_t2i_label) and (cap1_t2i_ans.lower() == cap1_t2i_label):
            t2i_count += 1
            flag1 = True
        if total < 50:
            print(total)
            print("pred_ans: ", repr(img0_i2t_ans), repr(img1_i2t_ans), repr(cap0_t2i_ans), repr(cap1_t2i_ans))
            print("label", img0_i2t_label, img1_i2t_label, cap0_t2i_label, cap1_t2i_label)
            print("img2text correct:", flag0, "text2img correct:", flag1)

        #print("\t image2text: {}, text2image: {}".format(flag0, flag1))
        if flag0 and flag1:
            group_count += 1
    
    return nbad, i2t_count, t2i_count, group_count, total


def save_pred_res(all_preds, all_label, outfile):
    res = [all_preds, all_label]
    with open(outfile, "w", encoding="utf-8") as f1:
        json.dump(res, f1)
    print("save done: ", outfile)


def evaluate_on_winoground(ds, model_path, device_id, REASON):
    
    all_sample, all_label, all_data = process_winoground_for_internvl3(ds, REASON=REASON) 
    model, tokenizer, generation_config = load_model(model_path, device_id)


    generation_config = dict(max_new_tokens=1024, do_sample=True)
    
    all_outputs = []
    for i in tqdm(range(0, len(all_sample))):
        #if i>3: break
        print(i)
        single_res = []
        for j, sample in enumerate(all_sample[i]):
            prompt, pixel_values, num_patches_list = sample
            pixel_values = pixel_values.to("cuda:{}".format(device_id))
            response, _ = model.chat(tokenizer, pixel_values, prompt, generation_config=generation_config, num_patches_list=num_patches_list, history=None, return_history=True)
            single_res.append(response)
        all_outputs.append(single_res)
    print("num of sample", len(all_sample))

    ckpt = os.path.basename(model_path).split("-")[1]
    outfile = "results/internvl3_2b_grpo_numgen8_ckpt{}_{}.json".format(ckpt, ds)
    save_pred_res(all_outputs, all_label, outfile)
    
    
    nbad, i2t_count, t2i_count, group_count, total = accuracy_on_winoground_v2(all_outputs, all_label, all_sample, extract_ans=REASON)
    print("model_path", model_path)
    print("data: {}, REASON: {}".format(ds, REASON))
    print("nbad: {}, total: {}, bad_ratio: {:.4f}".format(nbad, total, nbad/total))
    ### image2text acc也就是winoground中的text_score
    print("image2text_count: {}, total: {}, image2text_acc: {:.4f}".format(i2t_count, total-nbad, i2t_count/(total-nbad)))
    ### text2image acc就是winoground中的image_score
    print("text2image_count: {}, total: {}, text2image_acc: {:.4f}".format(t2i_count, total-nbad, t2i_count/(total-nbad)))
    print("group_count: {}, total: {}, group_acc: {:.4f}".format(group_count, total-nbad, group_count/(total-nbad)))
    

def accuracy_on_winoground_itm(all_pred, all_label, extract_ans=True):
    total1, total2, total3, total4 = 0, 0, 0, 0
    count1, count2, count3, count4 = 0, 0, 0, 0
    
    for i in range(len(all_pred)):
        i0c0_pred, i0c1_pred, i1c1_pred, i1c0_pred = all_pred[i]
        i0c0_label, i0c1_label, i1c1_label, i1c0_label = all_label[i]
        print("i, pred", all_pred[i])
        print("i, label", all_label[i])
        if extract_ans:
            i0c0_pred = extract_ans(i0c0_pred)
            i0c1_pred = extract_ans(i0c1_pred)
            i1c1_pred = extract_ans(i1c1_pred)
            i1c0_pred = extract_ans(i1c0_pred)

        i0c0_pred = i0c0_pred.replace(".", "").lower()
        i0c1_pred = i0c1_pred.replace(".", "").lower()
        i1c1_pred = i1c1_pred.replace(".", "").lower()
        i1c0_pred = i1c0_pred.replace(".", "").lower()
        
        if i0c0_pred in ['yes', 'no']:
            total1 += 1
            if i0c0_pred == i0c0_label:
                count1 += 1
        if i0c1_pred in ['yes', 'no']:
            total2 += 1
            if i0c1_pred == i0c1_label:
                count2 += 1
        if i1c1_pred in ['yes', 'no']:
            total3 += 1
            if i1c1_pred == i1c1_label:
                count3 += 1
        if i1c0_pred in ['yes', 'no']:
            total4 += 1
            if i1c0_pred == i1c0_label:
                count4 += 1
    print("i0c0 count: {}, total: {}, acc: {:.3f}".format(count1, total1, count1/total1))
    print("i0c1 count: {}, total: {}, acc: {:.3f}".format(count2, total2, count2/total2))
    print("i1c1 count: {}, total: {}, acc: {:.3f}".format(count3, total3, count3/total3))
    print("i1c0 count: {}, total: {}, acc: {:.3f}".format(count4, total4, count4/total4))
    sum_count = count1 + count2 + count3 + count4
    sum_total = total1 + total2 + total3 + total4

    print("avg count: {}, total: {}, acc: {:.3f}".format(sum_count, sum_total, sum_count/sum_total))



########################################################################################################################################################

def get_mmvp_prompt(image_path, ques, option1, option2, reasoning=True):
    if reasoning:
        #QUESTION_TEMPLATE = "{Question}\nFirst output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Only output the final answer with the option's letter A or B."
        QUESTION_TEMPLATE = "{Question}\nMust first output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. The final result contains only the option letter A or B.."
        question = "{}\nA.{}\nB.{}".format(ques, option1, option2)
    else:
        QUESTION_TEMPLATE = "{Question}\nOutput the final answer with the option's letter A or B."
        question = "{}\nA.{}\nB.{}".format(ques, option1, option2)
    prompt = QUESTION_TEMPLATE.format(Question=question)
    prompt = "<image>/n" + prompt
    pixel_values1 = load_image(image_path, max_num=12).to(torch.bfloat16)
    num_patches_list = [pixel_values1.size(0)]
    return prompt, pixel_values1, num_patches_list


def process_mmvp_for_internvl3(reasoning):
    dir1 = "/your/local/data/path//MMVP"
    ques_file = os.path.join(dir1, "Questions.csv")
    img_dir = os.path.join(dir1, "MMVP_Images")

    df = pd.read_csv(ques_file)
    all_prompt, all_label = [], []
    for idx, row in tqdm(df.iterrows(), total=300):
        answer = "A" if row["Correct Answer"] == "(a)" else "B"
        all_label.append(answer)
        option1, option2 = row["Options"].split(" (b) ")
        option1 = option1.split("(a) ")[1]
        ques = row['Question']
        img_path = os.path.join(img_dir, f"{idx+1}.jpg")
        prompt1 = get_mmvp_prompt(img_path, ques=ques, option1=option1, option2=option2, reasoning=reasoning)
        all_prompt.append(prompt1)
    return all_prompt, all_label


def accuracy_on_mmvp(all_pred, all_label, extract_ans=True):
    nbad, count, total = 0, 0, 0
    all_mark = []
    for content, label in zip(all_pred, all_label):
        total += 1
        if extract_ans:
            cont_ans = extract_answer(content)
            print("total", total)
            print("raw ans", repr(content))
            print("extract ans", cont_ans)
        else:
            cont_ans = content
        if not cont_ans:
            nbad += 1
            all_mark.append(0)
        elif cont_ans == label:
            count += 1
            all_mark.append(1)
        else:
            all_mark.append(0)
    print("MMVP, nbad: {}, total: {}, count: {}, Single ACC: {:.4f}".format(nbad, total, count, count/total))
    pair_count, pair_total = 0, 0
    for i in range(0, len(all_mark), 2):
        pair_total += 1
        mark1 = all_mark[i]
        mark2 = all_mark[i+1]
        if mark1 == mark2 == 1:
            pair_count += 1
    print("pair count: {}, pair_total: {} Pair Acc: {:.4f}".format(pair_count, pair_total, pair_count/pair_total))



def evaluate_on_mmvp(model_path, device_id, REASON):
    model, tokenizer, generation_config = load_model(model_path, device_id)

    all_sample, all_label = process_mmvp_for_internvl3(reasoning=REASON)

    all_outputs = []
    for i in tqdm(range(0, len(all_sample))):
        #if i>10: break
        prompt, pixel_values, num_patches_list = all_sample[i]
        pixel_values = pixel_values.to("cuda:{}".format(device_id))
        response, _ = model.chat(tokenizer, pixel_values, prompt, generation_config=generation_config, num_patches_list=num_patches_list, history=None, return_history=True)
        all_outputs.append(response)
    accuracy_on_mmvp(all_outputs, all_label, extract_ans=REASON)
    print("Reason", REASON, "model_path", model_path)


###########################################################################################################################################################################
###############          Tally-QA
###########################################################################################################################################################################


def tally_accuracy(all_pred, all_label, extract_ans):
    nbad, count, total = 0, 0, 0
    all_mark = []
    for content, label in zip(all_pred, all_label):
        total += 1
        if extract_ans:
            cont_ans = extract_answer(content)
        else:
            cont_ans = content
        label = str(label)
        if not cont_ans:
            nbad += 1
            all_mark.append(0)
        elif cont_ans == label:
            count += 1
            all_mark.append(1)
        else:
            all_mark.append(0)
    print("Tally, nbad: {}, total: {}, count: {}, ACC: {:.4f}".format(nbad, total, count, count/total))


def get_tally_qa_data(source, split):
    if split == "train":
        path1 = "/your/local/data/path//TallyQA-official/train.json"
    elif split == "dev":
        path1 = "/your/local/data/path//TallyQA-official/dev.json"
    elif split == "test":
        path1 = "/your/local/data/path//TallyQA-official/test.json"
    select_data = []
    with open(path1, "r", encoding="utf-8") as f1:
        cont = json.load(f1)
        print(len(cont))
        for i, ct in enumerate(cont):
            ds = ct['data_source']
            if ds == source:
                select_data.append(ct)
    return select_data


def get_internvl3_tally_res(model_path, device_id, REASON, type):
    model, tokenizer, generation_config = load_model(model_path, device_id)

    img_dir = "/your/local/visual_genome/path/vg"
    if type == "simple":
        all_data = get_tally_qa_data("imported_genome", split="test")
    elif type == "complex":
        all_data = get_tally_qa_data('amt', split="test")
    else:
        raise KeyError("dont support type: {}".format(type))

    all_data = shuffle(all_data, random_state=1234)
    all_data = all_data[:2000]

    all_prompt, all_label = [], []
    for i, data in tqdm(enumerate(all_data), total=len(all_data)):
        ques = data['question']
        answer = data['answer']
        #img = data['image']
        img_path = os.path.join(img_dir, data['image'])
        prompt = get_vqa_prompt(img_path, ques, reasoning=REASON)
        all_prompt.append(prompt)
        all_label.append(answer)
    
    all_outputs = []
    for i in tqdm(range(0, len(all_prompt))):
        #if i>10: break
        prompt, pixel_values, num_patches_list = all_prompt[i]
        pixel_values = pixel_values.to("cuda:{}".format(device_id))
        response, _ = model.chat(tokenizer, pixel_values, prompt, generation_config=generation_config, num_patches_list=num_patches_list, history=None, return_history=True)
        all_outputs.append(response)
        if i<10:
            print(i, prompt)
            print("output", response)

    tally_accuracy(all_outputs, all_label, extract_ans=REASON)
    print("REASON:", REASON, "model_path", model_path, "type", type)


###########################################################################################################################################################################
###############          VSR
###########################################################################################################################################################################

def process_vsr_for_internvl3(reasoning, split):
    if split == "train":
        path1 = "/your/local/data/path//VSR-random/train.jsonl" ## train: 7680条数据，7386条数据的img在img_dir中
    elif split == "dev":
        path1 = "/your/local/data/path//VSR-random/dev.jsonl"  ## dev: 1097条数据，1067条数据的img在img_dir中
    elif split == "test":
        path1 = "/your/local/data/path//VSR-random/test.jsonl" ## test: 2195条数据，2113条数据的img在img_dir中；

    img_dir = "/your/local/coco_2017/path/coco/train2017"
    
    total, count = 0, 0
    all_prompt, all_label = [], []
    with open(path1, "r", encoding="utf-8") as f1:
        for i, line in enumerate(f1):
            #if i>10: break
            total += 1
            cont = json.loads(line.rstrip())
            img_path = os.path.join(img_dir, cont['image'])
            img_link = cont['image_link']
            if "val2017" in img_link:
                count += 1
                continue
            cap = cont['caption']
            label = cont['label']
            if label == 1:
                answer = 'yes'
            else:
                answer = 'no'
            prompt = get_itm_prompt(img_path, cap, reasoning)
            all_prompt.append(prompt)
            all_label.append(answer)
    print("total: {}, abandon num: {}".format(total, count))
    return all_prompt, all_label

def accuracy_on_vsr(all_pred, all_label, extract_ans=True):
    nbad, count, total = 0, 0, 0
    all_mark = []
    all_clean_pred = []
    for content, label in zip(all_pred, all_label):
        total += 1
        if extract_ans:
            cont_ans = extract_answer(content)
            print("total", total)
            print("raw ans", repr(content))
            print("extract ans", cont_ans)
        else:
            cont_ans = content
        cont_ans = cont_ans.replace(".", "")
        all_clean_pred.append(cont_ans.lower())
        if not cont_ans:
            nbad += 1
            all_mark.append(0)
        elif cont_ans.lower() == label:
            count += 1
            all_mark.append(1)
        else:
            all_mark.append(0)
    print("VSR, nbad: {}, total: {}, count: {}, Single ACC: {:.4f}".format(nbad, total, count, count/total))
    return all_clean_pred

def evaluate_on_vsr(model_path, device_id, REASON):
    #split = "train"
    split = "test"
    model, tokenizer, generation_config = load_model(model_path, device_id)

    all_sample, all_label = process_vsr_for_internvl3(reasoning=REASON, split=split)

    all_outputs = []
    for i in tqdm(range(0, len(all_sample))):
        #if i>10: break
        prompt, pixel_values, num_patches_list = all_sample[i]
        pixel_values = pixel_values.to("cuda:{}".format(device_id))
        response, _ = model.chat(tokenizer, pixel_values, prompt, generation_config=generation_config, num_patches_list=num_patches_list, history=None, return_history=True)
        all_outputs.append(response)
        if i<10:
            print(i, prompt)
            print("output", response)
    
    accuracy_on_vsr(all_outputs, all_label, extract_ans=REASON)
    print("Split:", split, "REASON:", REASON, "model_path:", model_path)



def main():
    if args.reason == "False":
        xreason = False
    elif args.reason == "True":
        xreason = True

    if args.task == "winoground":
        evaluate_on_winoground('winoground', args.model_path, device_id=args.device, REASON=xreason)
    elif args.task == "cola":
        evaluate_on_winoground('cola', args.model_path, device_id=args.device, REASON=xreason)
    elif args.task == "mmvp":
        evaluate_on_mmvp(args.model_path, device_id=args.device, REASON=xreason)
    elif args.task == "vsr":
        evaluate_on_vsr(args.model_path, device_id=args.device, REASON=xreason)
    elif args.task == "tally-qa-simple":
        get_internvl3_tally_res(args.model_path, args.device, REASON=xreason, type="simple")
    elif args.task == "tally-qa-complex":
        get_internvl3_tally_res(args.model_path, args.device, REASON=xreason, type="complex")

    print("Xreason", xreason)
    print("args", args)

if __name__ == "__main__":
    main()
    
