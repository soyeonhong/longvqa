import requests
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import json

from torch.utils.data import DataLoader, Dataset
from transformers import SiglipVisionModel, SiglipImageProcessor, SiglipTextModel, SiglipTokenizer
from PIL import Image
from decord import VideoReader
from pathlib import Path
from collections import defaultdict
from einops import rearrange
from tqdm import tqdm

class HourVideoDataset(Dataset):
    def __init__(self, vid_path, desired_fps=1):
        self.vid_path = vid_path 
        self.desired_fps = desired_fps

        self.vr = None
        self.total_idx = None
        self.fps = None
        self.frame_idxs = None

    def __len__(self):
        if self.frame_idxs is None:
            self._initialize_video_reader()
        return len(self.frame_idxs)

    def __getitem__(self, idx): 
        if self.vr is None:
            self._initialize_video_reader()        
        batch_t = self.vr.get_batch([idx]).asnumpy()  # [T, H, W, C]
        
        return batch_t
    
    def _initialize_video_reader(self):
        if self.vr is None:
            self.vr = VideoReader(str(self.vid_path), num_threads=1)
            self.total_idx = len(self.vr)
            self.fps = self.vr.get_avg_fps()
            self.frame_idxs = [idx for idx in range(0, self.total_idx, round(self.fps / self.desired_fps))]
            
class HourVideoCLS(Dataset):
    def __init__(self, img_cls):
        self.img_cls = img_cls
        self.len = len(img_cls)
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        return self.img_cls[idx]

# Define worker initialization function
def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # Get the dataset instance
    dataset._initialize_video_reader()  # Initialize VideoReader in the worker
    
def process_annotation():
    p_anns = Path('/data2/local_datasets/HourVideo/v1.0_release/json/dev_v1.0_annotations.json')
    anns = json.loads(p_anns.read_text())
    vid_id_list = list(anns.keys())
    vid2questions = defaultdict(list)
    
    for vid_id in vid_id_list:
        datas = anns[vid_id]['benchmark_dataset']
        
        for data in datas:
            qid = data['qid']
            question = data['question']
            vid2questions[vid_id].append({'qid': qid, 'question': question})
        vid2questions[vid_id]
    
    return vid2questions

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    
    args = parser.parse_args()
    
    # path
    p_vid = Path('/local_datasets/HourVideo/videos')
    p_out = Path('/data/soyeon/longvqa/repos/lmms-eval/outputs/selection')
    p_out_cls = Path('/data/soyeon/longvqa/repos/lmms-eval/outputs/cls')
    p_out.mkdir(parents=True, exist_ok=True)
    p_out_cls.mkdir(parents=True, exist_ok=True)
    
    # process annotation
    vid2questions = process_annotation()
    vid_id_list = list(vid2questions.keys())
    vid_id_list = vid_id_list[args.rank::args.world_size]  # distribute video ids across ranks
    print(f"Rank {args.rank} processing {len(vid_id_list)} videos.")
    
    # model loading
    model_path = "google/siglip-so400m-patch14-384"
    vision_model = SiglipVisionModel.from_pretrained(model_path)
    vision_model.eval().cuda()
    
    text_model = SiglipTextModel.from_pretrained(model_path)
    text_model.eval().cuda()
    
    processor = SiglipImageProcessor.from_pretrained(model_path)
    tokenizer = SiglipTokenizer.from_pretrained(model_path)
    print("Model and processor loaded successfully.")
    
    for vid_id in vid_id_list:
        datas = vid2questions[vid_id]
        vid_feature = None
        
        vid_path = p_vid / f"{vid_id}.mp4"
        
        p_vid_cls_out = p_out_cls / f"{vid_id}.pt"
        
        if p_vid_cls_out.exists():
            vid_feature = torch.load(p_vid_cls_out)
        else:
            vid_dataset = HourVideoDataset(vid_path)
            vid_dataloader = DataLoader(vid_dataset,
                                        batch_size=args.batch_size,
                                        shuffle=False,
                                        num_workers=1,
                                        worker_init_fn=worker_init_fn,
                                        pin_memory=True)
            
            for idx, frames in tqdm(enumerate(vid_dataloader), total=len(vid_dataloader)):
                with torch.no_grad():
                    frames = rearrange(frames, 'b t h w c -> (b t) h w c')  # [batch * num_frames, H, W, C]
                    inputs = processor(images=frames, return_tensors="pt")
                    inputs = inputs["pixel_values"].cuda()
                    
                    outputs = vision_model(pixel_values=inputs)
                    image_cls = outputs.last_hidden_state[:, 0, :]  # [batch, hidden_dim]
                
                    if vid_feature is None:
                        vid_feature = image_cls
                    else:
                        vid_feature = torch.cat((vid_feature, image_cls), dim=0)
                
            torch.save(vid_feature.cpu().detach(), p_vid_cls_out)
            
        cls_dataset = HourVideoCLS(vid_feature.cpu())
        cls_dataloader = DataLoader(cls_dataset,
                                      batch_size=args.batch_size,
                                      shuffle=False,
                                      num_workers=2,
                                      pin_memory=True)
        
        for data in datas:
            sim_list = None
            qid = data['qid']
            question = data['question']
            
            p_q_out = p_out / f"{qid}.pt"
            if p_q_out.exists():
                continue
            
            text_inputs = tokenizer(question, return_tensors="pt")
            input_ids = text_inputs["input_ids"].cuda()
            
            with torch.no_grad():
                text_outputs = text_model(input_ids=input_ids)
                text_cls = text_outputs.last_hidden_state[:, 0, :]  # [batch, hidden_dim]
                
            for cls_idx, cls_feature in enumerate(cls_dataloader):
                cls_feature = cls_feature.cuda()
                sim = F.cosine_similarity(cls_feature, text_cls, dim=-1)
                
                if sim_list is None:
                    sim_list = sim
                else:
                    sim_list = torch.cat((sim_list, sim), dim=0)
                    
            torch.save(sim_list.cpu().detach(), p_out / f"{qid}.pt")
            
            print(f"Processed video {vid_id} for question {qid}. Similarity scores saved.")


if __name__ == "__main__":
    main()