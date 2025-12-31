This is a RWFR Repo

# Download Guide

1. stage1 weights
    !wget https://huggingface.co/lxq007/DiffBIR/resolve/main/face_swinir_v1.ckpt
    
3. sd v2.1 base weights
    !wget https://huggingface.co/Manojb/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.safetensors

# correctly loaded or not , run this file to check
    from safetensors.torch import load_file
    import torch
    
    ckpt = load_file("/kaggle/working/Code-Diff-Lo-RWFR/v2-1_512-ema-pruned.safetensors")
    
    torch.save(
        {"state_dict": ckpt, "global_step": 0},
        "v2-1_512-ema-pruned.ckpt"
    )
    
    print("✅ Converted safetensors → ckpt")
    sd = torch.load("/kaggle/working/Code-Diff-Lo-RWFR/v2-1_512-ema-pruned.ckpt", map_location="cpu", weights_only=False)
    print(type(sd), sd.keys())

# expected output 
    ✅ Converted safetensors → ckpt
    <class 'dict'> dict_keys(['state_dict', 'global_step'])


# HOW TO TRAIN STAGE-2
    1. clone github repo
    2. download stage-1 and stage-2 stable diffusion v2.1 weights (given above)
    3. install req.txt + torch setup given in req.txt ( if you have new gpu's RTX , Ampere Based use xformer also)
    4. make sure LQ,GT AND CF have same image names
    5. configure train_stage2.yaml for changing train setting mostly the "train:..." section
    6. run using !accelerate launch train_stage2.py --config configs/train/train_stage2.yaml (make sure to be inside the repo dir.)

# HOW TO RUN INFERENCE
    1. clone this github and CodeFormer Repo ( !git clone https://github.com/sczhou/CodeFormer)
    2. install req.txt given in this repo + torch setup + xformer if GPU supports
    3. go to codeformer repo dir , and install basic-sr using !python basicsr/setup.py develop
    4. download and arrange LQ and LoRA weights obtained from training rest weights like (STAGE-1 , STAGE-2 , CodeFormer will be automatically downloaded)
    5. run this :
                %cd ./Code-Diff-Lo-RWFR
                !python inference.py \
                  --task unaligned_face \
                  --upscale 2 \
                  --version v2 \
                  --sampler spaced \
                  --steps 50 \
                  --cfg_scale 4.0 \
                  --captioner none \
                  --pos_prompt "" \
                  --neg_prompt "low quality, blurry, low-resolution, noisy, unsharp, weird textures" \
                  --input /kaggle/working/CelebAdebug_20_celebA \
                  --output /kaggle/working/debug_lora_step200 \
                  --lora_path /kaggle/working/lora_0000200.pt \
                  --rank_unet 64 \
                  --rank_controlnet 16 \
                  --batch_size 1 \
                  --n_samples 1 \
                  --precision fp32 \
                  --device cuda

