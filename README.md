This is a RWFR Repo

# Download Guide

    1. stage1 weights
    !wget https://huggingface.co/lxq007/DiffBIR/resolve/main/face_swinir_v1.ckpt
    
    2. sd v2.1 base weights
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


