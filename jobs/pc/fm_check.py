"""Print each foundation model's normalisation, prefix tokens and feature size as FMNet builds them."""
import sys; sys.path.insert(0, r"C:\midog\repo\axel_model")
import torch, fm_lora
for fm in ("hopt0", "virchow2", "uni", "uni2h"):
    m = fm_lora.FMNet(adversarial=False, checkpointing=False, fm=fm).to("cuda").eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        f = m.features(torch.zeros(2, 3, 224, 224, device="cuda", dtype=torch.bfloat16))
    print(f"{fm:9s} mean {tuple(round(v, 4) for v in m.mean.flatten().tolist())} std {tuple(round(v, 4) for v in m.std.flatten().tolist())} "
          f"prefix_tokens {m.enc.num_prefix_tokens} embed {m.enc.embed_dim} feature {tuple(f.shape)} global_pool '{getattr(m.enc, 'global_pool', '?')}'")
    del m; torch.cuda.empty_cache()
