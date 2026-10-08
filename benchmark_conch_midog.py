import torch, numpy as np, json, csv, os, time
from PIL import Image
from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer

MODELO = "/Volumes/DATOS/modelos/CONCH/pytorch_model.bin"
IMGS = "/Volumes/DATOS/datasets/MIDOG++/imagenes"
CSV_GT = "/Volumes/DATOS/datasets/MIDOG++/datasets_xvalidation.csv"
OUT = os.path.expanduser("~/Desktop/Proyectos IA 2026/paper_histo_vet_MIDOG")
RESULTS = OUT + "/resultados_conch.json"

rows = list(csv.DictReader(open(CSV_GT), delimiter=";"))
def norm(t):
    t=t.lower()
    if "mast" in t: return "mast_cell"
    if "lymph" in t: return "lymphoma"
    if "lung" in t: return "lung_cancer"
    return "otros"
gt = {int(r["Slide"]): norm(r["Tumor"]) for r in rows if r["Species"]=="Canine"}

slides = sorted(s for s in gt if os.path.exists("%s/%03d.tiff" % (IMGS, s)))
print("Slides a procesar: %d" % len(slides))

CATS = [
    ("mast_cell", "canine mast cell tumor round cells with granulated eosinophilic cytoplasm and scattered eosinophils"),
    ("lymphoma", "malignant lymphoma monomorphic discohesive round blue lymphoid cells scant cytoplasm"),
    ("lung_cancer", "pulmonary adenocarcinoma lung neoplasm with glandular structures and acinar pattern"),
    ("sarcoma", "fibrosarcoma malignant spindle cells in interlacing bundles"),
    ("scc", "squamous cell carcinoma invasive nests with keratinization"),
    ("normal", "normal healthy tissue histology"),
    ("inflammation", "chronic inflammation lymphoplasmacytic infiltrate"),
    ("melanoma", "melanoma pigmented polygonal neoplastic cells with brown melanin"),
]
labels = [c[0] for c in CATS]; texts = [c[1] for c in CATS]

print("Cargando CONCH...")
model, preprocess = create_model_from_pretrained(model_cfg="conch_ViT-L-14", checkpoint_path=MODELO)
tokenizer = get_tokenizer()
DEVICE="mps"; model=model.to(DEVICE).eval()
with torch.no_grad():
    tk=tokenizer(texts, return_tensors="pt", padding=True)
    cf=model.encode_text(tk["input_ids"].to(DEVICE))
    cf=cf/cf.norm(dim=-1,keepdim=True)
print("CONCH listo.\n")

hechos = json.load(open(RESULTS)) if os.path.exists(RESULTS) else {}
t0=time.time()
for i,s in enumerate(slides,1):
    if str(s) in hechos: continue
    try:
        im=Image.open("%s/%03d.tiff" % (IMGS,s)).convert("RGB")
    except Exception as e:
        print("Error slide %d: %s"%(s,e)); continue
    W,H=im.size
    g=np.array(im.convert("L"))
    votos=np.zeros(len(labels)); n=0
    with torch.no_grad():
        for ii in range(4):
            for jj in range(4):
                ps=900
                x=(W//5)+ii*(ps//2); y=(H//5)+jj*(ps//2)
                if x+ps>W or y+ps>H: continue
                crop=im.crop((x,y,x+ps,y+ps))
                gris=g[y:y+ps,x:x+ps]
                if np.median(gris)>225: continue
                img_t=preprocess(crop).unsqueeze(0).to(DEVICE)
                f=model.encode_image(img_t)
                f=f/f.norm(dim=-1,keepdim=True)
                sim=(100.0*f@cf.T).softmax(dim=-1)
                votos+=sim[0].cpu().numpy(); n+=1
    if n>0: votos/=n
    pred=labels[int(np.argmax(votos))]
    hechos[str(s)]={"gt":gt[s],"pred":pred,"probs":{labels[k]:round(float(votos[k]),4) for k in range(len(labels))},"n":n}
    if i%5==0 or i==len(slides):
        json.dump(hechos, open(RESULTS,"w"))
        ok=sum(1 for v in hechos.values() if v["gt"]==v["pred"])
        elap=time.time()-t0
        print("[%3d/%d] %4ds | acc %.1f%% (%d/%d)" % (i,len(slides),elap,100*ok/len(hechos),ok,len(hechos)))

json.dump(hechos, open(RESULTS,"w"))
ok=sum(1 for v in hechos.values() if v["gt"]==v["pred"])
print("\n=== COMPLETADO en %ds ===" % (time.time()-t0))
print("Accuracy global: %.1f%% (%d/%d)" % (100*ok/len(hechos), ok, len(hechos)))
print("Resultados:", RESULTS)
