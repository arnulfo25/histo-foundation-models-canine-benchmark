from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler
from PIL import Image
import io, base64, json, os, time, csv

MMP = "/Volumes/DATOS/modelos/Pathos-Gemma4/gemma-4-e2b-it.F16-mmproj.gguf"
MDL = "/Volumes/DATOS/modelos/Pathos-Gemma4/gemma-4-e2b-it.Q4_K_M.gguf"
IMGS = "/Volumes/DATOS/datasets/MIDOG++/imagenes"
CSV_GT = "/Volumes/DATOS/datasets/MIDOG++/datasets_xvalidation.csv"
OUT = os.path.expanduser("~/Desktop/Proyectos IA 2026/paper_histo_vet_MIDOG")
RESULTS = OUT + "/resultados_pathos.json"

# Leer slides reales del CSV, filtrar los que existen en disco, 10 por tumor
rows = list(csv.DictReader(open(CSV_GT), delimiter=";"))
def norm(t):
    t=t.lower()
    if "mast" in t: return "mast_cell"
    if "lymph" in t: return "lymphoma"
    if "lung" in t: return "lung_cancer"
    return None
GT = {"mast_cell":[], "lymphoma":[], "lung_cancer":[]}
for r in rows:
    if r["Species"]!="Canine": continue
    c = norm(r["Tumor"])
    if not c: continue
    s = int(r["Slide"])
    if os.path.exists("%s/%03d.tiff"%(IMGS,s)) and len(GT[c])<10:
        GT[c].append(s)
print("Slides seleccionados:", {k:len(v) for k,v in GT.items()})

PROMPT = ("This is a canine H&E histopathology image. Classify into exactly ONE: "
          "'mast cell tumor', 'lymphoma', or 'lung adenocarcinoma'. "
          "Reply with ONLY the diagnosis name, nothing else.")

def parse(resp):
    r = resp.lower()
    if "mast" in r: return "mast_cell"
    if "lymph" in r: return "lymphoma"
    if "lung" in r or "adenocarc" in r or "pulm" in r: return "lung_cancer"
    return "unknown"

print("Cargando Pathos-Gemma4 (VLM)...")
handler = Llava15ChatHandler(clip_model_path=MMP, verbose=False)
llm = Llama(model_path=MDL, chat_handler=handler, n_gpu_layers=-1,
            logits_all=True, verbose=False, n_ctx=2048)
print("Pathos listo.\n")

hechos = json.load(open(RESULTS)) if os.path.exists(RESULTS) else {}
t0=time.time()
n_total = sum(len(v) for v in GT.values()); done=0
for clase, slides in GT.items():
    for s in slides:
        if str(s) in hechos: done+=1; continue
        im = Image.open("%s/%03d.tiff"%(IMGS,s)).convert("RGB")
        W,H = im.size
        g = im.convert("L")
        votos=[]
        for (fx,fy) in [(0.3,0.3),(0.5,0.5),(0.7,0.7)]:
            x,y=int(W*fx),int(H*fy)
            patch = im.crop((x,y,x+1024,y+1024)).resize((448,448))
            buf=io.BytesIO(); patch.save(buf,format="JPEG",quality=85); b64=base64.b64encode(buf.getvalue()).decode()
            msg={"role":"user","content":[
                {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,"+b64}},
                {"type":"text","text":PROMPT}]}
            try:
                r=llm.create_chat_completion([msg],max_tokens=12,temperature=0.0)
                resp=r["choices"][0]["message"]["content"].strip()
                pred=parse(resp)
                votos.append(pred)
            except Exception as e:
                votos.append("error")
        from collections import Counter
        final = Counter(votos).most_common(1)[0][0]
        hechos[str(s)]={"gt":clase,"pred":final,"votos":votos}
        done+=1
        ok=sum(1 for v in hechos.values() if v["gt"]==v["pred"])
        print("[%2d/%d] slide %03d [%s] -> %s (votos:%s) | acc %.0f%% (%d/%d)"%(
            done,n_total,s,clase,final,votos,100*ok/len(hechos),ok,len(hechos)))
        json.dump(hechos, open(RESULTS,"w"))

json.dump(hechos, open(RESULTS,"w"))
ok=sum(1 for v in hechos.values() if v["gt"]==v["pred"])
print("\n=== COMPLETADO en %ds ==="%(time.time()-t0))
print("Accuracy Pathos: %.1f%% (%d/%d)"%(100*ok/len(hechos),ok,len(hechos)))
