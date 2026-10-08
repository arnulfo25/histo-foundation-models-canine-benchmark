import json, os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.expanduser("~/Desktop/Proyectos IA 2026/paper_histo_vet_MIDOG")
conch = json.load(open(OUT+"/resultados_conch.json"))
pathos = json.load(open(OUT+"/resultados_pathos.json"))

NOMBRES = {"mast_cell":"Mast cell", "lymphoma":"Linfoma", "lung_cancer":"Pulmon"}
slides = sorted(int(s) for s in pathos.keys())
print("Slides comparados:", len(slides))

CLASES = ["mast_cell","lymphoma","lung_cancer"]

def metricas(res, slides):
    y_t=[res[str(s)]["gt"] for s in slides]
    y_p=[res[str(s)]["pred"] for s in slides]
    n=len(slides)
    acc=sum(t==p for t,p in zip(y_t,y_p))/n
    por_clase={}
    for c in CLASES:
        idx=[i for i,t in enumerate(y_t) if t==c]
        if not idx: continue
        rec=sum(1 for i in idx if y_p[i]==c)/len(idx)
        por_clase[c]=rec
    return acc, por_clase, y_t, y_p

a_c, pc_c, yt, yp_c = metricas(conch, slides)
a_p, pc_p, _, yp_p = metricas(pathos, slides)

print("\n"+"="*60)
print("COMPARATIVA CONCH vs Pathos-Gemma4 (30 slides caninos)")
print("="*60)
print("%-12s | %5s | %11s | %11s" % ("Tumor","N","CONCH","Pathos"))
print("-"*48)
for c in CLASES:
    n=sum(1 for t in yt if t==c)
    print("%-12s | %5d | %9.0f%%  | %9.0f%%" % (NOMBRES[c], n, 100*pc_c[c], 100*pc_c[c]*0+100*pc_p[c]))
print("-"*48)
print("%-12s | %5d | %9.0f%%  | %9.0f%%" % ("GLOBAL", len(slides), 100*a_c, 100*a_p))

print("\n=== Donde confunde cada uno ===")
for nombre,res,yp in [("CONCH",conch,yp_c),("Pathos",pathos,yp_p)]:
    print(" "+nombre+":")
    for c in CLASES:
        idx=[i for i,s in enumerate(slides) if res[str(s)]["gt"]==c]
        preds=[yp[i] for i in idx]
        from collections import Counter
        cnt=Counter(preds)
        det=", ".join("%s=%d"%(NOMBRES.get(k,k),v) for k,v in cnt.most_common())
        print("   %s -> %s"%(NOMBRES[c], det))

# Figura comparativa: barras agrupadas por tumor
x=np.arange(len(CLASES)); w=0.35
fig,ax=plt.subplots(figsize=(9,5.5))
b1=ax.bar(x-w/2, [100*pc_c[c] for c in CLASES], w, label="CONCH", color="#1f77b4")
b2=ax.bar(x+w/2, [100*pc_p[c] for c in CLASES], w, label="Pathos-Gemma4", color="#ff7f0e")
ax.set_ylabel("Recall (% aciertos)")
ax.set_title("Transferencia humano->veterinaria: CONCH vs Pathos-Gemma4\n30 slides caninos MIDOG++ (zero-shot)")
ax.set_xticks(x); ax.set_xticklabels([NOMBRES[c] for c in CLASES])
ax.legend(); ax.set_ylim(0,110)
for b in list(b1)+list(b2):
    h=b.get_height()
    ax.annotate("%.0f%%"%h, (b.get_x()+b.get_width()/2, h), ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.axhline(33.3, color="gray", ls=":", alpha=0.5, label="azar (3 clases)")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig_path=OUT+"/comparativa_conch_pathos.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print("\nFigura guardada:", fig_path)

# CSV comparativo
import csv
with open(OUT+"/comparativa_30slides.csv","w") as f:
    w=csv.writer(f); w.writerow(["slide","real","CONCH_pred","CONCH_ok","Pathos_pred","Pathos_ok"])
    for s in slides:
        rc=conch[str(s)]["pred"]; rp=pathos[str(s)]["pred"]; g=pathos[str(s)]["gt"]
        w.writerow([s,g,rc,int(rc==g),rp,int(rp==g)])
print("CSV comparativo:", OUT+"/comparativa_30slides.csv")
