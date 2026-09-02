#!/usr/bin/env python3
"""t12_concept_hessian.py — direction 5, scoped to the ONE cell the literature leaves open.

The 2026-08-10 review found second-order interpretability largely EXECUTED already:
Hessian-bilinear interaction in activation patching (2606.27510, on GPT-2/Pythia/Qwen), HVP
attribution correction (2606.09899, 124M-9B), cross-token Hessian blocks defined by HETA
(2604.13258) but aggregated into token scores. What it lists as open is the RESTRICTED
CONCEPT HESSIAN: whether writing concept A changes the effect of writing concept B.

We measure it WITHOUT autograd, by finite differences on the interaction residual:

    R_ij(a,b) = F(h + a v_i + b v_j) - F(h + a v_i) - F(h + b v_j) + F(h)

which is 0 for any purely additive (locally linear) response and equals a*b*v_i^T H_F v_j to
second order. Four forward passes per pair. F is a scalar observable — a target-token logit.

STATED LIMITATION (the review's, and it is not fixable by better estimation): V^T H_F V is
OBSERVABLE-RELATIVE. It answers "do these directions interact in THIS logit", not "are these
concepts entangled in the representation". A Hessian lens is therefore not
representation-intrinsic the way the J-lens claims to be. This is why direction 5 stays out of
the paper's spine and is reported as a bounded diagnostic.

Controls: matched-norm random direction pairs (interaction should vanish), and a scale sweep in
(a,b) to locate where the additive approximation breaks -- the trust region.
"""
import argparse, itertools, json, os, sys, time
import torch
torch.set_num_threads(min(8, os.cpu_count() or 8))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "jacobian-lens"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from jlens.hf import Layout, from_hf                # noqa: E402
from jlens.hooks import ActivationRecorder          # noqa: E402
from jlens.lens import JacobianLens                 # noqa: E402
from joperator import token_direction               # noqa: E402
L5 = Layout("gpt_neox", layers="layers", norm="final_layer_norm", embed="embed_in", lm_head="lm_head")

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="EleutherAI/pythia-410m-deduped")
ap.add_argument("--lens", required=True)
ap.add_argument("--layer", type=int, default=None)
ap.add_argument("--concepts", default="France,China,Japan,Germany,water,music")
ap.add_argument("--observable", default=" Paris")
ap.add_argument("--prompt", default="The capital of France is the city of")
ap.add_argument("--amps", default="0.25,0.5,1,2,4")
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--out", required=True)
a = ap.parse_args()
t0 = time.perf_counter()
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
tok = AutoTokenizer.from_pretrained(a.model)
hf = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32)
try: model = from_hf(hf, tok)
except ValueError: model = from_hf(hf, tok, layout=L5)
lens = JacobianLens.load(a.lens)
layer = a.layer if a.layer is not None else lens.source_layers[len(lens.source_layers)*2//3]
WU = model._lm_head.weight.detach().float()
blocks = model.model.layers if hasattr(model, "model") else model.layers
obs_id = tok.encode(a.observable, add_special_tokens=False)[0]
J = lens.jacobians[layer].float()

names = a.concepts.split(",")
V = {}
for c in names:
    e = tok.encode(" " + c, add_special_tokens=False)
    if len(e) != 1:
        continue
    v = token_direction(J, WU[e[0]])
    V[c] = v / v.norm()                       # unit directions so amplitudes are comparable
names = list(V)
g = torch.Generator().manual_seed(a.seed)
R = {f"rand{i}": torch.nn.functional.normalize(
        torch.randn(J.shape[0], generator=g), dim=0) for i in range(len(names))}

def F(edits):
    """Scalar observable: logit of `observable` at the final position, under residual edits."""
    handles = []
    def mk(delta):
        def hook(mod, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            y = (t.float() + delta.to(t.device)).to(t.dtype)
            return (y,) + out[1:] if isinstance(out, tuple) else y
        return hook
    tot = torch.zeros(J.shape[0])
    for v, amp in edits:
        tot = tot + amp * v
    try:
        handles.append(blocks[layer].register_forward_hook(mk(tot)))
        ids = model.encode(a.prompt, max_length=64)
        with torch.no_grad():
            out = model.forward(ids)
            hs = out if torch.is_tensor(out) else out[0]
            return float(model._lm_head(hs[0, -1].float()).float()[obs_id])  # single norm
    finally:
        for h in handles: h.remove()

base = F([])
amps = [float(x) for x in a.amps.split(",")]
rows = []
for amp in amps:
    for label, pool in (("concept", V), ("random", R)):
        keys = list(pool)
        for i, j in itertools.combinations(range(len(keys)), 2):
            vi, vj = pool[keys[i]], pool[keys[j]]
            fi = F([(vi, amp)]); fj = F([(vj, amp)]); fij = F([(vi, amp), (vj, amp)])
            inter = fij - fi - fj + base
            lin = (fi - base) + (fj - base)
            rows.append({"amp": amp, "pool": label, "a": keys[i], "b": keys[j],
                         "interaction": round(inter, 5),
                         "additive_prediction": round(lin, 5),
                         "rel_interaction": round(abs(inter) / (abs(lin) + 1e-6), 5)})
    cs = [r["rel_interaction"] for r in rows if r["amp"] == amp and r["pool"] == "concept"]
    rs = [r["rel_interaction"] for r in rows if r["amp"] == amp and r["pool"] == "random"]
    med = lambda v: float(torch.tensor(v).median())
    print(f"  amp={amp:5.2f}  concept rel-interaction median {med(cs):.4f}   "
          f"random {med(rs):.4f}   ratio {med(cs)/max(med(rs),1e-9):.2f}", flush=True)

out = {"experiment": "T12 restricted concept Hessian via finite-difference interaction residual",
       "status": "EXPLORATORY / DIRECTIONAL — no pre-registration; bounded diagnostic, not a lens",
       "limitation": ("V^T H_F V is OBSERVABLE-RELATIVE: it measures interaction in the chosen "
                      "logit, not entanglement in the representation. Direction 5 is therefore "
                      "kept out of the paper's spine."),
       "model": a.model, "lens": a.lens, "layer": layer, "observable": a.observable,
       "prompt": a.prompt, "concepts": names, "amps": amps, "baseline_logit": round(base, 5),
       "centered": False,
       "env": {"transformers": transformers.__version__, "torch": torch.__version__},
       "jlens_commit": "581d398613e5602a5af361e1c34d3a92ea82ba8e",
       "rows": rows, "runtime_s": round(time.perf_counter() - t0, 1)}
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump(out, open(a.out, "w"), indent=1)
print(f"wrote {a.out}  ({out['runtime_s']}s)")
