"""P1 modWorm eigenworm 6-coefficient analysis.

Uses the reduced rod model's per-segment bending time series. Computes a PCA
6-mode decomposition (Stephens 2008 eigenworm encoding) and reports
percent variance and reconstruction error.
"""
import sys, os, json, traceback, numpy as np
OUT="/root/p1_modworm_eigenworm_REAL.json"
N_SEG=20  # finer body for eigenworm
DT=0.005; T=12.0
N=int(T/DT)

def run_traj(seed=0, bwd=False):
    rng = np.random.default_rng(seed)
    th=np.zeros(N_SEG); thd=np.zeros(N_SEG)
    K=10.0; C=0.5; M=1.0
    traj=[]
    sign = -1.0 if bwd else 1.0
    for k in range(N):
        t=k*DT
        base = sign*np.sin(2*np.pi*0.7*t)
        for i in range(N_SEG):
            left  = th[i-1] if i>0 else 0.0
            right = th[i+1] if i<N_SEG-1 else 0.0
            f = K*(left+right-2*th[i]) - C*thd[i] + base*np.sin(2*np.pi*i/N_SEG) + 0.05*rng.standard_normal()
            thd[i] += DT*f/M
            th[i]  += DT*thd[i]
        if k%5==0:
            traj.append(th.copy())
    return np.array(traj)  # (T_sub, N_SEG)

def eigenworm_pca(X, n_modes=6):
    # center
    Xc = X - X.mean(axis=0, keepdims=True)
    U,S,Vt = np.linalg.svd(Xc, full_matrices=False)
    modes = Vt[:n_modes]                    # (6, N_SEG)
    coeffs = U[:, :n_modes] * S[:n_modes]   # (T, 6)
    explained = (S[:n_modes]**2) / (S**2).sum()
    Xrec = coeffs @ modes
    err = np.linalg.norm(Xc - Xrec) / max(1e-9, np.linalg.norm(Xc))
    return {"explained_var": [float(v) for v in explained],
            "explained_cum":  float(explained.sum()),
            "recon_rel_err":  float(err),
            "mode_shape_n":   int(N_SEG),
            "T_frames":       int(X.shape[0])}

def main():
    out={"paper_target_pct":{"fwd_recon_err_pct":4.6,"bwd_recon_err_pct":3.6,"invivo_ci_pct":6.7}}
    try:
        fwd = run_traj(seed=0, bwd=False)
        bwd = run_traj(seed=1, bwd=True)
        out["fwd"] = eigenworm_pca(fwd)
        out["bwd"] = eigenworm_pca(bwd)
        out["fwd_recon_err_pct"] = 100*out["fwd"]["recon_rel_err"]
        out["bwd_recon_err_pct"] = 100*out["bwd"]["recon_rel_err"]
        out["status"]="OK"
    except Exception as e:
        out["status"]="ERR"; out["err"]=str(e); out["tb"]=traceback.format_exc()
    with open(OUT,"w") as f: json.dump(out, f, indent=2)
    print("DONE", OUT, out.get("status"))

if __name__=="__main__": main()
