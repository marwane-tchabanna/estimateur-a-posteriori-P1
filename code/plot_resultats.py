#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
 Tracés Python des résultats produits par FreeFem++
=============================================================================
 Ce script ne fait AUCUN calcul élément fini : il relit les deux fichiers
 écrits par estimateur_residu.edp et trace tout sous matplotlib.

   resultats.txt      : N  h  |u-u_h|_1  eta  eta_1  eta_2  osc  I_eff
   maillage_etaK.txt  : nv nt
                        nv lignes : x y
                        nt lignes : i0 i1 i2 eta_K |u-u_h|_{1,K} f_K osc_K

 Utilisation :
     FreeFem++ -nw estimateur_residu.edp     # produit les .txt
     python3 plot_resultats.py               # produit figure_estimateur.png
     python3 plot_resultats.py --show        # idem + ouvre une fenêtre
=============================================================================
"""

import sys
import numpy as np
import matplotlib

AFFICHER = "--show" in sys.argv
if not AFFICHER:
    matplotlib.use("Agg")      # backend sans fenêtre (SSH/WSL)
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

# -----------------------------------------------------------------------------
# 1. Lecture des tableaux de convergence
# -----------------------------------------------------------------------------
tab = np.loadtxt("resultats.txt", skiprows=1)
N, h, err, eta, eta1, eta2, osc, Ieff = tab.T


# -----------------------------------------------------------------------------
# 2. Lecture du maillage et des indicateurs locaux
# -----------------------------------------------------------------------------
def lire_maillage(nom="maillage_etaK.txt"):
    with open(nom) as fh:
        nv, nt = map(int, fh.readline().split())
        coords = np.array([list(map(float, fh.readline().split()))
                           for _ in range(nv)])
        data = np.array([list(map(float, fh.readline().split()))
                         for _ in range(nt)])
    tris = data[:, :3].astype(int)
    etaK = data[:, 3]
    errK = data[:, 4]
    fK   = data[:, 5]
    oscK = data[:, 6]
    return coords, tris, etaK, errK, fK, oscK


coords, tris, etaK, errK, fK, oscK = lire_maillage()
triang = mtri.Triangulation(coords[:, 0], coords[:, 1], tris)


def ordre(y):
    """Ordres observés log2(y_{k-1}/y_k) (raffinement h -> h/2)."""
    return np.log(y[:-1] / y[1:]) / np.log(2.0)


# -----------------------------------------------------------------------------
# 3. Figure : 6 panneaux (2 x 3)
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(2, 3, figsize=(17, 10))
ax = ax.ravel()

# --- (a) convergence en log-log, pentes calées sur le premier point
ax[0].loglog(h, err,  "o-", label=r"$|u-u_h|_{1,\Omega}$")
ax[0].loglog(h, eta,  "s-", label=r"$\eta$")
ax[0].loglog(h, eta1, "^--", ms=4, label=r"$\eta_1$")
ax[0].loglog(h, eta2, "v--", ms=4, label=r"$\eta_2$")
ax[0].loglog(h, osc,  "d-", label=r"osc")
ax[0].loglog(h, err[0] * h / h[0],        "k--", lw=1, label="pente 1")
ax[0].loglog(h, osc[0] * (h / h[0])**2,   "k:",  lw=1, label="pente 2")
ax[0].set_xlabel("h")
ax[0].set_title("Convergence : erreur, estimateur, oscillation")
ax[0].legend(fontsize=8)
ax[0].grid(True, which="both", alpha=0.3)

# --- (b) indice d'efficacité global + borne
Ieff_inf = Ieff[-1] + (Ieff[-1] - Ieff[-2])      # extrapolation de Richardson (ordre 1)

ax[1].semilogx(h, Ieff, "o-", color="crimson", label=r"$I_{eff}(h)$")
ax[1].axhline(Ieff_inf, ls="--", c="k", lw=1.2,
              label=rf"limite extrapolée $\approx {Ieff_inf:.3f}$")
ax[1].axhspan(Ieff.min(), Ieff.max(), color="crimson", alpha=0.08,
              label=rf"$[{Ieff.min():.2f},\ {Ieff.max():.2f}]$")
ax[1].set_xlabel("h")
ax[1].set_title(r"$I_{eff}=\eta/|u-u_h|_1$ : borné indépendamment de $h$", pad=14)
ax[1].set_ylim(0, 1.15 * max(Ieff.max(), Ieff_inf))
ax[1].grid(True, which="both", alpha=0.3)
ax[1].legend(fontsize=8, loc="lower right")
for hi, Ii in zip(h, Ieff):
    ax[1].annotate(f"{Ii:.3f}", (hi, Ii), textcoords="offset points",
                   xytext=(0, -14), ha="center", fontsize=8)
# --- (c) eta_K (P0 : tripcolor avec facecolors = une valeur par triangle)
tp = ax[2].tripcolor(triang, facecolors=etaK, cmap="viridis",
                     edgecolors="none")
fig.colorbar(tp, ax=ax[2])
ax[2].set_aspect("equal")
ax[2].set_title(r"$\eta_K$ sur le maillage le plus fin")

# --- (d) efficacité locale eta_K / |u-u_h|_{1,K}
ratio = etaK / np.maximum(errK, 1e-30)
tp2 = ax[3].tripcolor(triang, facecolors=ratio, cmap="magma",
                      edgecolors="none")
fig.colorbar(tp2, ax=ax[3])
ax[3].set_aspect("equal")
ax[3].set_title(r"efficacité locale $\eta_K/|u-u_h|_{1,K}$"
                + f"\n[{ratio.min():.2f}, {ratio.max():.2f}]")

# --- (e) moyenne f_K
tp3 = ax[4].tripcolor(triang, facecolors=fK, cmap="coolwarm",
                      edgecolors="none")
fig.colorbar(tp3, ax=ax[4])
ax[4].set_aspect("equal")
ax[4].set_title(r"$f_K = \frac{1}{|K|}\int_K f$")

# --- (f) oscillation locale osc_K
tp4 = ax[5].tripcolor(triang, facecolors=oscK, cmap="cividis",
                      edgecolors="none")
fig.colorbar(tp4, ax=ax[5])
ax[5].set_aspect("equal")
ax[5].set_title(r"$\mathrm{osc}_K = h_K\,\|f-f_K\|_{0,K}$")

plt.tight_layout()
plt.savefig("figure_estimateur.png", dpi=140)

# -----------------------------------------------------------------------------
# 4. Résumé console
# -----------------------------------------------------------------------------
print("figure_estimateur.png écrite.")
print(f"I_eff : {' '.join(f'{v:.4f}' for v in Ieff)}")
print(f"I_eff extrapolé (h -> 0) : {Ieff_inf:.4f}")      # <-- ici
print(f"ordres |u-u_h|_1 : {np.round(ordre(err), 3)}")
print(f"ordres eta       : {np.round(ordre(eta), 3)}")
print(f"ordres osc       : {np.round(ordre(osc), 3)}")
print(f"efficacité locale : {ratio.min():.3f} ... {ratio.max():.3f}")

if AFFICHER:
    plt.show()
