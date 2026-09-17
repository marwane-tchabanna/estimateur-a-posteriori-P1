#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
 Tracés Python des résultats produits par FreeFem++
=============================================================================
 Ce script ne fait AUCUN calcul élément fini : il relit simplement les deux
 fichiers écrits par estimateur_residu_P1.edp et trace tout sous matplotlib.
 Intérêt : plus aucune dépendance à ffglut / serveur graphique.

   resultats.txt      : N  h  |u-u_h|_1  eta  eta_1  eta_2  I_eff
   maillage_etaK.txt  : nv nt
                        nv lignes : x y
                        nt lignes : i0 i1 i2 eta_K |u-u_h|_{1,K}

 Utilisation :
     FreeFem++ -nw estimateur_residu_P1.edp     # produit les .txt
     python3 plot_resultats.py                  # produit figure_estimateur.png
=============================================================================
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")          # backend sans fenêtre : indispensable en SSH/WSL
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

# -----------------------------------------------------------------------------
# 1. Lecture des tableaux de convergence
# -----------------------------------------------------------------------------
tab = np.loadtxt("resultats.txt", skiprows=1)
N, h, err, eta, eta1, eta2, Ieff = tab.T

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
    return coords, tris, etaK, errK

coords, tris, etaK, errK = lire_maillage()
triang = mtri.Triangulation(coords[:, 0], coords[:, 1], tris)

# -----------------------------------------------------------------------------
# 3. Figure : 4 panneaux
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(1, 4, figsize=(20, 4.4))

# --- (a) convergence en log-log
ax[0].loglog(h, err, "o-", label=r"$|u-u_h|_{1,\Omega}$")
ax[0].loglog(h, eta, "s-", label=r"$\eta$")
ax[0].loglog(h, 0.25 * h, "k--", lw=1, label="pente 1")
ax[0].set_xlabel("h")
ax[0].set_title("Convergence : erreur et estimateur")
ax[0].legend()
ax[0].grid(True, which="both", alpha=0.3)

# --- (b) indice d'efficacité
ax[1].semilogx(h, Ieff, "o-", color="crimson")
ax[1].axhline(4 * np.sqrt(2), ls="--", c="k", lw=1,
              label=r"$4\sqrt{2}\simeq 5.657$")
ax[1].set_xlabel("h")
ax[1].set_ylim(0, 7)
ax[1].set_title(r"$I_{eff}=\eta/|u-u_h|_1$")
ax[1].legend()
ax[1].grid(alpha=0.3)

# --- (c) répartition spatiale de eta_K (tripcolor = valeur constante par
#         triangle, c'est exactement la nature P0 de l'indicateur)
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

plt.tight_layout()
plt.savefig("figure_estimateur.png", dpi=140)
print("figure_estimateur.png écrite.")
print(f"I_eff final = {Ieff[-1]:.4f}   (4*sqrt(2) = {4*np.sqrt(2):.4f})")
print(f"efficacité locale : {ratio.min():.3f} ... {ratio.max():.3f}")
