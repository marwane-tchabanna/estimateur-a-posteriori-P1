##!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
 Estimateur d'erreur a posteriori par residus -- elements finis P1
 Probleme de Poisson, Dirichlet homogene, carre unite
 Version Python autonome (numpy/scipy) : validation croisee de
 estimateur_residu_P1.edp
=============================================================================

 Probleme continu
 ----------------
      -Delta u = f   dans Omega = ]0,1[^2
             u = 0   sur dOmega

 Solution manufacturee :
      u(x,y) = x(x-1) y(y-1)        (nulle sur les 4 cotes : Dirichlet homogene)
      => Delta u = 2 y(y-1) + 2 x(x-1)
      => f(x,y)  = 2[ x(1-x) + y(1-y) ]

 Estimateur (cours) : pour chaque triangle K de la triangulation T_h,
      eta_1K = h_K || f + Delta u_h ||_{0,K}          (residu interieur)
      eta_2K = ( 1/2 * sum_{E in E_K^i} h_E || [[du_h/dn]] ||^2_{0,E} )^{1/2}
      eta_3K = 0                                      (Dirichlet homogene)
      eta_K^2 = eta_1K^2 + eta_2K^2 ,   eta^2 = sum_K eta_K^2

 En P1 : u_h est affine par morceaux donc Delta u_h = 0 sur chaque K,
 et eta_1K se reduit a h_K ||f||_{0,K}.

 Oscillation des donnees :
      f_K   = (1/|K|) int_K f          (moyenne de f sur K)
      osc_K = h_K || f - f_K ||_{0,K}  -> osc = O(h^2), d'ordre superieur

 Indice d'efficacite :
      I_eff = eta / |u - u_h|_{1,Omega}
 Fiabilite  : |u - u_h|_1 <= C_rel * eta                -> I_eff >= 1/C_rel
 Efficacite : eta <= C_eff * (|u - u_h|_1 + osc)        -> I_eff <= C_eff
 On attend I_eff ~ constante (independante de h).

 Utilisation :  python3 estimateur_residu_P1.py

 Auteur : Marwane Tchabanna -- M2 MANU, Universite de Montpellier
=============================================================================
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# -----------------------------------------------------------------------------
# 1. Donnees du probleme
# -----------------------------------------------------------------------------

def u_exact(x, y):
    """Solution exacte u(x,y) = x(x-1)y(y-1)."""
    return x * (x - 1.0) * y * (y - 1.0)


def grad_u_exact(x, y):
    """Gradient exact : (du/dx, du/dy)."""
    dux = (2.0 * x - 1.0) * y * (y - 1.0)
    duy = x * (x - 1.0) * (2.0 * y - 1.0)
    return dux, duy


def f_source(x, y):
    """Second membre f = -Delta u = 2[x(1-x) + y(1-y)]."""
    return 2.0 * (x * (1.0 - x) + y * (1.0 - y))


# -----------------------------------------------------------------------------
# 2. Maillage structure du carre unite
# -----------------------------------------------------------------------------

def maillage_carre(N):
    """
    Maillage uniforme N x N du carre unite, chaque petit carre coupe
    en 2 triangles par la meme diagonale (bas-gauche -> haut-droite),
    comme square(N, N) de FreeFem++.

    Retourne
    --------
    coords    : (nv,2)  coordonnees des sommets
    tris      : (nt,3)  connectivite (indices des 3 sommets, sens direct)
    bord      : (nb,)   indices des sommets du bord (Dirichlet)
    """
    xs = np.linspace(0.0, 1.0, N + 1)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    coords = np.column_stack([X.ravel(), Y.ravel()])

    def idx(i, j):
        return i * (N + 1) + j

    tris = []
    for i in range(N):
        for j in range(N):
            v00, v10 = idx(i, j), idx(i + 1, j)
            v01, v11 = idx(i, j + 1), idx(i + 1, j + 1)
            tris.append([v00, v10, v11])   # triangle bas-droite
            tris.append([v00, v11, v01])   # triangle haut-gauche
    tris = np.array(tris, dtype=np.int64)

    tol = 1e-12
    onbord = (np.abs(coords[:, 0]) < tol) | (np.abs(coords[:, 0] - 1.0) < tol) \
           | (np.abs(coords[:, 1]) < tol) | (np.abs(coords[:, 1] - 1.0) < tol)
    bord = np.where(onbord)[0]
    return coords, tris, bord


# -----------------------------------------------------------------------------
# 3. Quadrature sur le triangle de reference (transformation de Duffy)
# -----------------------------------------------------------------------------
# ATTENTION : |u-u_h|_1 fait intervenir des polynomes de degre 6, ||f||^2 de
# degre 4. Une quadrature au barycentre (degre 1) fausserait l'indice
# d'efficacite. On utilise une regle exacte a haut degre.

def quadrature_triangle(n=6):
    """
    Regle de quadrature sur T_ref = {(a,b) : a,b>=0, a+b<=1}, via Duffy :
        int_T g = int_0^1 int_0^1 g(a=s, b=t(1-s)) (1-s) ds dt
    avec Gauss-Legendre tensorise (n=6 : exacte bien au-dela du degre 6).
    Retourne les coordonnees barycentriques (l0,l1,l2) et les poids
    normalises (somme = 1 ; on multiplie ensuite par l'aire reelle).
    """
    gs, gw = np.polynomial.legendre.leggauss(n)
    gs = 0.5 * (gs + 1.0)          # points sur [0,1]
    gw = 0.5 * gw                  # poids sur [0,1]

    pts, wts = [], []
    for s, ws in zip(gs, gw):
        for t, wt in zip(gs, gw):
            pts.append([s, t * (1.0 - s)])
            wts.append(ws * wt * (1.0 - s))      # jacobien de Duffy
    pts = np.array(pts)
    wts = np.array(wts) * 2.0       # normalisation : sum(wts) = 1 (|T_ref|=1/2)
    l0 = 1.0 - pts[:, 0] - pts[:, 1]
    lam = np.column_stack([l0, pts[:, 0], pts[:, 1]])
    return lam, wts


def points_quadrature(coords, tris, l):
    """Coordonnees physiques du point de quadrature l dans chaque triangle."""
    p0, p1, p2 = coords[tris[:, 0]], coords[tris[:, 1]], coords[tris[:, 2]]
    xq = l[0] * p0[:, 0] + l[1] * p1[:, 0] + l[2] * p2[:, 0]
    yq = l[0] * p0[:, 1] + l[1] * p1[:, 1] + l[2] * p2[:, 1]
    return xq, yq


# -----------------------------------------------------------------------------
# 4. Assemblage P1 et resolution
# -----------------------------------------------------------------------------

def geometrie_triangles(coords, tris):
    """Aires, gradients des fonctions de base P1, diametres h_K."""
    p0 = coords[tris[:, 0]]
    p1 = coords[tris[:, 1]]
    p2 = coords[tris[:, 2]]

    detJ = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) \
         - (p2[:, 0] - p0[:, 0]) * (p1[:, 1] - p0[:, 1])
    aires = 0.5 * np.abs(detJ)

    # grad(lambda_i) = rotation de l'arete opposee / (2*aire)
    gx = np.empty((len(tris), 3))
    gy = np.empty((len(tris), 3))
    for i, (_, b, c) in enumerate([(0, 1, 2), (1, 2, 0), (2, 0, 1)]):
        pb, pc = coords[tris[:, b]], coords[tris[:, c]]
        gx[:, i] = (pb[:, 1] - pc[:, 1]) / detJ
        gy[:, i] = (pc[:, 0] - pb[:, 0]) / detJ

    # diametre = plus grande arete (comme hTriangle de FreeFem++)
    e0 = np.linalg.norm(p2 - p1, axis=1)
    e1 = np.linalg.norm(p0 - p2, axis=1)
    e2 = np.linalg.norm(p1 - p0, axis=1)
    hK = np.maximum(np.maximum(e0, e1), e2)
    return aires, gx, gy, hK


def assemble_et_resoudre(coords, tris, bord, lam, wts):
    """Assemble A u = b pour -Delta u = f, P1, Dirichlet homogene."""
    nv = len(coords)
    aires, gx, gy, hK = geometrie_triangles(coords, tris)

    # --- matrice de rigidite : A_ij^K = |K| * grad(phi_i).grad(phi_j)
    rows = np.repeat(tris, 3, axis=1).ravel()
    cols = np.tile(tris, (1, 3)).ravel()
    Kloc = (gx[:, :, None] * gx[:, None, :] +
            gy[:, :, None] * gy[:, None, :]) * aires[:, None, None]
    A = sp.coo_matrix((Kloc.ravel(), (rows, cols)), shape=(nv, nv)).tocsr()

    # --- second membre : b_i^K = int_K f phi_i
    b = np.zeros(nv)
    for q in range(len(wts)):
        l = lam[q]
        xq, yq = points_quadrature(coords, tris, l)
        fq = f_source(xq, yq) * aires * wts[q]
        for i in range(3):
            np.add.at(b, tris[:, i], fq * l[i])

    # --- Dirichlet homogene : elimination des ddl du bord
    libres = np.setdiff1d(np.arange(nv), bord)
    uh = np.zeros(nv)
    uh[libres] = spla.spsolve(A[libres][:, libres].tocsc(), b[libres])
    return uh, aires, gx, gy, hK


# -----------------------------------------------------------------------------
# 5. Erreurs exactes : semi-norme H1 (par triangle) et norme L2
# -----------------------------------------------------------------------------

def erreurs(coords, tris, uh, aires, gx, gy, lam, wts):
    """|u - u_h|_{1,K} pour chaque K, ||u - u_h||_0 global, grad u_h."""
    uK = uh[tris]
    ghx = np.sum(uK * gx, axis=1)       # grad u_h constant par triangle
    ghy = np.sum(uK * gy, axis=1)

    err2 = np.zeros(len(tris))
    errL2 = np.zeros(len(tris))
    for q in range(len(wts)):
        l = lam[q]
        xq, yq = points_quadrature(coords, tris, l)
        ex, ey = grad_u_exact(xq, yq)
        err2 += wts[q] * ((ex - ghx) ** 2 + (ey - ghy) ** 2)
        uhq = l[0] * uK[:, 0] + l[1] * uK[:, 1] + l[2] * uK[:, 2]
        errL2 += wts[q] * (uhq - u_exact(xq, yq)) ** 2
    errK = np.sqrt(err2 * aires)
    eL2 = np.sqrt(np.sum(errL2 * aires))
    return errK, eL2, (ghx, ghy)


# -----------------------------------------------------------------------------
# 6. Estimateur par residus + moyenne f_K + oscillation
# -----------------------------------------------------------------------------

def estimateur(coords, tris, aires, hK, grad_uh, lam, wts):
    """
    Calcule eta_1K (residu interieur), eta_2K (sauts de flux), eta_K,
    la moyenne f_K et l'oscillation osc_K.
    En P1, Delta u_h = 0 => eta_1K = h_K ||f||_{0,K}.
    """
    nt = len(tris)

    # --- (a) residu interieur ||f||^2_{0,K} et moyenne f_K = int_K f / |K|
    normf2 = np.zeros(nt)
    fK = np.zeros(nt)
    for q in range(len(wts)):
        xq, yq = points_quadrature(coords, tris, lam[q])
        fq = f_source(xq, yq)
        normf2 += wts[q] * fq ** 2
        fK += wts[q] * fq                 # poids normalises -> moyenne
    normf2 *= aires
    eta1 = hK * np.sqrt(normf2)

    # --- (b) oscillation : osc_K = h_K ||f - f_K||_{0,K}
    osc2 = np.zeros(nt)
    for q in range(len(wts)):
        xq, yq = points_quadrature(coords, tris, lam[q])
        osc2 += wts[q] * (f_source(xq, yq) - fK) ** 2
    osc = hK * np.sqrt(osc2 * aires)

    # --- (c) sauts de flux sur les aretes interieures (vectorise)
    ghx, ghy = grad_uh
    aretes = np.vstack([tris[:, [1, 2]], tris[:, [2, 0]], tris[:, [0, 1]]])
    tri_id = np.tile(np.arange(nt), 3)
    cle = np.sort(aretes, axis=1)
    cle_vue, inv, cnt = np.unique(cle, axis=0, return_inverse=True,
                                  return_counts=True)
    inv = inv.ravel()                        # securite numpy >= 2.0

    # les 2 occurrences d'une arete interieure sont consecutives apres tri
    ordre_tri = np.argsort(inv, kind="stable")
    debut = np.concatenate(([0], np.cumsum(cnt)[:-1]))
    inter = np.where(cnt == 2)[0]            # aretes de bord exclues (eta_3K = 0)
    K1 = tri_id[ordre_tri[debut[inter]]]
    K2 = tri_id[ordre_tri[debut[inter] + 1]]

    a, bb = cle_vue[inter, 0], cle_vue[inter, 1]
    vec = coords[bb] - coords[a]
    hE = np.linalg.norm(vec, axis=1)
    nx, ny = vec[:, 1] / hE, -vec[:, 0] / hE          # normale unitaire a E

    # [[du_h/dn]] constant sur E (P1) ; le signe de n est sans effet (carre)
    saut = (ghx[K1] - ghx[K2]) * nx + (ghy[K1] - ghy[K2]) * ny
    norm_saut2 = hE * saut ** 2                        # ||[[.]]||^2_{0,E}
    contrib = 0.5 * hE * norm_saut2                    # 1/2 h_E ||[[.]]||^2

    eta2_sq = (np.bincount(K1, weights=contrib, minlength=nt)
             + np.bincount(K2, weights=contrib, minlength=nt))

    eta2 = np.sqrt(eta2_sq)
    etaK = np.sqrt(eta1 ** 2 + eta2 ** 2)
    return etaK, eta1, eta2, osc, fK


# -----------------------------------------------------------------------------
# 7. Etude de convergence : eta vs erreur vraie, indice d'efficacite
# -----------------------------------------------------------------------------

def ordre(y):
    """Ordres observes log2(y_{k-1}/y_k) (raffinement h -> h/2)."""
    y = np.asarray(y)
    return np.log(y[:-1] / y[1:]) / np.log(2.0)


def etude(Ns=(4, 8, 16, 32, 64, 128), nquad=6, verbose=True):
    lam, wts = quadrature_triangle(nquad)
    res = []
    for N in Ns:
        coords, tris, bord = maillage_carre(N)
        uh, aires, gx, gy, hK = assemble_et_resoudre(coords, tris, bord, lam, wts)
        errK, eL2, grad_uh = erreurs(coords, tris, uh, aires, gx, gy, lam, wts)
        etaK, eta1, eta2, osc, fK = estimateur(coords, tris, aires, hK,
                                               grad_uh, lam, wts)
        err = np.sqrt(np.sum(errK ** 2))
        eta = np.sqrt(np.sum(etaK ** 2))
        ratio = etaK / np.maximum(errK, 1e-30)      # efficacite locale
        res.append(dict(N=N, h=1.0 / N, ndof=len(coords), nt=len(tris),
                        err=err, errL2=eL2, eta=eta,
                        eta1=np.sqrt(np.sum(eta1 ** 2)),
                        eta2=np.sqrt(np.sum(eta2 ** 2)),
                        osc=np.sqrt(np.sum(osc ** 2)),
                        Ieff=eta / err,
                        eff_min=ratio.min(), eff_max=ratio.max(),
                        fK_min=fK.min(), fK_max=fK.max()))

    if verbose:
        print("\n  Poisson, Dirichlet homogene, carre unite,  u = x(x-1)y(y-1)")
        print("  Elements finis P1 -- estimateur par residus (version Python)\n")
        print(f"  {'N':>4} {'h':>8} {'ndof':>7} {'|u-u_h|_1':>12} "
              f"{'eta':>12} {'eta_1':>11} {'eta_2':>11} {'osc':>10} {'I_eff':>8}")
        print("  " + "-" * 92)
        for r in res:
            print(f"  {r['N']:>4} {r['h']:>8.4f} {r['ndof']:>7} {r['err']:>12.6e} "
                  f"{r['eta']:>12.6e} {r['eta1']:>11.4e} {r['eta2']:>11.4e} "
                  f"{r['osc']:>10.3e} {r['Ieff']:>8.4f}")

        err_t  = [r["err"]   for r in res]
        eta_t  = [r["eta"]   for r in res]
        osc_t  = [r["osc"]   for r in res]
        L2_t   = [r["errL2"] for r in res]
        Ieff_t = np.array([r["Ieff"] for r in res])

        print("\n  Ordres de convergence observes :")
        for k, (oe, on, oo, ol) in enumerate(zip(ordre(err_t), ordre(eta_t),
                                                 ordre(osc_t), ordre(L2_t)),
                                             start=1):
            print(f"    h = {res[k]['h']:.5f}   ordre(|u-u_h|_1) = {oe:6.4f}"
                  f"   ordre(eta) = {on:6.4f}   ordre(osc) = {oo:6.4f}"
                  f"   ordre(||u-u_h||_0) = {ol:6.4f}")

        if len(Ieff_t) >= 2:
            # ecarts divises par 2 a chaque raffinement -> Richardson ordre 1
            Ieff_inf = Ieff_t[-1] + (Ieff_t[-1] - Ieff_t[-2])
            print(f"\n  I_eff extrapole (h -> 0) : {Ieff_inf:.4f}")
        print(f"  Efficacite locale (maillage le plus fin) : "
              f"{res[-1]['eff_min']:.4f} <= eta_K/|u-u_h|_(1,K) <= "
              f"{res[-1]['eff_max']:.4f}")
        print(f"  Moyenne f_K : {res[-1]['fK_min']:.5f} <= f_K <= "
              f"{res[-1]['fK_max']:.5f}")

        print("\n  Lecture : I_eff = eta / |u-u_h|_1 tend vers une CONSTANTE")
        print("  independante de h -> l'estimateur est fiable ET efficace ;")
        print("  osc = O(h^2) est d'ordre superieur a eta = O(h).\n")
    return res


if __name__ == "__main__":
    etude()