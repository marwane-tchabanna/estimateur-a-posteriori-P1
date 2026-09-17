#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
 Estimateur d'erreur a posteriori par residus -- elements finis P1
 Probleme de Poisson, Dirichlet homogene, carre unite
=============================================================================

 Probleme continu
 ----------------
      -Delta u = f   dans Omega = ]0,1[^2
             u = 0   sur dOmega

 Solution manufacturee (celle de l'exercice) :
      u(x,y) = x(x-1) y(y-1)        (nulle sur les 4 cotes : Dirichlet homogene)
      => Delta u = 2 y(y-1) + 2 x(x-1)
      => f(x,y)  = 2[ x(1-x) + y(1-y) ]

 Estimateur (cours) : pour chaque triangle K de la triangulation T_h,
      eta_1K = h_K || f + Delta u_h ||_{0,K}          (residu interieur)
      eta_2K = ( 1/2 * sum_{E in E_K^i} h_E || [[du_h/dn]] ||^2_{0,E} )^{1/2}
      eta_3K = 0                                      (Dirichlet homogene :
                                                       les termes de bord
                                                       s'annulent, cf. notes)
      eta_K^2 = eta_1K^2 + eta_2K^2 ,   eta^2 = sum_K eta_K^2

 En P1 : u_h est affine par morceaux donc Delta u_h = 0 sur chaque K,
 et eta_1K se reduit a h_K ||f||_{0,K}.

 Indice d'efficacite :
      I_eff = eta / |u - u_h|_{1,Omega}
 Fiabilite  : |u - u_h|_1 <= C_rel * eta      -> I_eff >= 1/C_rel
 Efficacite : eta <= C_eff * |u - u_h|_1      -> I_eff <= C_eff
 On attend I_eff ~ constante (independante de h) : c'est la validation
 numerique du theoreme du cours.

 Auteur : script pedagogique M2 MANU
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
# 2. Maillage structure du carre unite (decoupage "criss-cross" simple)
# -----------------------------------------------------------------------------

def maillage_carre(N):
    """
    Maillage uniforme N x N du carre unite, chaque petit carre coupe
    en 2 triangles par la diagonale (bas-gauche -> haut-droite).

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
# ATTENTION (piege classique) : l'erreur d'energie |u-u_h|_1 et ||f||_{0,K}
# font intervenir des polynomes de degre eleve (jusqu'a 6 ici). Une quadrature
# au barycentre (degre 1) fausserait completement l'indice d'efficacite.
# On utilise donc une regle exacte a haut degre.

def quadrature_triangle(n=6):
    """
    Regle de quadrature sur T_ref = {(a,b) : a,b>=0, a+b<=1}, via Duffy :
        int_T g = int_0^1 int_0^1 g(a=s, b=t(1-s)) (1-s) ds dt
    avec Gauss-Legendre tensorise (exacte pour des polynomes de degre >= 2n-2).
    Retourne les coordonnees barycentriques (l0,l1,l2) et les poids
    normalises (somme des poids = 1 = |T_ref| / |T_ref|, on multiplie par
    l'aire reelle ensuite).
    """
    gs, gw = np.polynomial.legendre.leggauss(n)
    gs = 0.5 * (gs + 1.0)          # points sur [0,1]
    gw = 0.5 * gw                  # poids sur [0,1]

    pts, wts = [], []
    for s, ws in zip(gs, gw):
        for t, wt in zip(gs, gw):
            a = s
            b = t * (1.0 - s)
            jac = (1.0 - s)         # jacobien de Duffy
            pts.append([a, b])
            wts.append(ws * wt * jac)
    pts = np.array(pts)
    wts = np.array(wts) * 2.0       # normalisation : sum(wts) = 1 ( |T_ref|=1/2 )
    l0 = 1.0 - pts[:, 0] - pts[:, 1]
    lam = np.column_stack([l0, pts[:, 0], pts[:, 1]])
    return lam, wts


# -----------------------------------------------------------------------------
# 4. Assemblage P1 et resolution
# -----------------------------------------------------------------------------

def geometrie_triangles(coords, tris):
    """Aires, gradients des fonctions de base P1, diametres h_K."""
    p0 = coords[tris[:, 0]]
    p1 = coords[tris[:, 1]]
    p2 = coords[tris[:, 2]]

    # aire signee (sens direct attendu)
    detJ = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) \
         - (p2[:, 0] - p0[:, 0]) * (p1[:, 1] - p0[:, 1])
    aires = 0.5 * np.abs(detJ)

    # grad(lambda_i) = rotation de l'arete opposee / (2*aire)
    gx = np.empty((len(tris), 3))
    gy = np.empty((len(tris), 3))
    for i, (a, b, c) in enumerate([(0, 1, 2), (1, 2, 0), (2, 0, 1)]):
        pa, pb, pc = coords[tris[:, a]], coords[tris[:, b]], coords[tris[:, c]]
        gx[:, i] = (pb[:, 1] - pc[:, 1]) / detJ
        gy[:, i] = (pc[:, 0] - pb[:, 0]) / detJ

    # diametre = plus grande arete
    e0 = np.linalg.norm(p2 - p1, axis=1)
    e1 = np.linalg.norm(p0 - p2, axis=1)
    e2 = np.linalg.norm(p1 - p0, axis=1)
    hK = np.maximum(np.maximum(e0, e1), e2)
    return aires, gx, gy, hK


def assemble_et_resoudre(coords, tris, bord, lam, wts):
    """Assemble A u = b pour -Delta u = f, P1, Dirichlet homogene."""
    nv, nt = len(coords), len(tris)
    aires, gx, gy, hK = geometrie_triangles(coords, tris)

    # --- matrice de rigidite : A_ij^K = |K| * grad(phi_i).grad(phi_j)
    rows = np.repeat(tris, 3, axis=1).ravel()
    cols = np.tile(tris, (1, 3)).ravel()
    Kloc = (gx[:, :, None] * gx[:, None, :] +
            gy[:, :, None] * gy[:, None, :]) * aires[:, None, None]
    A = sp.coo_matrix((Kloc.ravel(), (rows, cols)), shape=(nv, nv)).tocsr()

    # --- second membre : b_i^K = int_K f phi_i (quadrature exacte)
    p0, p1, p2 = coords[tris[:, 0]], coords[tris[:, 1]], coords[tris[:, 2]]
    b = np.zeros(nv)
    for q in range(len(wts)):
        l = lam[q]
        xq = l[0] * p0[:, 0] + l[1] * p1[:, 0] + l[2] * p2[:, 0]
        yq = l[0] * p0[:, 1] + l[1] * p1[:, 1] + l[2] * p2[:, 1]
        fq = f_source(xq, yq) * aires * wts[q]
        for i in range(3):
            np.add.at(b, tris[:, i], fq * l[i])

    # --- conditions de Dirichlet homogenes (elimination des ddl du bord)
    libres = np.setdiff1d(np.arange(nv), bord)
    uh = np.zeros(nv)
    uh[libres] = spla.spsolve(A[libres][:, libres].tocsc(), b[libres])
    return uh, aires, gx, gy, hK


# -----------------------------------------------------------------------------
# 5. Erreur exacte en semi-norme H1 (energie)
# -----------------------------------------------------------------------------

def erreur_H1(coords, tris, uh, aires, gx, gy, lam, wts):
    """|u - u_h|_{1,K} pour chaque K, par quadrature de haut degre."""
    p0, p1, p2 = coords[tris[:, 0]], coords[tris[:, 1]], coords[tris[:, 2]]
    # grad u_h constant par triangle
    uK = uh[tris]
    ghx = np.sum(uK * gx, axis=1)
    ghy = np.sum(uK * gy, axis=1)

    err2 = np.zeros(len(tris))
    for q in range(len(wts)):
        l = lam[q]
        xq = l[0] * p0[:, 0] + l[1] * p1[:, 0] + l[2] * p2[:, 0]
        yq = l[0] * p0[:, 1] + l[1] * p1[:, 1] + l[2] * p2[:, 1]
        ex, ey = grad_u_exact(xq, yq)
        err2 += wts[q] * ((ex - ghx) ** 2 + (ey - ghy) ** 2)
    err2 *= aires
    return np.sqrt(err2), (ghx, ghy)


# -----------------------------------------------------------------------------
# 6. Estimateur par residus
# -----------------------------------------------------------------------------

def estimateur(coords, tris, aires, hK, grad_uh, lam, wts):
    """
    Calcule eta_1K (residu interieur), eta_2K (sauts de flux) et eta_K.
    En P1, Delta u_h = 0 => eta_1K = h_K ||f||_{0,K}.
    """
    nt = len(tris)
    p0, p1, p2 = coords[tris[:, 0]], coords[tris[:, 1]], coords[tris[:, 2]]

    # --- (a) residu interieur : ||f||^2_{0,K}
    normf2 = np.zeros(nt)
    for q in range(len(wts)):
        l = lam[q]
        xq = l[0] * p0[:, 0] + l[1] * p1[:, 0] + l[2] * p2[:, 0]
        yq = l[0] * p0[:, 1] + l[1] * p1[:, 1] + l[2] * p2[:, 1]
        normf2 += wts[q] * f_source(xq, yq) ** 2
    normf2 *= aires
    eta1 = hK * np.sqrt(normf2)

    # --- (b) oscillation : h_K ||f - f_K||_{0,K}, f_K = moyenne de f sur K
    fbar = np.zeros(nt)
    for q in range(len(wts)):
        l = lam[q]
        xq = l[0] * p0[:, 0] + l[1] * p1[:, 0] + l[2] * p2[:, 0]
        yq = l[0] * p0[:, 1] + l[1] * p1[:, 1] + l[2] * p2[:, 1]
        fbar += wts[q] * f_source(xq, yq)
    osc2 = np.zeros(nt)
    for q in range(len(wts)):
        l = lam[q]
        xq = l[0] * p0[:, 0] + l[1] * p1[:, 0] + l[2] * p2[:, 0]
        yq = l[0] * p0[:, 1] + l[1] * p1[:, 1] + l[2] * p2[:, 1]
        osc2 += wts[q] * (f_source(xq, yq) - fbar) ** 2
    osc = hK * np.sqrt(osc2 * aires)

    # --- (c) sauts de flux sur les aretes interieures
    # construction du tableau des aretes : chaque arete locale (i,j)
    ghx, ghy = grad_uh
    aretes = np.vstack([tris[:, [1, 2]], tris[:, [2, 0]], tris[:, [0, 1]]])
    tri_id = np.tile(np.arange(nt), 3)
    cle = np.sort(aretes, axis=1)
    cle_vue, inv, cnt = np.unique(cle, axis=0, return_inverse=True,
                                  return_counts=True)

    eta2_sq = np.zeros(nt)          # contribution eta_2K^2 par triangle
    saut_global2 = 0.0
    for e in np.where(cnt == 2)[0]:        # aretes interieures uniquement
        loc = np.where(inv == e)[0]
        K1, K2 = tri_id[loc[0]], tri_id[loc[1]]
        a, bb = cle_vue[e]
        vec = coords[bb] - coords[a]
        hE = np.linalg.norm(vec)
        nE = np.array([vec[1], -vec[0]]) / hE      # normale unitaire a E
        # [[du_h/dn]] est CONSTANT sur E (P1) :
        saut = (ghx[K1] - ghx[K2]) * nE[0] + (ghy[K1] - ghy[K2]) * nE[1]
        norm_saut2 = hE * saut ** 2                # ||[[.]]||^2_{0,E} = |E| * saut^2
        contrib = 0.5 * hE * norm_saut2            # 1/2 h_E ||[[.]]||^2
        eta2_sq[K1] += contrib
        eta2_sq[K2] += contrib
        saut_global2 += hE * norm_saut2

    eta2 = np.sqrt(eta2_sq)
    etaK = np.sqrt(eta1 ** 2 + eta2 ** 2)
    return etaK, eta1, eta2, osc, saut_global2, normf2


# -----------------------------------------------------------------------------
# 7. Etude de convergence : eta vs erreur vraie, indice d'efficacite
# -----------------------------------------------------------------------------

def etude(Ns=(4, 8, 16, 32, 64), nquad=6, verbose=True):
    lam, wts = quadrature_triangle(nquad)
    res = []
    for N in Ns:
        coords, tris, bord = maillage_carre(N)
        uh, aires, gx, gy, hK = assemble_et_resoudre(coords, tris, bord, lam, wts)
        errK, grad_uh = erreur_H1(coords, tris, uh, aires, gx, gy, lam, wts)
        etaK, eta1, eta2, osc, _, _ = estimateur(coords, tris, aires, hK,
                                                 grad_uh, lam, wts)
        err = np.sqrt(np.sum(errK ** 2))
        eta = np.sqrt(np.sum(etaK ** 2))
        E1 = np.sqrt(np.sum(eta1 ** 2))
        E2 = np.sqrt(np.sum(eta2 ** 2))
        OSC = np.sqrt(np.sum(osc ** 2))
        # efficacite locale : max_K eta_K / |u-u_h|_{1,K}
        res.append(dict(N=N, h=1.0 / N, ndof=len(coords), nt=len(tris),
                        err=err, eta=eta, eta1=E1, eta2=E2, osc=OSC,
                        Ieff=eta / err))
    if verbose:
        print("\n  Poisson, Dirichlet homogene, carre unite,  u = x(x-1)y(y-1)")
        print("  Elements finis P1 -- estimateur par residus\n")
        print(f"  {'N':>4} {'h':>8} {'ndof':>7} {'|u-u_h|_1':>12} "
              f"{'eta':>12} {'eta_1':>11} {'eta_2':>11} {'osc':>10} {'I_eff':>8} {'ordre':>7}")
        print("  " + "-" * 100)
        prev = None
        for r in res:
            if prev is None:
                ordre = "   -"
            else:
                ordre = f"{np.log(prev['err']/r['err'])/np.log(2.0):6.3f}"
            print(f"  {r['N']:>4} {r['h']:>8.4f} {r['ndof']:>7} {r['err']:>12.6e} "
                  f"{r['eta']:>12.6e} {r['eta1']:>11.4e} {r['eta2']:>11.4e} "
                  f"{r['osc']:>10.3e} {r['Ieff']:>8.4f} {ordre:>7}")
            prev = r
        print("\n  Lecture : I_eff = eta / |u-u_h|_1 doit tendre vers une CONSTANTE")
        print("  independante de h -> l'estimateur est fiable ET efficace.")
    return res


if __name__ == "__main__":
    etude()
