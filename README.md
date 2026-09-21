# Estimation d'erreur *a posteriori* par résidus — éléments finis P1

**Problème de Poisson sur le carré unité — implémentation FreeFem++, post-traitement Python.**

*Marwane Tchabanna — M2 Modélisation et Analyse Numérique, Université de Montpellier.*

Ce dépôt contient une étude numérique complète d'un estimateur d'erreur *a posteriori*
par résidus pour l'équation de Poisson discrétisée par éléments finis P1 : implémentation,
validation, calibration, étude de l'oscillation des données, et démonstration de l'intérêt
de l'adaptation de maillage sur un cas singulier.

| | |
|---|---|
| **Langages** | FreeFem++ 4.x (solveur), Python 3 / numpy / scipy / matplotlib (analyse) |
| **Résultat principal** | indice d'efficacité borné et convergent : `I_eff` passe de 5.30 à 5.67, limite extrapolée **≈ 5.68**, indépendante de `h` |
| **Validation croisée** | deux implémentations indépendantes (FreeFem++ et numpy) donnent les mêmes valeurs au 5ᵉ chiffre |
| **Démonstration** | sur domaine en L, l'adaptation atteint la même précision avec **22× moins d'inconnues** |

---

## Table des matières

1. [L'idée en deux minutes](#1-lidée-en-deux-minutes)
2. [Le problème mathématique](#2-le-problème-mathématique)
3. [L'estimateur, terme par terme](#3-lestimateur-terme-par-terme)
4. [Objectifs numériques](#4-objectifs-numériques)
5. [Plan de l'étude](#5-plan-de-létude)
6. [Architecture : pourquoi FreeFem++ *et* Python](#6-architecture--pourquoi-freefem-et-python)
7. [Le code FreeFem++ expliqué](#7-le-code-freefem-expliqué)
8. [Le code Python expliqué](#8-le-code-python-expliqué)
9. [Le couplage des deux mondes](#9-le-couplage-des-deux-mondes)
10. [Résultats et interprétation](#10-résultats-et-interprétation)
11. [Reproduire les résultats](#11-reproduire-les-résultats)
12. [Pièges rencontrés](#12-pièges-rencontrés)
13. [Structure du dépôt](#13-structure-du-dépôt)
14. [Références](#14-références)

---

## 1. L'idée en deux minutes

*(section volontairement sans prérequis)*

Quand on simule un phénomène physique — la température dans une pièce, la déformation
d'une poutre, l'écoulement d'un fluide — on remplace une équation continue, qui a une
infinité d'inconnues, par un système linéaire fini que l'ordinateur sait résoudre. On
découpe le domaine en petits triangles (le **maillage**) et on cherche une solution
approchée, affine sur chaque triangle. Plus les triangles sont petits, plus la solution
approchée est proche de la vraie.

Se pose alors la question qui décide de tout en pratique : **comment savoir si le résultat
obtenu est assez précis, alors que justement on ne connaît pas la vraie solution ?**
Si on la connaissait, on ne ferait pas de simulation.

Deux familles de réponses existent.

- L'**estimation a priori** dit : « l'erreur est plus petite que `C·h` ». C'est un théorème
  rassurant, mais inutilisable en pratique, parce que ni la constante `C`, ni la quantité
  qu'elle multiplie ne sont calculables sans connaître la solution exacte. Cela dit comment
  l'erreur *décroît*, pas combien elle *vaut*.

- L'**estimation a posteriori** dit : « voici un nombre η, que je calcule à partir de la
  solution approchée et des données du problème uniquement, et qui encadre l'erreur ».
  C'est un thermomètre. On peut le lire après chaque calcul.

Mieux : ce nombre η se décompose en une contribution par triangle, `η_K`. On obtient donc
une **carte de l'erreur**. Les triangles où `η_K` est grand sont ceux où la solution est
mal représentée. Il suffit de les découper plus finement et de recommencer. C'est
l'**adaptation de maillage**, et c'est ce qui permet de mettre les degrés de liberté
là où ils servent au lieu de raffiner partout bêtement.

> **Une image pour fixer les idées.** Imagine que tu dois repeindre un mur en n'ayant le
> droit de regarder que ton propre travail, jamais le mur « idéal ». Tu ne peux pas
> comparer à la perfection, mais tu peux repérer les coulures, les raccords mal faits entre
> deux coups de rouleau, les zones où la peinture est trop fine. Un estimateur a posteriori
> fait exactement cela : il inspecte la solution calculée pour y chercher les **traces
> visibles** de l'erreur, sans jamais avoir besoin de la vraie solution.

Ce projet fait deux choses :

1. **Vérifier que le thermomètre est bien gradué.** On se place sur un problème dont on
   connaît la solution exacte — on triche volontairement, une fois — pour comparer η à
   l'erreur vraie et mesurer la qualité de l'estimateur.
2. **S'en servir.** Sur un domaine en forme de L, où la solution possède une singularité
   de coin, on montre que l'estimateur repère tout seul la zone problématique et guide un
   raffinement bien plus économique que le raffinement uniforme.

---

## 2. Le problème mathématique

On résout l'équation de Poisson avec condition de Dirichlet homogène sur le carré unité
`Ω = ]0,1[²` :

```
-Δu = f     dans Ω
   u = 0    sur ∂Ω
```

*Physiquement*, on peut lire `u` comme la température d'une plaque carrée dont les bords
sont maintenus à 0 °C, et `f` comme une source de chaleur répartie dans la plaque.

Sa **formulation variationnelle** : trouver `u ∈ H¹₀(Ω)` tel que

```
∫_Ω ∇u·∇v  =  ∫_Ω f v      pour tout v ∈ H¹₀(Ω)
```

La discrétisation P1 consiste à chercher `u_h` dans l'espace des fonctions continues,
affines sur chaque triangle et nulles au bord :

```
V_h = { v_h ∈ C⁰(Ω̄) : v_h|_K ∈ P¹(K) ∀K ∈ T_h,  v_h = 0 sur ∂Ω }
```

Concrètement, `u_h` ressemble à une **surface faite de facettes planes triangulaires**
recollées bord à bord, comme un dôme géodésique : continue partout, mais avec des « plis »
le long des arêtes.

### La solution manufacturée

Pour pouvoir mesurer l'erreur vraie, on choisit la solution d'abord et on en déduit le
second membre — c'est la **méthode des solutions manufacturées** :

```
u(x,y) = x(x-1)·y(y-1)
```

Elle s'annule sur les quatre côtés du carré (`x=0`, `x=1`, `y=0`, `y=1`), donc la condition
de Dirichlet homogène est automatiquement satisfaite. C'est la première chose à vérifier :
si `u` ne s'annulait pas au bord, tout le reste de l'étude serait faussé.

En dérivant deux fois :

```
∂²u/∂x² = 2·y(y-1)
∂²u/∂y² = 2·x(x-1)
Δu      = 2[ y(y-1) + x(x-1) ]
```

d'où le second membre

```
f(x,y) = -Δu = 2[ x(1-x) + y(1-y) ]
```

qui est positif sur tout le carré, maximal au centre (`f(½,½) = 1`) et nul aux quatre coins.
Ce détail servira à interpréter les cartes d'indicateurs.

Le gradient exact, dont on aura besoin pour calculer l'erreur en norme d'énergie :

```
∂u/∂x = (2x-1)·y(y-1)
∂u/∂y = x(x-1)·(2y-1)
```

---

## 3. L'estimateur, terme par terme

L'erreur se mesure en **semi-norme H¹**, aussi appelée norme d'énergie :

```
|u - u_h|²_{1,Ω} = ∫_Ω |∇u - ∇u_h|²
```

C'est la norme naturelle du problème : c'est elle qui apparaît dans le lemme de Céa et
dans laquelle la méthode de Galerkin est optimale. Elle mesure l'écart entre les **pentes**
de la vraie solution et de la solution approchée, pas seulement entre leurs valeurs.

L'estimateur par résidus se compose de trois termes, définis triangle par triangle.

### η₁K — le résidu volumique

```
η₁K = h_K · ‖ f + Δu_h ‖_{0,K}
```

**Ce qu'il mesure :** à quel point la solution approchée ne satisfait pas l'équation
`-Δu = f` *à l'intérieur* du triangle K. On réinjecte `u_h` dans l'équation et on regarde
ce qui reste. Le facteur `h_K` (le diamètre du triangle) est un poids d'échelle : il vient
de l'analyse et rend le terme homogène à une norme d'énergie.

**Simplification en P1 :** `u_h` est affine sur chaque triangle, donc ses dérivées secondes
sont nulles et `Δu_h = 0`. Le terme se réduit à

```
η₁K = h_K · ‖f‖_{0,K}
```

> **En clair :** une facette plane n'a aucune courbure. Or l'équation demande une courbure
> égale à `f`. Là où `f` est grand, la solution devrait être très bombée, et une facette
> plate est une mauvaise approximation : η₁K le signale.

### η₂K — les sauts de flux

```
η₂K = ( ½ · Σ_{E ⊂ ∂K, E intérieure}  h_E · ‖ [[∂u_h/∂n]] ‖²_{0,E} )^{1/2}
```

**Ce qu'il mesure :** la solution exacte a un flux `∂u/∂n` continu à travers toute interface.
La solution P1, elle, a un gradient constant par triangle, donc discontinu d'un triangle à
l'autre. Le **saut** `[[∂u_h/∂n]]` de la dérivée normale à travers une arête quantifie ce
défaut de régularité. Plus le saut est grand, plus le maillage est trop grossier pour suivre
la solution à cet endroit.

> **En clair :** reprends l'image du dôme à facettes. Le long de chaque arête, la surface
> fait un pli. La vraie solution, elle, est lisse : aucun pli. Plus le pli est marqué
> (grand changement de pente d'une facette à l'autre), plus la surface facettée s'écarte
> de la surface lisse. η₂K mesure « l'angle des plis ».
>
> En termes de chaleur : le flux de chaleur qui sort d'un triangle doit être exactement
> celui qui entre dans son voisin. Un saut de flux, c'est de la chaleur qui « apparaît » ou
> « disparaît » sur l'arête, ce qui est physiquement impossible, donc un signe d'erreur.

Le facteur `½` répartit équitablement la contribution d'une arête entre les deux triangles
qui la partagent.

**Seules les arêtes intérieures comptent.** Sur une arête de bord, il n'y a pas de voisin
de l'autre côté, donc pas de saut à mesurer. Ce point, facile à écrire, est facile à rater
dans le code : voir le piège n°3 au §12, qui a réellement faussé une version de ce projet.

### η₃K — le terme de bord

```
η₃K = 0   ici
```

Ce terme prend en charge les conditions de Neumann non homogènes (écart entre le flux imposé
`g_E` et le flux calculé `∂u_h/∂n`). Avec un Dirichlet homogène, il n'y a pas de flux imposé
et les termes de bord s'annulent — comme établi dans la partie théorique du cours. Le
reconnaître évite d'ajouter à tort des contributions fantômes sur le bord.

### Assemblage

```
η_K² = η₁K² + η₂K²                 (indicateur local, un nombre par triangle)
η²   = Σ_{K ∈ T_h} η_K²            (estimateur global, un seul nombre)
```

### L'oscillation des données : f_K et osc_K

L'estimateur ci-dessus utilise `f` telle quelle. Mais dans l'analyse de l'efficacité, on
est obligé de remplacer `f` sur chaque triangle par une version simplifiée : sa **valeur
moyenne**

```
f_K = (1/|K|) ∫_K f
```

et ce qu'on perd dans ce remplacement s'appelle l'**oscillation des données** :

```
osc_K = h_K · ‖ f - f_K ‖_{0,K}         osc² = Σ_K osc_K²
```

> **En clair :** sur un petit triangle, `f` ne varie presque pas. La remplacer par sa
> moyenne, c'est comme décrire le relief d'un champ par son altitude moyenne : sur un
> mouchoir de poche, l'erreur est minuscule. `osc_K` mesure exactement cette erreur de
> « résumé ».

**Pourquoi s'en soucier ?** Parce que l'efficacité locale du cours ne s'écrit pas
`η_K ≤ C·|u-u_h|_{1,K}` mais

```
η_K  ≤  C · ( |u - u_h|_{1,ω_K}  +  osc(f, ω_K) )
```

où `ω_K` est le **patch** de K (K et ses voisins). Le terme d'oscillation est le « prix à
payer » pour que le théorème soit vrai pour n'importe quel `f`. Tout l'enjeu est de vérifier
qu'il est **négligeable** devant l'erreur.

**Pourquoi il est d'ordre supérieur.** Sur un triangle de taille `h`, `f - f_K` est de
l'ordre de `h·|∇f|` (on s'écarte de la moyenne proportionnellement à la distance). En
multipliant par le poids `h_K` et en sommant, on obtient

```
osc = O(h²)     alors que     η = O(h)
```

Quand on divise `h` par 2, η est divisé par 2 mais osc est divisé par 4 : l'oscillation
devient vite invisible. C'est ce qu'on vérifie numériquement (§10.1, ordre mesuré 1.9999).

### Les deux théorèmes et l'indice d'efficacité

```
Fiabilité   :  |u - u_h|₁  ≤  C_rel · η                  ⟹  I_eff ≥ 1/C_rel
Efficacité  :  η  ≤  C_eff · ( |u - u_h|₁ + osc )        ⟹  I_eff ≤ C_eff (à osc près)
```

avec

```
I_eff = η / |u - u_h|_{1,Ω}
```

La **fiabilité** garantit que l'estimateur ne sous-estime jamais l'erreur : si η est petit,
l'erreur l'est aussi. C'est ce qui permet de l'utiliser comme critère d'arrêt certifié.
L'**efficacité** garantit qu'il ne surestime pas indéfiniment : sans elle, un estimateur
constant égal à 10⁶ serait « fiable » et parfaitement inutile.

Les deux théorèmes affirment l'*existence* des constantes sans donner leur valeur. C'est
précisément le rôle du calcul numérique que de les mesurer.

> **Pourquoi `I_eff` ne vaut pas 1, et pourquoi ce n'est pas grave.** Un thermomètre qui
> affiche toujours le double de la vraie température est parfaitement utilisable, à condition
> que ce soit **toujours** le double : on sait le lire. Ce qui serait grave, c'est un
> thermomètre qui afficherait le double à 20 °C et le décuple à 30 °C. Pour un estimateur,
> c'est pareil : la valeur de `I_eff` importe peu, ce qui compte c'est qu'elle **reste
> stable** quand on raffine le maillage.

---

## 4. Objectifs numériques

L'étude doit répondre à cinq questions, chacune avec un critère chiffré.

| # | Question | Critère de réussite |
|---|---|---|
| **1** | η et `\|u-u_h\|₁` suivent-ils la même pente en log-log ? | pentes égales, toutes deux ≈ 1 |
| **2** | Quelle est la valeur des constantes ? | `I_eff` se stabilise à une valeur modérée |
| **3** | L'oscillation est-elle négligeable ? | `osc` décroît en `O(h²)`, plus vite que η |
| **4** | L'estimateur localise-t-il correctement l'erreur ? | les cartes de `η_K` et de `\|u-u_h\|_{1,K}` ont leurs maxima aux mêmes endroits |
| **5** | L'adaptation paie-t-elle ? | erreur en fonction du nombre de ddl : l'adaptatif domine l'uniforme sur un cas singulier |

La question **1** est la plus importante théoriquement : si η décroissait plus lentement
que l'erreur, l'estimateur deviendrait de plus en plus pessimiste et cesserait d'être
exploitable. La question **4** est la plus importante pratiquement : c'est elle qui décide
si l'adaptation de maillage fonctionne, et elle porte sur le **classement** des éléments,
pas sur les valeurs absolues.

---

## 5. Plan de l'étude

### Étape 0 — Poser le cadre
Formulation variationnelle, discrétisation P1, choix de la solution manufacturée
`u = x(x-1)y(y-1)` sur le carré unité, calcul de `f = 2[x(1-x)+y(1-y)]`.
**Vérification obligatoire :** `u` s'annule bien sur les quatre côtés.

### Étape 1 — Valider le solveur *avant* l'estimateur
Étude de convergence a priori : `|u-u_h|₁` doit être en `O(h)` et `‖u-u_h‖₀` en `O(h²)`.
**Critère :** ordres mesurés 1 et 2.
Sauter cette étape est l'erreur classique — si le solveur est faux, toute l'analyse de
l'estimateur mesure le mauvais objet.
→ *Obtenu : 0.9998 et 1.9997 sur le maillage le plus fin.*

### Étape 2 — Implémenter les trois termes
`η₁K`, `η₂K`, `η₃K` (nul ici).
**Critère :** η décroît, et `η₁` et `η₂` sont du même ordre de grandeur. Si l'un domine
l'autre d'un facteur 100, il y a une erreur de facteur ou d'échelle en `h`.
→ *Obtenu : sur le maillage le plus fin, `η₁²` pèse 51 % de `η²` et `η₂²` 49 %.
Parfaitement équilibré.*

### Étape 2bis — Moyenne de f et oscillation
Calcul de `f_K` (vraie moyenne, par intégration) et de `osc_K = h_K‖f - f_K‖_{0,K}`.
**Critère :** `osc = O(h²)`.
→ *Obtenu : ordre 1.9999.*

### Étape 3 — Calibrer
Tableau de `I_eff` en fonction de `h`.
**Critère :** stabilisation vers une constante.
→ *Obtenu : `I_eff` croît de 5.30 à 5.67 avec des incréments divisés par 2 à chaque
raffinement ; limite extrapolée ≈ 5.68.*

### Étape 4 — Localisation
Cartes de `η_K` et du rapport `η_K / |u-u_h|_{1,K}`.
**Critère :** les zones de fort `η_K` coïncident avec les zones de forte erreur.
→ *Obtenu, avec un effet parasite d'alignement du maillage identifié et expliqué
(voir §10).*

### Étape 5 — Casser la régularité
Domaine en L, solution singulière `u = r^(2/3)·sin(2θ/3)`, avec `f = 0`.
**Objectif :** montrer que le raffinement uniforme perd l'ordre optimal
(`ndof^(-1/3)` au lieu de `ndof^(-1/2)`) et que **l'estimateur voit cette dégradation
sans qu'on lui donne `u`**. C'est le passage du régime « vérification » au régime
« utilisation ».

### Étape 6 — La boucle adaptative
```
RÉSOUDRE  →  ESTIMER (η_K)  →  MARQUER  →  RAFFINER  →  …
```
Stratégie de marquage par équidistribution : on vise `η_K` identique sur tous les
triangles, avec une taille cible `h_K^new = h_K·√(η_cible/η_K)`.
**Critère :** ordre optimal `ndof^(-1/2)` restauré malgré la singularité.
→ *Obtenu : 634 ddl adaptatifs atteignent la précision de 14 082 ddl uniformes.*

### Étape 7 — Ouvertures
Deux directions naturelles, non traitées ici mais documentées :
- comparer à un estimateur par **récupération de gradient** (Zienkiewicz–Zhu), bien moins
  cher et souvent plus proche de `I_eff = 1`, mais sans garantie de fiabilité ;
- traiter un **bord de Neumann non homogène**, pour que `η₃K` cesse d'être nul, et
  vérifier que la calibration tient.

---

## 6. Architecture : pourquoi FreeFem++ *et* Python

### Le partage des rôles

**FreeFem++ est le noyau de calcul.** Sa syntaxe colle à la formulation variationnelle — on
écrit `int2d(Th)(dx(u)*dx(v) + dy(u)*dy(v))` presque comme au tableau — et il fournit
gratuitement tout ce qui serait long à réimplémenter : mailleur, assemblage, solveurs
linéaires, `adaptmesh`, et surtout les primitives propres aux estimateurs (`intalledges`,
`jump`, `nTonEdge`, `lenEdge`, `hTriangle`).

**Python est le chef d'orchestre et l'analyste.** Il fait tout ce que FreeFem++ fait mal ou
pas du tout : boucler sur des paramètres, ajuster des pentes, extrapoler, produire des
figures publiables, agréger des résultats, archiver. Et il ouvre l'écosystème : une fois les
données sorties en texte, elles sont à un `import` de numpy, pandas, scikit-learn ou PyTorch.

**Le fichier texte est l'interface.** Ce découplage a une vertu propre : le calcul et
l'analyse deviennent indépendants. On peut refaire vingt figures sans relancer une seule
résolution. Le solveur tourne en batch sans écran, l'analyse tourne dans un notebook.

C'est le schéma standard de tout code de simulation sérieux — FreeFem/ParaView,
Code_Aster/Salome, OpenFOAM/PyVista. Ce projet en construit une version minimale, ce qui
est la meilleure façon de comprendre pourquoi elle est faite ainsi.

### Le flux de données

```
┌─────────────────────────────┐
│   estimateur_residu_P1.edp  │   FreeFem++
│                             │
│   maillage → assemblage     │
│   → résolution → η_K        │
│   → f_K, osc_K              │
└──────────────┬──────────────┘
               │  écrit
               ▼
   ┌───────────────────────┐
   │  resultats.txt        │   tableau de convergence (8 colonnes)
   │  maillage_etaK.txt    │   maillage + 4 champs P0 (η_K, erreur locale, f_K, osc_K)
   └───────────┬───────────┘
               │  relu par
               ▼
┌─────────────────────────────┐
│     plot_resultats.py       │   Python / numpy / matplotlib
│                             │
│   lecture → Triangulation   │
│   → 6 panneaux → PNG        │
└──────────────┬──────────────┘
               ▼
       figure_estimateur.png
```

---

## 7. Le code FreeFem++ expliqué

Fichier : [`code/estimateur_residu_P1.edp`](code/estimateur_residu_P1.edp)

### 7.1 Les données

```cpp
func f      = 2*(x*(1-x) + y*(1-y));      // second membre
func uex    = x*(x-1)*y*(y-1);            // solution exacte
func dxuex  = (2*x-1)*y*(y-1);            // du/dx
func dyuex  = x*(x-1)*(2*y-1);            // du/dy
```

`func` définit une expression symbolique en `x` et `y`, évaluée à la volée aux points de
quadrature. Ce n'est pas une fonction éléments finis : il n'y a ni maillage ni degrés de
liberté derrière, c'est une formule. C'est exactement ce qu'on veut pour la solution exacte,
qu'on ne veut surtout pas interpoler (interpoler `uex` sur le maillage introduirait une
erreur supplémentaire et fausserait la mesure).

À l'inverse, **`f_K` ne peut pas être une `func`** : c'est une moyenne sur chaque triangle,
elle dépend donc du maillage et doit être recalculée à chaque raffinement (§7.5).

### 7.2 La résolution

```cpp
mesh Th = square(Nx, Nx);        // maillage uniforme, labels 1,2,3,4 sur les 4 côtés
fespace Vh(Th, P1);              // espace d'approximation
fespace Ph(Th, P0);              // une valeur par triangle : les indicateurs
Vh u, v;

solve Poisson(u, v)
    = int2d(Th)( dx(u)*dx(v) + dy(u)*dy(v) )
    - int2d(Th, qforder=5)( f*v )
    + on(1, 2, 3, 4, u = 0);
```

`solve` assemble **et** résout en une instruction. La syntaxe est la transcription directe
de la formulation variationnelle : forme bilinéaire, moins la forme linéaire, plus les
conditions essentielles. FreeFem++ reconnaît `u` comme inconnue et `v` comme fonction test
d'après l'ordre de déclaration dans `Poisson(u, v)`.

`on(1,2,3,4, u=0)` impose Dirichlet sur les quatre labels de bord que `square` a
automatiquement attribués.

`qforder` fixe l'ordre de la formule de quadrature, c'est-à-dire le degré des polynômes
qu'elle intègre exactement. Par défaut, FreeFem++ utilise en 2D une formule à 7 points exacte
jusqu'au degré 5. Ici `f·v` est de degré 3 : le défaut suffit, `qforder=5` est une marge. Le
paramètre devient **indispensable** quand l'intégrande dépasse le degré 5, ce qui est le cas
de l'erreur d'énergie (§7.3).

### 7.3 L'erreur vraie, triangle par triangle

```cpp
varf erreurLoc(unused, chiK)                       // |u-u_h|²_{1,K}
    = int2d(Th, qforder=8)( chiK * ( square(dx(u)-dxuex)
                                   + square(dy(u)-dyuex) ) );
errKsq[] = erreurLoc(0, Ph);
real err = sqrt( errKsq[].sum );                   // |u-u_h|_1
```

On calcule d'abord l'erreur **sur chaque triangle** (même astuce `varf` que pour les
indicateurs, expliquée au §7.4), puis l'erreur globale comme racine de la somme des carrés.
C'est cohérent avec la définition : `∫_Ω = Σ_K ∫_K`. On obtient ainsi les deux informations
d'un coup, la globale pour `I_eff` et la locale pour l'efficacité triangle par triangle.

Le gradient exact a des composantes de degré 3, leur carré est de degré 6 : au-delà de ce
qu'intègre la formule par défaut, d'où `qforder=8`. *(Sur ce cas test, la validation croisée
avec la version numpy — quadrature exacte jusqu'au degré 10 — montre que l'écart reste sous
le 5ᵉ chiffre ; mais une quadrature insuffisante ne produit aucun message d'erreur, donc on
la fixe explicitement.)*

### 7.4 Les indicateurs locaux — l'astuce centrale

C'est le cœur technique du code, et l'idiome n'est pas évident à deviner.

```cpp
varf residuVol(unused, chiK)                       // η₁K²
    = int2d(Th)( chiK * square(hTriangle) * square(f + dxx(u)+dyy(u)) );

varf sautFlux(unused, chiK)                        // η₂K²
    = intalledges(Th)( chiK * 0.5 * (nTonEdge-1) * lenEdge
                       * square( jump( N.x*dx(u) + N.y*dy(u) ) ) );

Ph eta1sq, eta2sq;
eta1sq[] = residuVol(0, Ph);
eta2sq[] = sautFlux(0, Ph);
```

**Pourquoi une `varf` ?** On veut *une intégrale par triangle*, pas une intégrale globale.
L'astuce consiste à écrire une forme linéaire testée contre une fonction `chiK` de l'espace
`P0`. Comme les fonctions de base de `P0` sont les fonctions caractéristiques des triangles
(valeur 1 sur un triangle, 0 partout ailleurs), le vecteur résultat contient, composante par
composante, l'intégrale restreinte à chaque triangle. La ligne `eta1sq[] = residuVol(0, Ph)`
déclenche l'assemblage et remplit le tableau.

> **En clair :** c'est comme demander à FreeFem++ « calcule l'intégrale en ne regardant que
> le triangle n°1 », puis « que le triangle n°2 », etc. — sauf qu'il le fait pour tous les
> triangles d'un seul coup.

`unused` est un argument formel obligatoire par la syntaxe des `varf`, jamais utilisé ici.

**`dxx(u)+dyy(u)`** vaut exactement 0 en P1. On l'écrit quand même pour que le code reste
correct si l'on passe en P2, où le laplacien discret ne s'annule plus.

**`intalledges` et le facteur ½.** `intalledges(Th)` parcourt les trois arêtes de *chaque*
triangle. Une arête intérieure est donc visitée deux fois, une fois depuis chaque côté —
ce qui réalise tout seul le partage ½–½ du saut entre les deux triangles voisins, exactement
comme la formule théorique le demande.

**`nTonEdge` et l'exclusion du bord.** Cette variable vaut 1 sur une arête de bord et 2 sur
une arête intérieure. Le facteur `(nTonEdge-1)` vaut donc 0 au bord et 1 à l'intérieur :
c'est ainsi qu'on exclut les arêtes de bord sans écrire de test. **Sans lui, le résultat est
faux** — et pas marginalement : voir le piège n°3 (§12), où l'indice d'efficacité passait de
5.5 à 13.2 au lieu de se stabiliser.

**`jump`, `N.x`, `N.y`, `lenEdge`, `hTriangle`.** `N` est la normale unitaire à l'arête
courante, donc `N.x*dx(u) + N.y*dy(u)` est la dérivée normale. `jump(...)` en calcule le saut
à travers l'arête. `lenEdge` est la longueur de l'arête courante (le `h_E` de la théorie) et
`hTriangle` le diamètre du triangle courant (le `h_K`). Ces quantités géométriques sont
fournies par FreeFem++ et correspondent exactement aux échelles de l'analyse.

### 7.5 La moyenne f_K et l'oscillation

```cpp
varf aireVar(unused, chiK) = int2d(Th)( chiK );                  // |K|
varf moyF   (unused, chiK) = int2d(Th, qforder=5)( chiK * f );   // ∫_K f

Ph aireK, fK;
aireK[] = aireVar(0, Ph);
fK[]    = moyF(0, Ph);
fK[]  ./= aireK[];                                  // f_K = ∫_K f / |K|

varf oscData(unused, chiK)                          // osc_K²
    = int2d(Th, qforder=5)( chiK * square(hTriangle) * square(f - fK) );
```

On réutilise exactement l'astuce `varf` du §7.4 : tester contre `chiK` donne `∫_K f`, et
tester `1` contre `chiK` donne l'aire `|K|`. La division composante par composante `./=`
fournit la moyenne.

**Pourquoi ne pas écrire simplement `Ph fK = f;` ?** Parce que l'interpolation P0 de
FreeFem++ évalue `f` **en un seul point** du triangle (son barycentre), ce qui n'est pas la
moyenne. L'écart est petit, en `O(h²)`, mais c'est justement l'ordre de grandeur de
l'oscillation qu'on cherche à mesurer : on fausserait la quantité même qu'on étudie.

### 7.6 L'efficacité : globale et locale

```cpp
real Ieff = eta / err;                        // un nombre
Ph ratio  = sqrt(etaKsq) / sqrt(errKsq);      // un nombre par triangle
```

Deux notions à ne pas confondre :

- l'**efficacité globale** `I_eff = η / |u-u_h|₁` est un seul nombre par maillage ; c'est
  elle qu'on suit en fonction de `h` ;
- l'**efficacité locale** `η_K / |u-u_h|_{1,K}` est un champ P0, un nombre par triangle ;
  c'est elle qu'on trace sous forme de carte.

La première est un `real`, la seconde un `Ph`. Seule la seconde peut être passée à `plot`
(voir piège n°6).

### 7.7 L'export vers Python

```cpp
ofstream mm("maillage_etaK.txt");
mm << Th.nv << " " << Th.nt << endl;
for (int iv = 0; iv < Th.nv; iv++)
    mm << Th(iv).x << " " << Th(iv).y << endl;
for (int it = 0; it < Th.nt; it++)
    mm << int(Th[it][0]) << " " << int(Th[it][1]) << " "
       << int(Th[it][2]) << " " << etaK[][it] << " "
       << sqrt(errKsq[][it]) << " " << fK[][it] << " "
       << sqrt(oscsq[][it]) << endl;
```

Pour retracer un champ éléments finis ailleurs que dans FreeFem++, il faut transmettre
**trois choses** :

1. la **géométrie** — les coordonnées des sommets ;
2. la **topologie** — quels sommets forment quel triangle ;
3. les **valeurs des degrés de liberté**, plus l'information de quel espace il s'agit.

Le point 3 est le plus subtil : un tableau de nombres ne veut rien dire tant qu'on ne sait
pas s'il s'agit d'une valeur par triangle (P0) ou par sommet (P1). C'est cela qui décidera,
côté Python, de la primitive de tracé à employer.

**Homogénéité des colonnes.** Les tableaux `errKsq` et `oscsq` contiennent des **carrés**.
On exporte leur racine pour que les quatre colonnes de valeurs (`η_K`, `|u-u_h|_{1,K}`,
`f_K`, `osc_K`) soient toutes dans la même unité. Mélanger des carrés et des non-carrés
fausserait silencieusement tout rapport calculé en aval.

Détails de syntaxe qui méritent d'être connus :

- `Th(iv)` avec des **parenthèses** désigne le sommet numéro `iv` ; `Th[it]` avec des
  **crochets** désigne le triangle numéro `it`. Ne pas confondre.
- `Th[it][j]` est le j-ième sommet du triangle `it`, mais c'est un *objet sommet*, pas un
  entier. Le `int(...)` autour force la conversion vers le numéro global. Sans lui, l'export
  serait inexploitable.
- `etaK[][it]` : pour une fonction éléments finis `u`, `u[]` donne accès au **vecteur des
  degrés de liberté** et `u[][i]` à sa i-ème composante. Pour un espace P0, FreeFem++
  numérote les ddl comme les triangles, d'où la correspondance directe. **Cette identité est
  propre à P0** : en P2 la numérotation mêle sommets et milieux d'arêtes, et un export naïf
  ne voudrait plus rien dire.
- `cout.precision(5)` ne s'applique qu'à `cout`. Pour un export destiné à du calcul en aval,
  ajouter `mm.precision(12);` après l'ouverture du flux.

---

## 8. Le code Python expliqué

Fichier : [`code/plot_resultats.py`](code/plot_resultats.py)

Ce script ne fait **aucun calcul éléments finis**. Il relit et il trace.

### 8.1 Le tableau de convergence

```python
tab = np.loadtxt("resultats.txt", skiprows=1)
N, h, err, eta, eta1, eta2, osc, Ieff = tab.T
```

`resultats.txt` est un tableau homogène : `loadtxt` suffit, `skiprows=1` saute l'en-tête, et
`.T` transpose pour dépaqueter les **huit** colonnes en huit variables nommées d'un coup.
Si le nombre de variables à gauche ne correspond pas au nombre de colonnes du fichier,
Python s'arrête avec `too many values to unpack` : c'est le premier symptôme à reconnaître
quand on ajoute une colonne côté FreeFem++ sans mettre à jour le script.

### 8.2 Le maillage

```python
with open(nom) as fh:
    nv, nt = map(int, fh.readline().split())
    coords = np.array([list(map(float, fh.readline().split())) for _ in range(nv)])
    data   = np.array([list(map(float, fh.readline().split())) for _ in range(nt)])
tris = data[:, :3].astype(int)
etaK, errK, fK, oscK = data[:, 3], data[:, 4], data[:, 5], data[:, 6]
```

`maillage_etaK.txt` est hétérogène — une ligne d'en-tête, puis `nv` lignes à 2 colonnes, puis
`nt` lignes à 7 colonnes. `loadtxt` ne sait pas gérer cela, d'où la lecture manuelle. C'est
ici que le `nv nt` de la première ligne sert : il pilote exactement combien de lignes
appartiennent à chaque bloc, ce qui rend le format auto-descriptif.

`.astype(int)` est indispensable : tout a été lu en flottants, et un indice flottant fait
échouer l'indexation numpy.

### 8.3 La reconstruction de la triangulation

```python
triang = mtri.Triangulation(coords[:, 0], coords[:, 1], tris)
```

`matplotlib.tri.Triangulation` reconstruit côté Python exactement la même triangulation que
celle de FreeFem++, et il lui faut précisément les deux mêmes ingrédients : coordonnées et
connectivité. Rien de plus.

**Point de vigilance :** matplotlib attend une numérotation **à partir de 0**. FreeFem++ l'est
aussi, donc cela tombe juste. Si un jour tu lis un format numéroté à partir de 1 (certains
`.msh`), il faudra écrire `tris - 1`. Le test rapide : `tris.max()` doit valoir `nv - 1`.

### 8.4 Le tracé du champ P0

```python
tp = ax[2].tripcolor(triang, facecolors=etaK, cmap="viridis")
```

Le mot-clé **`facecolors=`** signifie : une couleur *par face*, donc une valeur constante par
triangle. C'est la traduction graphique exacte de « η_K est un champ P0 ».

Si l'on passait le même tableau en second argument positionnel, matplotlib attendrait une
valeur *par sommet* et interpolerait linéairement : ce serait la bonne façon de tracer un
champ P1 (`tripcolor(triang, u_aux_sommets, shading="gouraud")` ou `tricontourf`), mais ce
serait faux pour un indicateur. **La primitive de tracé doit correspondre à l'espace
éléments finis.**

### 8.5 La limite de I_eff par extrapolation de Richardson

```python
Ieff_inf = Ieff[-1] + (Ieff[-1] - Ieff[-2])
ax[1].axhline(Ieff_inf, ls="--", c="k")                  # droite en pointillés
ax[1].axhspan(Ieff.min(), Ieff.max(), alpha=0.08)        # bande [min, max]
```

On ne peut pas calculer `I_eff` pour `h = 0`, mais on peut **deviner la limite** à partir
de la façon dont les valeurs s'en approchent.

> **En clair :** si tu marches vers un mur en faisant à chaque fois la moitié du pas
> précédent (1 m, puis 50 cm, puis 25 cm…), et que ton dernier pas faisait 1 cm, alors il te
> reste exactement 1 cm avant le mur : la somme 0.5 + 0.25 + … vaut 1.

C'est ce qui se passe ici : les écarts successifs de `I_eff` sont 0.187, 0.097, 0.047, 0.023,
0.011 — **divisés par 2 à chaque fois** (convergence en `O(h)`). Il reste donc à parcourir
environ le dernier écart, d'où la formule. La droite en pointillés matérialise cette limite ;
la bande grisée montre que toutes les valeurs calculées restent dans un intervalle étroit.
C'est la preuve visuelle que `I_eff` est **borné indépendamment de `h`**.

### 8.6 Le backend sans fenêtre, et l'option `--show`

```python
AFFICHER = "--show" in sys.argv
if not AFFICHER:
    matplotlib.use("Agg")      # AVANT d'importer pyplot
import matplotlib.pyplot as plt
```

`Agg` est un backend qui dessine en mémoire et écrit un fichier, sans jamais ouvrir de
fenêtre. C'est ce qui permet au script de tourner en SSH, dans WSL ou sur un cluster.
L'appel doit impérativement précéder `import matplotlib.pyplot`, sinon le backend est déjà
choisi et l'instruction est ignorée silencieusement. Avec `python3 plot_resultats.py --show`,
on garde le backend interactif et une fenêtre s'ouvre en plus du PNG.

*Dans un notebook Jupyter*, remplacer ces lignes par `%matplotlib inline` et ajouter
`plt.show()` à la fin pour un affichage direct sous la cellule.

---

## 9. Le couplage des deux mondes

### Sens FreeFem++ → Python : les fichiers

**`resultats.txt`** — un tableau, une ligne par maillage :

```
    N      h        |u-u_h|_1       eta          eta_1        eta_2       osc         I_eff
  4   0.25   0.058777   0.31177   0.24721   0.18997   0.033268   5.3042
  8   0.125  0.030161   0.16562   0.12360   0.11024   0.0084585  5.4913
  ...
```

**`maillage_etaK.txt`** — le maillage et quatre champs P0 :

```
16641 32768                    <- nv nt
0 0                            <- 16641 lignes : x y
0.0078125 0
...
0 1 130  eta_K  err_K  f_K  osc_K     <- 32768 lignes : i0 i1 i2 + 4 valeurs
...
```

### Sens Python → FreeFem++ : le pilotage

On peut aussi faire piloter le solveur **par** Python, ce qui est très pratique pour une
étude paramétrique. FreeFem++ expose un tableau prédéfini `ARGV` contenant les arguments de
la ligne de commande, sans aucun plugin :

```cpp
int n0 = 4;
for (int i = 0; i < ARGV.n-1; i++) if (ARGV[i] == "-n") n0 = atoi(ARGV[i+1]);
```

*(Surtout pas `int N` : voir le piège n°1.)*

et côté Python :

```python
import subprocess
for n in [8, 16, 32, 64, 128]:
    subprocess.run(["FreeFem++", "-nw", "estimateur_residu_P1.edp", "-n", str(n)],
                   check=True)
```

Python devient alors le chef d'orchestre : il balaie les paramètres, relance le solveur,
agrège les sorties et trace. C'est le squelette de n'importe quelle étude de convergence
automatisée ou boucle d'adaptation pilotée.

### Et pour aller plus loin : le format VTK

Le format texte maison est parfait pédagogiquement — on voit exactement ce qui circule — mais
il ne passe pas à l'échelle. Dès qu'on fait du 3D ou du vectoriel :

```cpp
load "iovtk"
savevtk("solution.vtu", Th, u, etaK, dataname="u etaK");
```

Côté Python, `meshio.read("solution.vtu")` rend points, cellules et champs dans des tableaux
numpy, et le même fichier s'ouvre directement dans ParaView.

---

## 10. Résultats et interprétation

### 10.1 Convergence et calibration

| N | h | \|u-u_h\|₁ | η | η₁ | η₂ | osc | I_eff |
|---|---|---|---|---|---|---|---|
| 4 | 0.2500 | 5.8777e-2 | 3.1177e-1 | 2.4721e-1 | 1.8997e-1 | 3.3268e-2 | 5.3042 |
| 8 | 0.1250 | 3.0161e-2 | 1.6562e-1 | 1.2360e-1 | 1.1024e-1 | 8.4585e-3 | 5.4913 |
| 16 | 0.0625 | 1.5181e-2 | 8.4827e-2 | 6.1802e-2 | 5.8105e-2 | 2.1234e-3 | 5.5878 |
| 32 | 0.0313 | 7.6030e-3 | 4.2838e-2 | 3.0901e-2 | 2.9669e-2 | 5.3139e-4 | 5.6344 |
| 64 | 0.0156 | 3.8031e-3 | 2.1514e-2 | 1.5450e-2 | 1.4971e-2 | 1.3288e-4 | 5.6569 |
| 128 | 0.0078 | 1.9017e-3 | 1.0779e-2 | 7.7252e-3 | 7.5173e-3 | 3.3223e-5 | **5.6680** |

Ordres mesurés entre les deux maillages les plus fins :

| quantité | ordre observé | ordre attendu |
|---|---|---|
| `\|u-u_h\|₁` | 0.9998 | 1 |
| `η` | 0.9970 | 1 |
| `osc` | 1.9999 | 2 |
| `‖u-u_h‖₀` | 1.9997 | 2 |

Limite extrapolée : **`I_eff → ≈ 5.68`**.

![Figure de validation](figures/figure_estimateur.png)

La figure comporte six panneaux, (a) à (c) en haut, (d) à (f) en bas.

**(a) Convergence.** Des droites en log-log, donc des lois de puissance. Ce qui compte n'est
pas leur position mais leur **pente**. L'erreur, η, η₁ et η₂ ont toutes une pente 1 : l'erreur
et l'estimateur décroissent au même rythme, ce qui est le résultat non trivial. L'oscillation
suit la pente 2, visiblement plus raide : elle s'écrase beaucoup plus vite que tout le reste.

> **Lire un graphe log-log :** un rapport constant entre deux quantités s'y voit comme un
> **écart vertical constant** entre deux droites parallèles. L'écart entre la courbe de η et
> celle de l'erreur, c'est le facteur ≈ 5.7. Si les droites n'étaient pas parallèles, cet
> écart se creuserait et l'estimateur deviendrait de plus en plus faux.

**(b) L'indice d'efficacité.** C'est le panneau (a) vu autrement, et c'est le cœur de
l'étude. `I_eff` monte légèrement (5.30 → 5.67) puis se tasse : les incréments successifs
(0.187, 0.097, 0.047, 0.023, 0.011) sont divisés par 2 à chaque raffinement. La courbe
s'approche donc d'une limite, que la droite en pointillés situe vers 5.68 (extrapolation de
Richardson, §8.5). La bande grisée montre l'intervalle `[5.30, 5.67]` : **quelle que soit la
finesse du maillage, `I_eff` ne sort pas de cette fenêtre.**

*D'où vient la légère montée ?* Le résidu η₁ décroît exactement en `O(h)` (ordre 1.0000),
tandis que les sauts η₂ n'atteignent leur régime asymptotique que progressivement (ordres
0.79 → 0.99). Sur les maillages grossiers, η₂ est donc un peu « en avance », et le rapport
se stabilise à mesure que η₂ rejoint la pente 1.

En pratique, η surestime l'erreur d'un facteur ≈ 5.7, donc `η/5.7` en est une bonne
approximation — mais cette constante n'est connue ici que parce qu'on connaît `u`. Sur un
vrai problème on ne l'a pas, et c'est pourquoi on se sert de η pour **comparer** les éléments
entre eux plutôt que comme valeur absolue.

*Remarque :* la limite est proche de `4√2 ≈ 5.657`, et il est tentant d'y voir une valeur
exacte. Les calculs jusqu'à `N = 128` montrent que ce n'est pas le cas : `I_eff(128) = 5.668`
dépasse déjà `4√2`, et la suite continue de croître. La constante dépend du problème, du
maillage et de la géométrie des triangles ; la théorie garantit seulement qu'elle est bornée.

**(c) La carte des η_K.** Sur le maillage le plus fin, les valeurs vont de 3.1e-5 à 1.0e-4,
soit un facteur 3 à 4 seulement : sur cette solution régulière et ce maillage uniforme, tous
les triangles se valent à peu près, **il n'y a rien à raffiner**. C'est le message négatif du
cas test, et il est instructif : un estimateur ne sert à rien si le problème n'a pas de zone
difficile.

Le détail du motif vient de la compétition entre les deux termes (maillage `N = 128`) :

| | maximum | où | minimum | où |
|---|---|---|---|---|
| η₁K (résidu) | 6.1e-5 | centre | 6.7e-7 | coins |
| η₂K (sauts) | 1.0e-4 | coins | 7.5e-7 | milieux des côtés |

η₁ est maximal au centre parce que `f = 2[x(1-x)+y(1-y)]` y vaut 1 et s'annule aux coins.
η₂ fait l'inverse : il est maximal aux coins parce que la dérivée croisée
`u_xy = (2x-1)(2y-1)` y atteint son maximum, donc c'est là que `∇u_h` tourne le plus vite
d'un triangle à l'autre — les « plis » du dôme facetté y sont les plus marqués. Le bord
l'emporte sur la carte totale.

**(d) L'efficacité locale, et le piège.** Le rapport `η_K/|u-u_h|_{1,K}` va de 3.44 à 10.69.
Les deux lobes clairs sont sur la diagonale `y = x`. L'erreur locale vraie y est anormalement
petite alors que `η_K` ne descend pas autant : le rapport explose donc là où le
**dénominateur s'effondre**, pas là où l'estimateur se trompe.

> **En clair :** si un élève a 0,1 faute par page et qu'on l'estime à 1, on se « trompe
> d'un facteur 10 » — mais sur quelque chose de quasi nul. Le grand rapport vient de la
> petitesse de l'erreur, pas d'une défaillance de l'estimateur.

*Expérience de contrôle réalisée :* en retournant la diagonale du maillage (bas-droite →
haut-gauche au lieu de bas-gauche → haut-droite), les lobes basculent exactement sur l'autre
diagonale. C'est donc un **artefact d'alignement du maillage** avec le signe de `u_xy`, une
forme de superconvergence locale : quand les arêtes sont orientées « dans le bon sens » par
rapport à la courbure de la solution, les facettes l'approchent anormalement bien. Sur un
maillage non structuré, le motif disparaîtrait.

**Ce n'est pas une contradiction avec le théorème.** L'efficacité locale du cours n'est jamais
élémentaire : elle s'écrit `η_K ≤ C·(|u-u_h|_{1,ω_K} + osc)`, sur le **patch** `ω_K` des
voisins de K, pas sur K seul. Un rapport de 10 sur un triangle où l'erreur s'annule presque
est parfaitement compatible avec la borne sur le patch. C'est aussi pourquoi, en pratique, on
marque les éléments par `η_K` relatif plutôt que de faire confiance à une valeur absolue
élément par élément.

**(e) La moyenne f_K.** Une copie « en escalier » de `f` : maximale au centre (≈ 1), quasi
nulle aux coins. Sur un maillage fin, les marches sont invisibles et la carte ressemble à `f`
elle-même — ce qui est précisément la raison pour laquelle l'oscillation est petite.

**(f) L'oscillation locale osc_K.** Elle est de l'ordre de 10⁻⁷ par triangle, contre 10⁻⁴
pour η_K : **trois ordres de grandeur en dessous**. Elle est minimale au centre, et c'est
logique : `f` y atteint son maximum, donc son gradient s'y annule — `f` est « plate » à cet
endroit, et la remplacer par sa moyenne n'y coûte presque rien. Elle est maximale aux coins,
où `f` varie le plus vite. On retrouve exactement l'estimation `‖f - f_K‖_K ≈ h·|∇f|·|K|^{1/2}`.

### 10.2 Domaine en L : l'estimateur au travail

Fichier : [`code/adaptation_Lshape.edp`](code/adaptation_Lshape.edp)

Domaine `Ω = ]-1,1[² \ ([0,1]×[-1,0])`, coin rentrant d'angle `3π/2` à l'origine, solution
exacte harmonique

```
u = r^(2/3) · sin(2θ/3)      θ ∈ [0, 3π/2],   f = 0
```

Cette fonction n'appartient qu'à `H^(5/3-ε)` : son gradient est infini à l'origine. La théorie
prédit que le raffinement uniforme perd l'ordre optimal.

> **En clair :** imagine une nappe tendue sur une table en forme de L. Au coin rentrant, la
> nappe fait un pli brusque. Pour bien représenter ce pli avec des facettes plates, il faut
> beaucoup de petites facettes **au coin** — et presque aucune ailleurs. Raffiner partout
> uniformément, c'est gaspiller des facettes là où la nappe est déjà bien représentée.

**Raffinement uniforme :**

| ndof | \|u-u_h\|₁ | η | I_eff | taux |
|---|---|---|---|---|
| 72 | 0.17729 | 0.53315 | 3.007 | — |
| 250 | 0.11679 | 0.35593 | 3.048 | 0.335 |
| 920 | 0.07616 | 0.24716 | 3.245 | 0.328 |
| 3583 | 0.04777 | 0.15219 | 3.186 | 0.343 |
| 14082 | 0.03017 | 0.09764 | 3.236 | 0.336 |

**Raffinement adaptatif piloté par η_K :**

| ndof | \|u-u_h\|₁ | η | I_eff | taux |
|---|---|---|---|---|
| 72 | 0.17729 | 0.53315 | 3.007 | — |
| 143 | 0.07681 | 0.26639 | 3.468 | 0.642 |
| 304 | 0.04854 | 0.17492 | 3.604 | 0.595 |
| 634 | 0.03297 | 0.11928 | 3.618 | 0.581 |

*(taux défini par `|u-u_h|₁ ~ ndof^(-taux)`)*

**Lecture.** L'uniforme plafonne à `ndof^(-1/3)`, l'adaptatif retrouve `ndof^(-1/2)`, l'ordre
optimal de P1. Concrètement : **634 degrés de liberté adaptatifs valent 14 082 degrés de
liberté uniformes**, soit 22 fois moins d'inconnues pour la même précision. Sur un calcul 3D
à plusieurs heures de temps machine, c'est la différence entre faisable et infaisable.

Et surtout, `I_eff` reste stable autour de 3.2–3.6 dans les deux régimes : **l'estimateur a
détecté la singularité et piloté le raffinement sans jamais recevoir la solution exacte**.
C'est exactement l'usage visé en situation réelle.

### 10.3 La stratégie de marquage

```cpp
real etacible = 0.5 * eta / sqrt(nt);                       // cible d'équidistribution
Ph hnew = hK * min(1.0, max(0.4, sqrt(etacible / etaK)));   // nouvelle carte de tailles
Th = adaptmesh(Th, metrique, IsMetric=1, nbvx=200000);
```

L'idée est d'**équidistribuer** : on veut `η_K` identique sur tous les triangles, car un
maillage optimal est un maillage où aucun élément ne domine l'erreur.

> **En clair :** c'est le principe d'une équipe bien organisée. Si une personne croule sous
> le travail pendant que les autres attendent, on redistribue. Un maillage optimal est un
> maillage où chaque triangle « porte » la même part de l'erreur.

Pour P1 on a localement `η_K ~ C·h_K²`, d'où la taille cible `h_K^new = h_K·√(η_cible/η_K)` :
un triangle dont l'indicateur est 4 fois trop grand voit sa taille divisée par 2.

Le `min(1.0, ...)` est un garde-fou essentiel : il interdit le déraffinement, donc le maillage
ne peut que croître. Sans lui, la boucle redistribue les éléments sans jamais faire décroître
l'erreur globale et stagne à nombre de ddl constant — c'est la première version de ce code qui
tournait ainsi et ne convergeait pas. Le `max(0.4, ...)` limite à l'inverse un raffinement
trop brutal en une seule itération.

---

## 11. Reproduire les résultats

### Dépendances

- FreeFem++ ≥ 4.9 — `sudo apt install freefem++` ou [freefem.org](https://freefem.org)
- Python 3 avec numpy, scipy et matplotlib — `pip install numpy scipy matplotlib`

### Exécution

```bash
git clone https://github.com/marwane-tchabanna/estimateur-a-posteriori-P1.git
cd estimateur-a-posteriori-P1/code

# 1. le solveur : écrit resultats.txt et maillage_etaK.txt
FreeFem++ -nw estimateur_residu_P1.edp

# 2. les figures
python3 plot_resultats.py            # -> figure_estimateur.png
python3 plot_resultats.py --show     # idem + fenêtre interactive

# 3. la démonstration sur domaine en L
FreeFem++ -nw adaptation_Lshape.edp
```

Les étapes 1 et 2 s'enchaînent en une ligne, Python ne se lançant que si FreeFem++ a réussi :

```bash
FreeFem++ -nw estimateur_residu_P1.edp && python3 plot_resultats.py
```

L'option `-nw` (*no window*) évite toute dépendance à un serveur graphique. Les fichiers de
sortie sont écrits dans le **répertoire courant** : lancer les deux commandes depuis le même
dossier.

### Validation croisée : version Python autonome

[`code/estimateur_residu_P1.py`](code/estimateur_residu_P1.py) réimplémente tout depuis zéro
en numpy/scipy — maillage, assemblage P1, résolution, estimateur, moyenne `f_K` et
oscillation — sans FreeFem++. Le calcul des sauts est entièrement vectorisé, ce qui permet
d'aller jusqu'à `N = 128` (32 768 triangles) en quelques secondes.

```bash
python3 estimateur_residu_P1.py
```

Le script affiche le même tableau que FreeFem++, les ordres de convergence de
`|u-u_h|₁`, η et osc, la limite extrapolée de `I_eff` et l'intervalle d'efficacité locale.

**Pourquoi c'est important.** Deux programmes écrits indépendamment, dans deux langages, avec
deux méthodes de quadrature différentes, donnent **les mêmes valeurs au cinquième chiffre
significatif** (même `I_eff = 5.3042, …, 5.6680`, même oscillation, même efficacité locale
`[3.44, 10.69]`). La probabilité que les deux contiennent exactement la même erreur est très
faible : c'est la meilleure garantie qu'aucun des deux ne se trompe silencieusement. Cela
confirme au passage que `square()` de FreeFem++ coupe les carrés selon la même diagonale
(bas-gauche → haut-droite) que le maillage Python.

---

## 12. Pièges rencontrés

Ces points ont réellement coûté du temps pendant le développement. Ils sont listés parce
qu'ils se reproduiront.

**1. Ne jamais nommer une variable `N` en FreeFem++.** `N` est la normale réservée (`N.x`,
`N.y`). Un `int N` la masque et le compilateur sort `No operator .x for type <Pl>`, message
parfaitement opaque. D'où le `Nx` dans le code.

**2. La quadrature n'est pas un détail.** `‖f‖²` est de degré 4, `|∇u-∇u_h|²` de degré 6. Une
quadrature insuffisante ne produit **aucun message d'erreur** : les nombres sortent, ils sont
juste approximatifs. Toujours fixer `qforder` explicitement quand une solution analytique
intervient, et valider par une implémentation indépendante (§11).

**3. Oublier `(nTonEdge-1)` — le piège le plus instructif du projet.** Une version du code
avait le bon commentaire (« `(nTonEdge-1)` annule le bord ») mais pas le facteur dans la
formule. Le code tournait, les nombres sortaient, et voici ce qu'ils disaient :

| N | η₂ (faux) | ordre de η | I_eff (faux) |
|---|---|---|---|
| 4 | 0.211 | — | 5.53 |
| 16 | 0.084 | 0.82 | 6.86 |
| 64 | 0.035 | 0.68 | 10.09 |
| 128 | 0.024 | 0.61 | **13.19** |

L'indice d'efficacité ne se stabilisait pas, et l'efficacité locale montait jusqu'à 253 sur
les triangles du bord.

*Le diagnostic, étape par étape :* η₁ gardait l'ordre 1, donc le problème venait de η₂, dont
l'ordre glissait vers **1/2**. Explication : sur une arête de bord, `jump(∂u_h/∂n)` vaut
simplement `∂u_h/∂n`, qui ne tend **pas** vers 0 quand on raffine. Chacune des ≈ `4N` arêtes
de bord apporte `h·|∂u_h/∂n|²·h`, soit au total `4N·h² ~ h`, donc une contribution en `h^{1/2}`
à η₂ qui finit par dominer le vrai saut intérieur en `O(h)`. Et `I_eff ~ h^{1/2}/h = h^{-1/2}`
explose.

*La leçon :* **un estimateur qui ne converge pas au bon ordre se diagnostique terme par
terme.** Regarder l'ordre de chaque composante séparément localise l'erreur en une minute.

**4. `f_K` n'est ni une `func`, ni une interpolation.** Une `func` ne connaît pas le maillage,
elle ne peut pas contenir une moyenne par triangle. Et `Ph fK = f;` évalue `f` au barycentre,
ce qui diffère de la moyenne d'un terme en `O(h²)` — précisément l'ordre de l'oscillation
qu'on veut mesurer. Seule l'intégration `varf` contre `chiK`, divisée par l'aire, donne la
vraie moyenne.

**5. Une erreur de syntaxe se lit sur la ligne d'avant.** FreeFem++ signale l'erreur à
l'endroit où il **s'aperçoit** du problème, pas là où il se trouve. Un `;` oublié à la fin de
la ligne 21 produit `Error line number 23 … before token int`. Réflexe : regarder la fin de
la ligne qui précède (souvent un `;`, une `)` ou une `}` manquant).

**6. Ne pas tracer un nombre.** `real ratio = eta/err;` puis `plot(ratio, …)` ne compile
pas : `plot` attend une fonction éléments finis ou un maillage. C'est le symptôme d'une
confusion entre efficacité globale (un `real`) et efficacité locale (un `Ph`), voir §7.6.

**7. Changer le format d'export, c'est changer deux fichiers.** Ajouter la colonne `osc` côté
FreeFem++ sans mettre à jour `plot_resultats.py` donne `too many values to unpack`. Et
exporter certains champs au carré et d'autres non fausse silencieusement les rapports
calculés en Python (§7.7).

**8. Rien ne s'affiche.** `plot()` passe par `ffglut`, qui exige un serveur graphique : en SSH,
sous WSL ou avec `-nw`, la sortie graphique échoue silencieusement. Solutions : l'option
`ps="fichier.eps"` de `plot()`, qui fonctionne même avec `-nw`, ou l'export texte vers Python.
Pour lire une EPS FreeFem : `gs -dEPSCrop -sDEVICE=png16m -r150 -sOutputFile=f.png f.eps`,
ou Inkscape. Attention, la palette FreeFem est arc-en-ciel avec **rouge = petit, magenta =
grand** — l'inverse de l'intuition, cause n°1 des contresens.

**9. L'adaptation qui stagne.** Sans le `min(1.0, ...)` dans la métrique, la boucle
redistribue les éléments à nombre de ddl constant au lieu de raffiner. Symptôme : `ndof` qui
oscille entre 41 et 78 au lieu de croître.

**10. Une belle constante n'est pas une constante exacte.** Avec 5 maillages, `I_eff` valait
5.6569, soit `4√2` à 4 chiffres près, ce qui était tentant. Le 6ᵉ maillage (5.668) a montré
que la suite continuait de croître. Avant d'affirmer qu'une limite vaut une valeur
remarquable, extrapoler (Richardson) et vérifier qu'on ne l'a pas déjà dépassée.

---

## 13. Structure du dépôt

```
.
├── README.md                          ce rapport
├── LICENSE                            licence MIT
├── code/
│   ├── estimateur_residu_P1.edp       solveur + estimateur + f_K + osc (FreeFem++)
│   ├── adaptation_Lshape.edp          boucle adaptative sur domaine en L
│   ├── plot_resultats.py              post-traitement et figure à 6 panneaux (Python)
│   └── estimateur_residu_P1.py        réimplémentation autonome numpy (validation croisée)
├── figures/
│   ├── figure_estimateur.png          les 6 panneaux de validation
│   └── efficacite_locale_freefem.png  sortie FreeFem native, convertie depuis l'EPS
└── resultats/
    ├── resultats.txt                  tableau de convergence (8 colonnes)
    └── maillage_etaK_extrait.txt      extrait du maillage + 4 champs P0
```

---

## 14. Références

- R. Verfürth, *A Posteriori Error Estimation Techniques for Finite Element Methods*,
  Oxford University Press, 2013 — la référence sur les estimateurs par résidus et
  l'oscillation des données.
- M. Ainsworth, J. T. Oden, *A Posteriori Error Estimation in Finite Element Analysis*,
  Wiley, 2000.
- W. Dörfler, *A convergent adaptive algorithm for Poisson's equation*,
  SIAM J. Numer. Anal. 33 (1996) — la stratégie de marquage de référence.
- F. Hecht, *New development in FreeFem++*, J. Numer. Math. 20 (2012) —
  documentation : [doc.freefem.org](https://doc.freefem.org)

---

**Marwane Tchabanna** — [@marwane-tchabanna](https://github.com/marwane-tchabanna)

Projet réalisé dans le cadre du cours « Estimations a posteriori », M2 Modélisation et
Analyse Numérique, Université de Montpellier.
