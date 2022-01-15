ENV["SINGULAR_PRINT_BANNER"] = "false"
import Singular as S
import Polynomials as P
import LinearAlgebra: I
import AbstractAlgebra as AA


"""
If we want to see how an observable responds to some set of vars, then we don't
want to have to do an ODE simulation for SS at each point in the varied space.
Rather, we want to get a parameterized steady state and simply evaluate at
each point.
"""


ORD = :lex # lex

VARS = [
    "A","B","C", "X", "Y",
    "kxy","kyx","kya","kay",
    "kax","kxa","kab","kba",
    "kac","kca", "kbc","kcb",
    "oY1", "oC1"]

function get_grobner(Ak::Matrix, Ψ::Matrix, Y::Matrix, Cnst::Vector)
  C = reshape(Cnst, 1, length(Cnst))
  dΨ = Ψ * Ak * Y + C;
  Ideal = S.Ideal(R, dΨ...)
  G = S.std(Ideal;complete_reduction=true)
  n = S.ngens(G)
  return [s_to_aa(G[i]) for i in 1:n]
end


"""
Convert singular to abstract algebra
"""
function s_to_aa(s::S.Singular.spoly)::AA.Generic.MPoly
  ts = [prod(VS_ .^ ev) for ev in S.exponent_vectors(s)]
  cs = [S.convert(Rational, x) for x in S.coefficients(s)]
  lt = length(ts)
  only(reshape(ts, 1, lt) * reshape(cs, lt, 1))
end

function get_vars(p::AA.Generic.MPoly)::Vector{String}
  xs = union([findall(!=(0), ev) for ev in AA.exponent_vectors(p)]...)
  return [VARS[v] for v in xs]
end

"""
Separate variables which appear in NO constraints, and order the rest by the
order in which they appear for the first time in the constraints.

[a,b], ([f], [c,d], [e]) means we could say that a/b/f are unconstrained, c(f),
d(c,f), and e(c,d,f).

If there are multiple vars in a given level, it means we can treat one of them
as dependent on the others and the others as dependent on everything earlier.
"""
function var_hierarchy(G::Vector{AA.Generic.MPoly{T}}
    )::Pair{Vector{String}, Vector{Vector{String}}} where {T}
  all_vars = deepcopy(VARS)
  res = Vector{String}[]
  for base in G
    vs = get_vars(base)
    new_vs = intersect(vs, all_vars)
    push!(res, new_vs)
    setdiff!(all_vars, new_vs)
  end
  return all_vars => res
end

function eval_p(p::AA.Generic.MPoly{T}, d::Dict{Symbol, S.spoly{T_}}
      )::AA.Generic.MPoly{T} where {T,T_}
    vals = [s_to_aa(get(d, Symbol(v), dflt)) for (v,dflt) in zip(VARS, VS)]
    return p(vals...)
end
function eval_ps(ps::Vector{AA.Generic.MPoly{T}}, d::Dict{Symbol, S.spoly{T_}}
  )::Vector{AA.Generic.MPoly{T}} where {T,T_}
 [eval_p(p, d) for p in ps]
end

"""
   <- Y
    /  ╲
   X <-> A <-> B
           ╲ /
            C ->

1->Y

  X  [1]   [3]
      1 0  1 0
        [2] -> 1

"""

R, VS = S.PolynomialRing(S.QQ, VARS, ordering=ORD)
#(Singular.AsEquivalentAbstractAlgebraPolynomialRing is broken)
R_, VS_ = AA.PolynomialRing(AA.QQ, VARS, ordering=:lex)

(A,B,C,X,Y,
 kxy,kyx,kya,kay,
 kax,kxa,kab,kba,
 kac,kca,kbc,kcb,
 oY1, oC1) = VS

Cnst = [0, 0, -oC1, 0, -oY1]
     # d[A]=0             d[B]=0     d[C]=0      d[X]=0       d[Y]=0
Ak = [-(kay+kax+kab+kac) kab          kac         kax          kay;
        kba              -(kba+kbc)   kbc         0            0;
        kca              kcb          -(kca+kcb)  0            0;
        kxa              0            0           -(kxy+kxa)   kxy;
        kya              0            0           kyx          -(kyx+kya);];
Y_ = Matrix(I(5))
Ψ = [A  B  C  X  Y];
G = get_grobner(Ak, Ψ, Y_, Cnst)
unused, vh = var_hierarchy(G)

ks = Symbol.(VARS[6:end-2])
kconst = Dict([k=>S.one(R) for k in ks])
G_ = eval_ps(G, kconst)




struct SemiAlgRelOb
  n::Int
end

Monomial = Vector{Int}  # [1,0,4] = x₁⋅(x₃)⁴
struct Poly
  coefs::Vector{Rational{Int}}
  exp_vector::Vector{Monomial} # implicitly a sum
end
struct SemiAlgRelMorphism
  dom::SemiAlgRelOb
  cod::SemiAlgRelOb
  # one element for each n in codomain
  # indices refer to n in domain
  rel::Vector{Poly}
  # (in theory, there are inequality constraints, too)
end


"""
From "The Smallest Multistationary Mass-Preserving Chemical Reaction Network"
(1) A³  ⇄ AB² (2)
    ↕      ↕
(4) A²B ⇄  B³ (3)
"""
#k12,k14,k21,k23,k32,k34,k41,k43 = 1//4,1//2,1,13,1,2,8,1
# R, (A,B,k12,k14,k21,k23,k32,k34,k41,k43) = S.PolynomialRing(S.QQ, ["A","B","k12","k14","k21","k23","k32","k34","k41","k43"])
# Ak = [-k12-k14 k12 0 k14;
#       k21 -k21-k23 k23 0;
#       0 k32 -k32-k34 k34;
#       k41 0 k43 -k41-k43];
# Y = [3 0; 1 2; 0 3; 2 1];
# Ψ = [A*A*A  A*B*B  B*B*B  A*A*B];
# dΨ = Ψ * (Ak * Y);





# I = S.Ideal(R, dΨ...)
# G = S.std(I;complete_reduction=true)

# p = P.Polynomial([-5,11,-6,1])
#P.roots(p)


# to_poly(s::spoly)::Poly = Poly(exponent_vectors(s), collect(coefficients(s))
# to_morph(G::sideal, n_codom::Int)::SemiAlgRelMorphism = SemiAlgRelMorphism(
#     SemiAlgRelOb(length(G.base_ring.S)),
#     SemiAlgRelOb(n_codom),
#     [toPoly(G[i]) for i in 1:ngens(G)])













if false
R, (x, y) = PolynomialRing(QQ, ["x", "y"])
I = Ideal(R, x^2 + 1, x*y + 1)
G = std(I)
G_ = std(I;complete_reduction=true)

"""
EXAMPLE

An open reaction network expressed as the composition of
two subnetworks

X = {1,2,3}, Y={4}, Z = {5,6}

|X|               |Y|                |Z|
|1|⟶ A ↘         | |          ↗ E ⟵|5|
|2| ↘    [α]⇉ C ⟵|4|⟶ D ⟶ [β]     | |
|3|⟶ B ↗         | |          ↘ F ⟵|6|


Grey boxing Rxn1 to dynam:
d[A] = -α[A][B] + I₁
d[B] = -α[A][B] + I₂ + I₃
d[C] = 2α[A][B] - I₄

Grey boxing Rxn2 to dynam
d[D] = I₄ - β[D]
d[E] = β[D] - I₅
d[F] = β[D] - I₆

Composition in dynam (add AND assert C=D)
d[A] = -α[A][B] + I₁
d[B] = -α[A][B] + I₂ + I₃
d[C] = 2α[A][B] - β[C]
d[E] = β[C] - I₅
d[F] = β[C] - I₆

Let Iₙ=n, α=2, β=5
"""
# I₁,I₂,I₃,I₅,I₆ = 1,2,3,5,6
α, β= 2, 5
R, (I₁,I₂,I₃,I₅,I₆, A, B, C, E, F) = PolynomialRing(
    QQ, ["I₁","I₂","I₃","I₅","I₆", "A", "B", "C", "E", "F"])
I_overall = Ideal(R, -α*A*B + I₁,
              -α*A*B + I₂ + I₃,
              2*α*A*B-β*C,
              β*C-I₅,
              β*C-I₆)
G_overall = std(I_overall;complete_reduction=true)

R, (I₁,I₂,I₃,I₄, A, B, C,) = PolynomialRing(
    QQ, ["I₁","I₂","I₃","I₄", "A", "B", "C"])
I_1 = Ideal(R, -α*A*B + I₁,
              -α*A*B + I₂ + I₃,
              2*α*A*B-I₄)
G_1 = std(I_1;complete_reduction=true)


R, (I₄, I₅, I₆, C, E, F) = PolynomialRing(
    QQ, ["I₄", "I₅", "I₆", "C", "E", "F"])
I_2 = Ideal(R, I₄-β*C,β*C-I₅, β*C-I₆)
G_2 = std(I_2;complete_reduction=true)
end