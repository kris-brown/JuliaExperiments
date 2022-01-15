using TermInterface
using Catlab
using Catlab.GAT

using Catlab.Theories
X, Y, Z, W = [ Ob(FreeCategory.Ob, sym) for sym in [:X, :Y, :Z, :W] ]
f, g, h = Hom(:f, X, Y), Hom(:g, Y, Z), Hom(:h, Z, W)
fg = compose(f,g)

#abstract type GATExpr{T} end

istree(x::Type{GATExpr}) = true
exprhead(e::GATExpr) = head(e)

operation(e::GATExpr) = head(e)
arguments(e::GATExpr) =  e.args


function similarterm(x::Type{GATExpr}, head, args, symtype=nothing; metadata=nothing, exprhead=:call)
    expr_similarterm(head, args, Val{exprhead}())
end
