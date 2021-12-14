# using AlgebraicDynamics.DWDDynam: DiscreteMachine

using LinearAlgebra
using Plots: plot
using Random, Distributions
using LightGraphs
Random.seed!(123) # Setting the seed

# Type abbreviations/defaults
#----------------------------
PVV = Pair{Vector{Float64},Vector{Float64}}
Readouts = Vector{Union{Nothing,PVV}}
VPI = Vector{Pair{Int,Int}}
VPI3 = Tuple{VPI,VPI,VPI}
VPI0 = (VPI(),VPI(),VPI())
VPI3_2 = Pair{VPI3, VPI3}
MaybeVVPI32 = Union{Nothing,Vector{VPI3_2}}
VPI00 = VPI0 => VPI0

# General helper functions
#-------------------------
"""Topologically sort based on a list of children for each box"""
function topsort(conn::Vector{Vector{Int}})
  g = DiGraph(length(conn))
  for (k, vs) in enumerate(conn)
    for v in vs
      add_edge!(g, k, v)
    end
  end
  return topological_sort_by_dfs(g)
end

function repeat_dc(x::Any, n::Int)::Vector
  return [deepcopy(x) for _ in 1:n]
end

# Interfaces
#-----------
"""
Something that will eventually be a dynamical system.

Either a box or primitive
"""
abstract type DS end

"""
A box with no internal structure, just (stateless) dynamics and knowledge of
how to propagate derivative information
"""
struct Primitive <: DS
  Yᵢ::Int
  Yₒ::Int
  f::Function # ℝⁿⁱ ⟶ ℝⁿᵒ
  df::Function # ℝⁿⁱ  ⟶ ℝⁿⁱˣⁿᵒ (dyₒ/dxᵢ for each output yₒ & input xᵢ)
end

"""A morphism X₁⊗...⊗Xₙ ⟶ Y"""
struct Box <: DS
  Yᵢ::Int
  Yₒ::Int
  X::Vector{Pair{Int,Int}}  # Nested structure
end

isPrim(ds::Primitive) = true
isPrim(ds::Box) = false

"""
Indices to turn a flat vector into a list of vectors to feed to inner boxes,
for both the value input as well as the loss input.

E.g. if there are three internal boxes: 1/1 0/1 1/2 (inputs/outputs)
     then this should return [1:1=>1:1, ∅=>2:2, 2:2=>3:4]
                              (Box 1)  (Box 2)  (Box 3)
"""
function input_loss_inds(b::Box)::Vector{Pair{UnitRange, UnitRange}}
  res, curr_in, curr_loss = Pair{UnitRange, UnitRange}[], 1, 1
  for (n_in, n_loss) in b.X
    push!(res, curr_in:curr_in+n_in-1 => curr_loss:curr_loss+n_loss-1)
    curr_loss+=n_loss
    curr_in += n_in
  end
  return res
end


"""
A list of boxes and their nesting pattern.
Validates that it makes sense upon construction
"""
struct DILS
  boxes::Vector{DS}
  nesting::Vector{Vector{Int}} # list boxes that are nested inside of this
  function DILS(boxes::Vector{DS}, nesting::Vector{Vector{Int}})
    topsort(nesting) # check acyclicity
    for (b, vs) in zip(boxes, nesting)
      ((isPrim(b) && isempty(vs))
        || (b.X == (z -> z.Yᵢ => z.Yₒ).(boxes[vs]))
        || error("b $b vs $vs ($((b -> b.Yᵢ => b.Yₒ).(boxes[vs])))"))
    end
    return new(boxes,nesting)
  end
end


# STATES
#-------
abstract type State end

mutable struct PrimState <: State
  readout::Vector{Float64}
  condition::Vector{Float64}
  function PrimState(p::Primitive)
    return new(zeros(p.Yₒ), zeros(p.Yᵢ))
  end
end

mutable struct BoxState <: State
  in_weights::Matrix{Float64}   # nᵢ × Σᵢₙₜₑᵣₙₐₗ nᵢ
  weights::Matrix{Float64}      # Σᵢₙₜₑᵣₙₐₗ nᵢ × Σᵢₙₜₑᵣₙₐₗ nₒ
  out_weights::Matrix{Float64}  # Σᵢₙₜₑᵣₙₐₗ nₒ × nₒ
  function BoxState(b::Box, σ::Float64=0.2)
    Xi, Xo = sum.([first.(b.X),last.(b.X)])
    new([rand(Normal(0, σ), row, col) for (row, col) in
               [(b.Yᵢ, Xi), (Xi, Xo), (Xo, b.Yₒ)]]...)
  end
end

struct DILSState
  states::Vector{State}
  function DILSState(d::DILS, σ::Float64)
    return new([State(b, σ) for b in d.boxes])
  end
end

State(d::Box, σ::Float64)::State = BoxState(d, σ)
State(d::Primitive, σ::Float64)::State = PrimState(d)

Base.getindex(ds::DILSState, i) = ds.states[i]
Base.length(ds::DILSState) = length(ds.states)

"""Pretty print a matrix with rounding"""
function pp(a::Union{Vector,Matrix,Adjoint}, digits::Int=3)::String
  b=IOBuffer();
  show(b, "text/plain", round.(a; digits=digits))
  return String(take!(b))
end

function pp(a::BoxState)::String
  "IN $(pp(a.in_weights))\nW $(pp(a.weights))\nOUT $(pp(a.out_weights'))"
end


# READOUT
#########
"""Get readouts for all of the boxes at each level of the hierarchy"""
function readout(d::DILS, ds::DILSState)::Readouts
  res = Readouts(repeat([nothing], length(d.boxes)))
  for i in reverse(topsort(d.nesting))
    res[i] = readout_box(ds[i], res[d.nesting[i]])
  end
  return res
end

readout_box(p::PrimState, _::Readouts)::PVV = p.readout => p.condition
function readout_box(bs::BoxState, r::Readouts)::PVV
  int_vals, int_derivs = (x->vcat(x...)).(zip(r...))
  new_vals = bs.out_weights' * int_vals
  new_derivs = bs.in_weights * int_derivs
  return new_vals => new_derivs
end


# UPDATE
########

"""
Update all boxes at all levels of the hierarchy at once.
Return the readout computed as an intermediate step.
"""
function update(d::DILS, ds::DILSState,
                input::Vector{Float64}, loss::Vector{Float64};
                freeze::MaybeVVPI32=nothing)::Readouts
  rs = readout(d, ds)
  println("COMPUTING UPDATE WITH (value) READOUT $(first.(rs))")
  readins = Readouts([repeat_dc(0., b.Yᵢ) => repeat_dc(0., b.Yₒ)
                      for b in d.boxes])
  readins[topsort(d.nesting)[1]] = input=>loss
  for i in topsort(d.nesting)
    (inputᵢ, lossᵢ), freezeᵢ = readins[i], isnothing(freeze) ? VPI00 : freeze[i]
    #println("\tupdating box $i w/ input $inputᵢ and loss $lossᵢ")
    update_box(d.boxes[i], ds[i], rs[d.nesting[i]],
               readins[d.nesting[i]], inputᵢ, lossᵢ; freeze=freezeᵢ)
  end
  return rs
end

"""Update Primitive box by applying f and df"""
function update_box(prim::Primitive, pstate::PrimState, r1::Readouts,
                    r2::Readouts, input::Vector{Float64}, loss::Vector{Float64};
                    freeze::VPI3_2=VPI00)::Nothing
  nᵢnₒ = [prim.Yᵢ, prim.Yₒ]
  length.([input, loss]) == nᵢnₒ || error("Wrong # of inputs")
  pstate.readout = prim.f(input)
  pstate.condition = reshape(prim.df(input), nᵢnₒ...) * loss # chain rule
  length.([pstate.condition, pstate.readout]) == nᵢnₒ || error("Wrong # outputs")
  return nothing
end


"""Update Box by modifying weights. Also compute data to be fed into children"""
function update_box(b::Box, boxstate::BoxState, nested_outputs::Readouts,
                    nested_inputs::Readouts, input::Vector{Float64},
                    loss::Vector{Float64};  freeze::VPI3_2=VPI00)::Nothing
  length.([input, loss]) == [b.Yᵢ, b.Yₒ] || error("Wrong # of inputs")

  # Compute weight changes
  internal_vals, internal_derivs = (x->vcat(x...)).(zip(nested_outputs...))
  d_ow = internal_vals*loss'
  d_w = internal_derivs * internal_vals'
  d_iw = input * internal_derivs'

  # Update weights
  boxstate.out_weights .+= d_ow
  boxstate.weights .+= d_w
  boxstate.in_weights .+= d_iw

  # Reset frozen weights
  w0, w1 = freeze
  set_wires!(boxstate, 0., w0)
  set_wires!(boxstate, 1., w1)

  # Add a *contribution* to inputs of children for their updates
  i_internal, l_internal = get_child_update_data(
    boxstate, input, loss, internal_vals, internal_derivs)
  for ((i_inds, l_inds), (i_in, l_in)) in zip(input_loss_inds(b), nested_inputs)
    i_in .+= i_internal[i_inds]
    l_in .+= l_internal[l_inds]
  end
end

"""Get input/loss info to be fed to all nested subboxes"""
function get_child_update_data(
    state::BoxState, val_input::Vector{Float64}, deriv_input::Vector{Float64},
    internal_vals::Vector{Float64}, internal_derivs::Vector{Float64})::PVV
  # Get valdata
  external_val_contrib = (val_input' * state.in_weights)'
  internal_val_contrib = state.weights * internal_vals
  valdata = external_val_contrib .+ internal_val_contrib
  # Get derivdata
  external_deriv_contrib = state.out_weights * deriv_input
  internal_deriv_contrib = state.weights' * internal_derivs
  derivdata = external_deriv_contrib .+ internal_deriv_contrib
  return valdata => derivdata
end

# MANIPULATING STATE
####################
set_wires!(s::PrimState, val::Float64, init::VPI3; zero_::Bool=false) = nothing
function set_wires!(s::BoxState, val::Float64, init::VPI3; zero_::Bool=false
                    )::Nothing
  if zero_
    s.in_weights *= 0
    s.weights *= 0
    s.out_weights *= 0
  end
  for (i,j) in init[1]
    s.in_weights[i,j] = val
  end
  for (i,j) in init[2]
    s.weights[i,j] = val
  end
  for (i,j) in init[3]
    s.out_weights[i,j] = val
  end
end

"""
Continuously feed a constant value, train based on L2 loss

Input and output are fixed.

init_state - If none given, a default state is used (w/ variance `σ`)
α - learning rate. Constant multiplier for the loss function (if low, e.g. 1e-8,
    then learning is stable but slow)
n - Number of iterations to run
freeze - specify indices that should be fixed to 0 and indices fixed to 1
"""
function stream!(d::DILS, input::Vector{Float64}, output::Float64;
                 init_state::Union{Nothing,DILSState}=nothing,
                 α::Float64=1e-5, n::Int=5000, σ::Float64=0.2,
                 verbose::Bool=true, freeze::MaybeVVPI32=nothing
                 )::Tuple{DILSState,Vector}
  s = isnothing(init_state) ? DILSState(d, σ) : init_state
  if !isnothing(freeze)
    for (st, (f0, f1)) in zip(s.states, freeze)
      set_wires!(st, 0., f0)
      set_wires!(st, 1., f1)
    end
  end
  err, res = 0, Float64[]

  for _ in 1:n
    update_res= update(d, s, input, [err*α]; freeze=freeze)
    push!(res, update_res[1][1][1])
    err = output - res[end]
    if verbose
      println("***res $(res[end]) err $err")
    end
  end
  return s, res
end



