include("DILS.jl")

"""
Construct a fully connected neural network. A final layer with 1 neuron will
be added.

There is one Const(1) box that is shared by all other Boxes. Each Box has its
own copy of the activation function.

TODO there is an activation before the final result, which should be removed.
"""
function nn(nᵢ::Int, layers::Vector{Int}, act::Primitive)::DILS
  bs, conn = DS[], Vector{Int}[]
  prev_layer = Int[]
  for (_, layer) in enumerate(vcat(layers,[1]))
    new_internals = vcat(repeat_dc(nᵢ=> 1, length(prev_layer)), [1=>1,0=>1])
    newbox = Box(nᵢ, 1, new_internals)
    newboxes = repeat_dc(newbox, layer)
    append!(bs, newboxes)
    append!(conn, repeat_dc(prev_layer, layer))
    prev_layer = collect(1+length(bs) - layer : length(bs))
  end
  const1_index = 2*length(bs) + 1 # Add const1 to each box
  for c in conn
    push!(bs, act)
    append!(c, [length(bs), const1_index])
  end
  push!(bs, Const1)
  append!(conn, repeat_dc(Int[], length(conn)+1))
  return DILS(bs, conn)
end

"""
Given a NN wiring pattern, return indices that should be fixed at 0 and to 1
for each Box.
"""
function fixed_wires(s::DILSState)::Vector{VI3_2}

  """
  Use the shapes of the weight matrices (and knowledge of how they were
  constructed) in order to write out which elements should be frozen at 0 and 1

  Construct matrices with either 0, 1, or 0.5 (for values that will be learned)
  and use `findall` to collect the respective indices.
  """
  function fixed_wires(s::BoxState)::VI3_2
  (n_input, in_cols), (wr, wc) = size.([s.in_weights, s.weights])
  n_nested = Int((in_cols-1)/n_input) # Number of neurons in previou layer
  getI12 = mat -> (x->findall(==(x), mat)).([0,1])

  in_matrix = ((n_nested == 0) ? ([0.5]')
              : (hcat(repeat(I(n_input), 1, n_nested), zeros(n_input,1))))
  w_matrix = vcat(zeros(wr-1, wc), hcat(ones(1,wc-2)*.5, [0 0.5]))
  out_matrix = vcat(zeros(n_nested, 1), [1,0])

  (i0, i1),(w0, w1), (o0, o1) = getI12.([in_matrix, w_matrix, out_matrix])
  return (i0, w0, o0) => (i1, w1, o1)
  end

  fixed_wires(s::PrimState)::VI3_2 = VI0 => VI0

  return fixed_wires.(s.states)
end

function run_nn!(layers::Vector{Int}, input::Vector{Float64}, output::Float64;
         α::Float64=1e-1, n::Int=5000, σ::Float64=0.2, verbose::Bool
         )::Tuple{DILSState,Vector}
  d = nn(length(input), layers, Tanh)
  s = DILSState(d, 0.05)
  freeze = fixed_wires(s)
  stream!(d, input, output; freeze=freeze, init_state=s, n=n, α=α, σ=σ, verbose=verbose)
end
