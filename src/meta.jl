export BBModelMeta

"""
    BBModelMeta

A composite type that represents the main features of the blackbox optimization problem of parameters.

---
    BBModelMeta(parameter_set::P [, subset::NTuple{N, Symbol} = fieldnames(P)])

Create an `BBModelMeta` of all parameters or a `subset` on a parameter_set of type `P <: AbstractParameterSet`.

`BBModelMeta` contains the following attributes:
- `nvar`: number of variables;
- `x_n::Vector{String}`: names of the parameters;
- `icat::Vector{Int}`: indices of categorical variables;
- `ibool::Vector{Int}`: indices of boolean variables;
- `iint::Vector{Int}`: indices of integer variables;
- `ifloat::Vector{Int}`: indices of real variables.

"""
struct BBModelMeta
  nvar::Int
  x_n::Vector{String}

  icat::Vector{Int}
  ibool::Vector{Int}
  iint::Vector{Int}
  ifloat::Vector{Int}

  function BBModelMeta(
    parameter_set::P,
    subset::NTuple{N, Symbol} = fieldnames(P),
  ) where {P <: AbstractParameterSet, N}
    x_n = Vector{String}(undef, N)
    for (i, field) in enumerate(subset)
      p = getfield(parameter_set, field)
      x_n[i] = name(p)
    end
    icat, ibool, iint, ifloat = types_indices(parameter_set, subset)
    new(N, x_n, icat, ibool, iint, ifloat)
  end
end

"""
    types_indices(parameter_set::P [, subset::NTuple{N, Symbol} = fieldnames(P)])

Return the set of indices of categorical, boolean, integer and real parameters within the `subset` of fields in `P <: AbstractParameterSet`.
"""
function types_indices(
  parameter_set::P,
  subset::NTuple{N, Symbol} = fieldnames(P),
) where {P <: AbstractParameterSet, N}
  icat, ibool, iint, ifloat = Int[], Int[], Int[], Int[]

  for (i, field) in enumerate(subset)
    p = getfield(parameter_set, field)
    if typeof(p.domain) <: CategoricalDomain
      push!(icat, i)
    elseif typeof(p.domain) <: BinaryRange
      push!(ibool, i)
    elseif typeof(p.domain) <: IntegerDomain
      push!(iint, i)
    elseif typeof(p.domain) <: RealDomain
      push!(ifloat, i)
    else
      throw(error("Unknown typing in parameter structure: $(typeof(p.domain))"))
    end
  end
  return icat, ibool, iint, ifloat
end
