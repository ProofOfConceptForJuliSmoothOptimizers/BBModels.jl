export AbstractBBModelMeta, BBModelMeta

"""
AbstractBBModelMeta

Base type for metadata related to an black box optimization model.
"""
abstract type AbstractBBModelMeta{T, S} <: AbstractNLPModelMeta{T, S} end

"""
    BBModelMeta <: AbstractBBModelMeta

A composite type that represents the main features of the optimization problem

    optimize    obj(x)
    subject to  lvar ≤    x    ≤ uvar
                lcon ≤ cons(x) ≤ ucon

where `x`        is an `nvar`-dimensional vector,
      `obj`      is the real-valued objective function,
      `cons`     is the vector-valued constraint function,
      `optimize` is either "minimize" or "maximize".

Here, `lvar`, `uvar`, `lcon` and `ucon` are vectors.
Some of their components may be infinite to indicate that the corresponding bound or general constraint is not present.

---

    NLPModelMeta(nvar; kwargs...)

Create an `NLPModelMeta` with `nvar` variables.
The following keyword arguments are accepted:
- `x0`: initial guess
- `lvar`: vector of lower bounds
- `uvar`: vector of upper bounds
- `nlvb`: number of nonlinear variables in both objectives and constraints
- `nlvo`: number of nonlinear variables in objectives (includes nlvb)
- `nlvc`: number of nonlinear variables in constraints (includes nlvb)
- `ncon`: number of general constraints
- `lcon`: vector of constraint lower bounds
- `ucon`: vector of constraint upper bounds
- `lin`: indices of linear constraints
- `minimize`: true if optimize == minimize
- `islp`: true if the problem is a linear program
- `name`: problem name
"""

# TODO: Create a Type for constraint vector (lcon ucon) (change the Float 64 vector to a parametric type)
struct BBModelMeta{T, S} <: AbstractBBModelMeta{T, S}
  nvar::Int
  x0::S
  x_n::Vector{Symbol}
  lvar::S
  uvar::S

  ifix::Vector{Int}
  ilow::Vector{Int}
  iupp::Vector{Int}
  irng::Vector{Int}
  ifree::Vector{Int}
  iinf::Vector{Int}

  nlvb::Int
  nlvo::Int
  nlvc::Int

  ncon::Int
  lcon::Vector{Float64}
  ucon::Vector{Float64}

  jfix::Vector{Int}
  jlow::Vector{Int}
  jupp::Vector{Int}
  jrng::Vector{Int}
  jfree::Vector{Int}
  jinf::Vector{Int}

  nnzo::Int
  nnzj::Int
  lin_nnzj::Int
  nln_nnzj::Int
  nnzh::Int

  nlin::Int
  nnln::Int

  lin::Vector{Int}
  nln::Vector{Int}

  minimize::Bool
  islp::Bool
  name::String

  function BBModelMeta{T, S}(
    nvar::Int,
    x0::S;
    x_n::Vector{Symbol} = Symbol[Symbol("param_", i) for i = 1:nvar],
    lvar::S = eltype(S)[typemin(typeof(x0ᵢ)) for x0ᵢ in x0],
    uvar::S = eltype(S)[typemax(typeof(x0ᵢ)) for x0ᵢ in x0],
    nlvb = nvar,
    nlvo = nvar,
    nlvc = nvar,
    ncon = 0,
    lcon::Vector{Float64} = fill!(Vector{Float64}(undef, ncon), -Inf64),
    ucon::Vector{Float64} = fill!(Vector{Float64}(undef, ncon), Inf64),
    lin = Int[],
    minimize = true,
    islp = false,
    name = "Generic",
  ) where {T, S}
    if (nvar < 1) || (ncon < 0)
      error("Nonsensical dimensions")
    end

    @lencheck nvar x0 lvar uvar
    @lencheck ncon lcon ucon
    @rangecheck 1 ncon lin

    x_types = DataType[typeof(i) for i in x0]
    x_min = [typemin(t) for t in x_types]
    x_max = [typemax(t) for t in x_types]

    ifix = findall(lvar .== uvar)
    ilow = findall((lvar .> x_min) .& (uvar .== x_max))
    iupp = findall((lvar .== x_min) .& (uvar .< x_max))
    irng = findall((lvar .> x_min) .& (uvar .< x_max) .& (lvar .< uvar))
    ifree = findall((lvar .== x_min) .& (uvar .== x_max))
    iinf = findall(lvar .> uvar)

    # TODO: fix type here. Do not simply use Float64
    if ncon > 0
      jfix = findall(lcon .== ucon)
      jlow = findall((lcon .> -Inf) .& (ucon .== Inf))
      jupp = findall((lcon .== -Inf) .& (ucon .< Inf))
      jrng = findall((lcon .> -Inf) .& (ucon .< Inf) .& (lcon .< ucon))
      jfree = findall((lcon .== -Inf) .& (ucon .== Inf))
      jinf = findall(lcon .> ucon)
    else
      jfix = Int[]
      jlow = Int[]
      jupp = Int[]
      jrng = Int[]
      jfree = Int[]
      jinf = Int[]
    end

    nln = setdiff(1:ncon, lin)
    nlin = length(lin)
    nnln = length(nln)

    new{T, S}(
      nvar,
      x0,
      x_n,
      lvar,
      uvar,
      ifix,
      ilow,
      iupp,
      irng,
      ifree,
      iinf,
      nlvb,
      nlvo,
      nlvc,
      ncon,
      lcon,
      ucon,
      jfix,
      jlow,
      jupp,
      jrng,
      jfree,
      jinf,
      0,
      0,
      0,
      0,
      0,
      nlin,
      nnln,
      lin,
      nln,
      minimize,
      islp,
      name,
    )
  end
end

BBModelMeta(nvar, x0::S; kwargs...) where {S} = BBModelMeta{eltype(S), S}(nvar, x0; kwargs...)
