export BBModelMeta

"""BBModelMeta:
Struct containing the necessary information of black box optimization problem.
Similar to `NLPModels.NLPModelMeta`, this struct keeps track of the types of variables (i.e, continuous, discrete and binary).
"""
struct BBModelMeta
  nvar::Int
  x0::Vector{Float64}
  x_n::Vector{String}
  lvar::Vector{Float64}
  uvar::Vector{Float64}
  iint::Vector{Int}
  ifloat::Vector{Int}
  ibool::Vector{Int}

  ncon::Int
  lcon::Vector{Float64}
  ucon::Vector{Float64}

  function BBModelMeta(
    nvar::Int,
    x0::AbstractVector,
    x_n::Vector{String},
    lvar::AbstractVector,
    uvar::AbstractVector;
    ncon = 0,
    lcon::Vector{Float64} = fill!(Vector{Float64}(undef, ncon), -Inf64),
    ucon::Vector{Float64} = fill!(Vector{Float64}(undef, ncon), Inf64),
  )
    if (nvar < 1) || (ncon < 0)
      error("Nonsensical dimensions")
    end

    @lencheck nvar x0 lvar uvar
    @lencheck ncon lcon ucon

    S = Float64
    iint = findall(x -> x isa Int, x0)
    ifloat = findall(x -> x isa AbstractFloat, x0)
    ibool = findall(x -> x isa Bool, x0)
    x0 = Vector{S}([S(i) for i in x0])
    lvar = Vector{S}([S(i) for i in lvar])
    uvar = Vector{S}([S(i) for i in uvar])

    new(nvar, x0, x_n, lvar, uvar, iint, ifloat, ibool, ncon, lcon, ucon)
  end
end
