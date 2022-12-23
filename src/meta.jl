export BBModelMeta

"""BBModelMeta:
Struct containing the necessary information of black box optimization problem.
Similar to `NLPModels.NLPModelMeta`, this struct keeps track of the types of variables (i.e, continuous, discrete and binary).
"""
struct BBModelMeta
  nvar::Int
  x_n::Vector{String} # do we need this?
  iint::Vector{Int}
  ifloat::Vector{Int}
  ibool::Vector{Int}

  function BBModelMeta(nvar::Int, x0::AbstractVector, x_n::Vector{String})
    if (nvar < 1)
      error("Nonsensical dimensions")
    end

    @lencheck nvar x0 x_n

    # debug this...
    iint = findall(x -> x isa Int, x0)
    ifloat = findall(x -> x isa AbstractFloat, x0)
    ibool = findall(x -> x isa Bool, x0)

    new(nvar, x_n, iint, ifloat, ibool)
  end
end
