struct R2ParameterSet{T <: AbstractFloat, I <: Integer} <: AbstractParameterSet
  atol::Parameter{T, RealInterval{T}}
  rtol::Parameter{T, RealInterval{T}}
  η1::Parameter{T, RealInterval{T}}
  η2::Parameter{T, RealInterval{T}}
  γ1::Parameter{T, RealInterval{T}}
  γ2::Parameter{T, RealInterval{T}}
  σmin::Parameter{T, RealInterval{T}}
  β::Parameter{T, RealInterval{T}}
  mem::Parameter{I, IntegerRange{I}}
  is_scaling::Parameter{Bool, BinaryRange{Bool}}

  function R2ParameterSet{T, I}(;
    atol::T = √eps(T),
    rtol::T = √eps(T),
    η1::T = T(eps(T)^(1 / 4)),
    η2::T = T(0.95),
    γ1::T = T(1 / 2),
    γ2::T = T(2),
    σmin::T = zero(T),
    β::T = zero(T),
    mem::I = I(5),
    is_scaling::Bool = true,
  ) where {T <: AbstractFloat, I <: Integer}
    (atol > 0) || error("invalid atol")
    (rtol > 0) || error("invalid rtol")
    (0 ≤ β < 1) || error("invalid: β needs to be between [0,1)")
    (γ1 ≤ γ2) || error("invalid γ1 <= γ2")
    (0 < η1 ≤ η2 ≤ 1) || error("invalid: 0 < η1 <= η2 <= 1")
    p_set = new{T, I}(
      Parameter(T(atol), RealInterval(T(0), T(1)), ""),
      Parameter(T(rtol), RealInterval(T(0), T(1)), ""),
      Parameter(T(η1), RealInterval(T(0), T(1)), ""),
      Parameter(T(η2), RealInterval(T(0), T(10)), ""),
      Parameter(T(γ1), RealInterval(T(0), T(1)), ""),
      Parameter(T(γ2), RealInterval(T(1), T(10)), ""),
      Parameter(σmin, RealInterval(T(0), T(1000)), ""),
      Parameter(T(β), RealInterval(T(0), T(1000)), ""),
      Parameter(I(mem), IntegerRange(I(5), I(20)), ""),
      Parameter(is_scaling, BinaryRange(), ""),
    )
    set_names!(p_set)
    return p_set
  end

  function R2ParameterSet(; kwargs...)
    R2ParameterSet{Float64, Int}(; kwargs...)
  end
end
