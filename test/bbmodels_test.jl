# TODO: create a function that calls a solver
function solver_func(nlp::AbstractNLPModel, p::AbstractParameterSet)
  return GenericExecutionStats(nlp)
end

@testset "Testing multi-precision $(T)-BBModels" verbose = true for T in (Float32, Float64)
  param_set = R2ParameterSet()
  x0 = T.(values(param_set))
  nlp = BBModel(param_set, problems, solver_func, time_only, x0 = x0)
  @test eltype(nlp.meta.x0) == T
  @test eltype(nlp.meta.lvar) == T
  @test eltype(nlp.meta.uvar) == T
end

@testset "Testing multi-precision for constrained $(T)-BBModels" verbose = true for T in (
  Float32,
  Float64,
)
  param_set = R2ParameterSet()
  c = x -> [x[1]]
  con = zeros(T, 1)
  x0 = T.(values(param_set))
  nlp = BBModel(param_set, problems, solver_func, time_only, c, con, con, x0 = x0)
  @test eltype(nlp.meta.x0) == T
  @test eltype(nlp.meta.lvar) == T
  @test eltype(nlp.meta.uvar) == T
  @test eltype(nlp.meta.lcon) == T
  @test eltype(nlp.meta.ucon) == T
  @test eltype(cons(nlp, nlp.meta.x0)) == T
end

@testset "Testing multi-precision for (constrained) $(T)-BBModels with categorical parameters" verbose =
  true for T in (Float32, Float64)
  param_set = TestParameterSet()
  x0 = values(param_set)
  x = rand(T, 2)
  values_num!(param_set, x)
  @test x == T[0, 5]
  bbmeta = BBModelMeta(param_set)
  @test bbmeta.nvar == 3
  @test bbmeta.icat == [3]
  @test bbmeta.iint == [2]
  @test bbmeta.ifloat == [1]
  @test bbmeta.ibool == []

  nlp = BBModel(param_set, problems, solver_func, time_only, x0 = x)
  @test get_nvar(nlp) == 2
  @test eltype(get_x0(nlp)) == T
  @test get_lvar(nlp) == [0; 5]
  @test get_uvar(nlp) == [1000; 20]
  @test obj(nlp, x) ≥ 0
  @test obj_cat(nlp, x0) ≥ 0

  c = x -> [x[1]]
  con = zeros(T, 1)
  nlp = BBModel(param_set, problems, solver_func, time_only, c, con, con, x0 = x)
  @test get_nvar(nlp) == 2
  @test eltype(get_x0(nlp)) == T
  @test get_lvar(nlp) == [0; 5]
  @test get_uvar(nlp) == [1000; 20]
  @test eltype(nlp.meta.lcon) == T
  @test eltype(nlp.meta.ucon) == T
  @test eltype(cons(nlp, nlp.meta.x0)) == T
  @test obj(nlp, x) ≥ 0
  @test obj_cat(nlp, x0) ≥ 0
end

@testset "Testing multi-precision $T with a subset of parameters" verbose = true for T in (
  Float32,
  Float64,
)
  param_set = TestParameterSet()
  subset = (:submethod, :mem) # switched the order

  x0 = values(subset, param_set)
  @test x0 == [:cg, 5]
  x = rand(T, 1)
  values_num!(subset, param_set, x)
  @test x == T[5]

  bbmeta = BBModelMeta(param_set, subset)
  @test bbmeta.nvar == 2
  @test bbmeta.icat == [1]
  @test bbmeta.iint == [2]
  @test bbmeta.ifloat == []
  @test bbmeta.ibool == []

  nlp = BBModel(param_set, problems, solver_func, time_only, subset = subset, x0 = x)
  @test get_nvar(nlp) == 1
  @test eltype(get_x0(nlp)) == T
  @test get_lvar(nlp) == [5]
  @test get_uvar(nlp) == [20]
  @test obj(nlp, x) ≥ 0
  @test obj_cat(nlp, x0) ≥ 0

  c = x -> [x[1]]
  con = zeros(T, 1)
  nlp = BBModel(param_set, problems, solver_func, time_only, c, con, con, subset = subset, x0 = x)
  @test get_nvar(nlp) == 1
  @test eltype(get_x0(nlp)) == T
  @test get_lvar(nlp) == [5]
  @test get_uvar(nlp) == [20]
  @test eltype(nlp.meta.lcon) == T
  @test eltype(nlp.meta.ucon) == T
  @test eltype(cons(nlp, nlp.meta.x0)) == T
  @test obj(nlp, x) ≥ 0
  @test obj_cat(nlp, x0) ≥ 0
end

function tailored_aux_func(vec_metric::Vector{ProblemMetrics})
  total = 0.0
  for p_metric in vec_metric
    median_time = median(get_times(p_metric))
    memory = get_memory(p_metric)
    solved = !is_failure(get_status(p_metric))
    counters = get_counters(p_metric)
    total += median_time + memory + counters.neval_obj + (Float64(!solved) * 5.0 * median_time)
  end
  return total
end

@testset "Testing BBModels with cost functions $(string(aux_func))" for aux_func in (
  time_only,
  memory_only,
  sumfc,
  tailored_aux_func,
)
  T = Float64
  I = Int64
  param_set = R2ParameterSet()
  nlp = BBModel(param_set, problems, solver_func, aux_func)

  @testset "Test BBModels attributes" begin
    x = nlp.meta.x0
    x_n = nlp.bb_meta.x_n
    lvar = nlp.meta.lvar
    uvar = nlp.meta.uvar
    icat = nlp.bb_meta.icat
    ibool = nlp.bb_meta.ibool
    iint = nlp.bb_meta.iint
    ifloat = nlp.bb_meta.ifloat
    @test x == values(nlp.parameter_set)
    @test x_n == [string(i) for i in fieldnames(typeof(nlp.parameter_set))]
    @test lvar == lower_bounds(nlp.parameter_set)
    @test uvar == upper_bounds(nlp.parameter_set)
    @test icat == Int[]
    @test ibool == Int[10]
    @test iint == Int[9]
    @test ifloat == Int[i for i = 1:8]
  end

  @testset "Test `obj` method with BBModel" begin
    @test BBModels.obj(nlp, nlp.meta.x0) ≥ 0.0
  end
end
