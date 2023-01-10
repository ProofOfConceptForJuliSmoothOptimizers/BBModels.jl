@testset "Test Problems struct functions" for pbs in (problems, problems_expr)
  p1 = Problem(1, pbs[1])

  nlp = get_nlp(p1)
  @test nlp.meta.name === problems[1].meta.name
  @test get_id(p1) == 1

  bmark, stats = BBModels.@benchmark_with_result lbfgs($nlp; mem = 15)
  BBModels.normalize_times!(bmark.times)
  p_metrics = ProblemMetrics(1, bmark.times, bmark.memory, stats.status, deepcopy(nlp.counters))

  @test get_pb_id(p_metrics) == 1
  @test get_times(p_metrics) === bmark.times
  @test get_memory(p_metrics) == bmark.memory
  @test is_failure(get_status(p_metrics)) isa Bool
  for field in fieldnames(Counters)
    @test getfield(get_counters(p_metrics), field) == getfield(nlp.counters, field)
  end
  @test get_counters(p_metrics) !== nlp.counters
  reset!(nlp)
  for field in fieldnames(Counters)
    @test getfield(get_counters(p_metrics), field) >= getfield(nlp.counters, field)
  end
end
