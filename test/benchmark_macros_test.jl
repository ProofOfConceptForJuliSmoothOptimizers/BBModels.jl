@testset "Test modified benchmark macros" begin
  bmark, result = BBModels.@benchmark_with_result sin(π / 2)
  @test bmark !== nothing
  @test result == 1.0
end
