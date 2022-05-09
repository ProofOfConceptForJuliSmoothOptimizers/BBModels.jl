# export BlackBoxException

# struct BlackBoxException <: Exception
#     msg::String
# end
function is_failure(stats::AbstractExecutionStats)
  failure_status = [:exception, :infeasible, :max_eval, :max_iter, :max_time, :stalled, :neg_pred]
  return any(s -> s == stats.status, failure_status)
end

function normalize_times!(times::Vector{Float64})
  times ./= get_cpu_frequency()
end

function get_cpu_frequency()
  return first(Sys.cpu_info()).speed / 1.0e3
end
