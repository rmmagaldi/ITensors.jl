function _precompile_()
  ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
  Base.precompile(Tuple{typeof(*),ITensor{2},ITensor{2}})   # time: 0.6906999
  Base.precompile(Tuple{typeof(compute_contraction_labels),IndexSet{2, Index{Int64}, Tuple{Index{Int64}, Index{Int64}}},IndexSet{2, Index{Int64}, Tuple{Index{Int64}, Index{Int64}}}})   # time: 0.006884135
  Base.precompile(Tuple{typeof(nblocks),NTuple{4, Index{Int64}}})   # time: 0.002266951
  Base.precompile(Tuple{typeof(dim),NTuple{4, Index{Int64}}})   # time: 0.002029559
  Base.precompile(Tuple{typeof(itensor),DenseTensor{Float64, 2, Tuple{Index{Int64}, Index{Int64}}, Dense{Float64, Vector{Float64}}}})   # time: 0.001160539
end

