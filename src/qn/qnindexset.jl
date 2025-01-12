
const QNIndexSet = IndexSet{QNIndex}

const QNIndices = Union{QNIndexSet,Tuple{Vararg{QNIndex}}}

# Get a list of the non-zero blocks given a desired flux
# TODO: make a fillqns(inds::Indices) function that makes all indices
# in inds have the same qns. Then, use a faster comparison:
#   ==(flux(inds,block; assume_filled=true), qn; assume_filled=true)
function nzblocks(qn::QN, inds::Indices)
  N = length(inds)
  blocks = Block{N}[]
  for block in eachblock(inds)
    if flux(inds, block) == qn
      push!(blocks, block)
    end
  end
  return blocks
end

function nzdiagblocks(qn::QN, inds::Indices)
  N = length(inds)
  blocks = NTuple{N,Int}[]
  for block in eachdiagblock(inds)
    if flux(inds, block) == qn
      push!(blocks, Tuple(block))
    end
  end
  return blocks
end

removeqns(is::QNIndices) = map(i -> removeqns(i), is)

anyfermionic(is::Indices) = any(isfermionic, is)

allfermionic(is::Indices) = all(isfermionic, is)

# TODO: specialize for Vector vs. Tuple
function Base.show(io::IO, is::QNIndices)
  print(io, "$(length(is)) Indices\n")
  for i in is
    print(io, i)
    println(io)
  end
end
