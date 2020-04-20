module NDTensors

using Random
using LinearAlgebra
using StaticArrays
using HDF5
using CUDAdrv
using CuArrays 

const devs = Ref{Vector{CUDAdrv.CuDevice}}()
const dev_rows = Ref{Int}(0)
const dev_cols = Ref{Int}(0)
function __init__()
    voltas    = filter(dev->occursin("V100", CUDAdrv.name(dev)), collect(CUDAdrv.devices()))
    pascals    = filter(dev->occursin("P100", CUDAdrv.name(dev)), collect(CUDAdrv.devices()))
    devs[] = voltas[1:4]
    #devs[] = pascals[1:2]
    CUBLASMG.cublasMgDeviceSelect(CUBLASMG.mg_handle(), length(devs[]), devs[])
    dev_rows[] = 2
    dev_cols[] = 2
end
#####################################
# DenseTensor and DiagTensor
#
include("tupletools.jl")
include("dims.jl")
include("tensorstorage.jl")
include("tensor.jl")
include("contraction_logic.jl")
include("dense.jl")
include("symmetric.jl")
include("linearalgebra.jl")
include("diag.jl")
include("combiner.jl")
include("truncate.jl")
include("svd.jl")

#####################################
# BlockSparseTensor
#
include("blocksparse/blockdims.jl")
include("blocksparse/blockoffsets.jl")
include("blocksparse/blocksparse.jl")
include("blocksparse/blocksparsetensor.jl")
include("blocksparse/diagblocksparse.jl")
include("blocksparse/combiner.jl")
include("blocksparse/linearalgebra.jl")

end # module NDTensors
