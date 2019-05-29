using ITensors

#@code_warntype ITensors.contract_inds(IndexSet(),Int[],IndexSet(),Int[])

#@code_warntype ITensors.storage_contract(Dense{Float64}(),IndexSet(),Dense{Float64}(),IndexSet())

#@code_warntype *(ITensor(Index()),ITensor(Index()))

#@code_warntype ITensors.contract(IndexSet(),Vector{Int}(),Dense{Float64}(),IndexSet(),Vector{Int}(),Dense{Float64}(),IndexSet(),Vector{Int}())

#@code_warntype dmrg(MPO(),MPS(),Sweeps(1))

@code_warntype davidson(ProjMPO(MPO()),ITensor())

#@code_warntype ITensors.contract!(Vector{Float64}(),ITensors.CProps(Vector{Int}(),Vector{Int}(),Vector{Int}()),Vector{Float64}(),Vector{Float64}(),zero(Float64),zero(Float64))

#@code_warntype ProjMPO(MPO())(ITensor())

#@code_warntype ITensors.makeL!(ProjMPO(MPO()),MPS(),0)

#@code_warntype ITensors.makeR!(ProjMPO(MPO()),MPS(),0)


