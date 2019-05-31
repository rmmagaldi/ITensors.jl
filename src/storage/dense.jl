
struct Dense{T} <: TensorStorage
  data::Vector{T}
  Dense{T}(data::Vector{T}) where {T} = new{T}(data)
  #Dense{T}(size::Integer) where {T} = new{T}(zeros(T,size))
  Dense{T}(size::Integer) where {T} = new{T}(Vector{T}(undef,size))
  Dense{T}(x::Number,size::Integer) where {T} = new{T}(fill(x,size))
  Dense{T}() where {T} = new{T}(Vector{T}())
end

data(D::Dense{Float64}) = D.data
length(D::Dense{Float64}) = length(data(D))
eltype(D::Dense{Float64}) = Float64
getindex(D::Dense{Float64},i::Int)::Float64 = data(D)[i]

storage_randn!(S::Dense{Float64}) = randn!(data(S))
storage_norm(S::Dense{Float64})::Float64 = norm(data(S))
storage_conj(S::Dense{Float64}) = Dense{Float64}(conj(data(S)))

#TODO: this should do proper promotions of the storage data
#e.g. ComplexF64*Dense{Float64} -> Dense{ComplexF64}
*(D::Dense{Float64},x::Float64) = Dense{Float64}(x*data(D))
*(x::Float64,D::Dense{Float64}) = D*x
/(D::Dense{Float64},x::Float64) = Dense{Float64}(data(D)/x)
-(D::Dense{Float64}) = Dense{Float64}(-data(D))

dot(D1::Dense{Float64},D2::Dense{Float64}) = dot(data(D1),data(D2))

function scale!(D::Dense{Float64},x::Float64)
  Ddata = data(D)
  rmul!(Ddata,x)
  return
end

copy(D::Dense{Float64}) = Dense{Float64}(copy(data(D)))

outer(D1::Dense{Float64},D2::Dense{Float64}) = Dense{Float64}(vec(data(D1)*transpose(data(D2))))

storage_convert(::Type{Array},D::Dense{Float64},is::IndexSet) = reshape(data(D),dims(is))

storage_fill!(D::Dense{Float64},x::Float64) = fill!(data(D),x)

function storage_getindex(Tstore::Dense{Float64},
                          Tis::IndexSet,
                          vals::Union{Int, AbstractVector{Int}}...)
  return getindex(reshape(data(Tstore),dims(Tis)),vals...)
end

function storage_setindex!(Tstore::Dense{Float64},
                           Tis::IndexSet,
                           x::Union{Float64, Array{Float64}},
                           vals::Union{Int, AbstractVector{Int}}...)
  return setindex!(reshape(data(Tstore),dims(Tis)),x,vals...)
end

function storage_permute(Astore::Dense{Float64},
                         Adims::NTuple{N,Int},
                         perm::Vector{Int}) where {N}
  return Dense{Float64}(vec(permutedims(reshape(data(Astore),Adims),perm)))
end

function storage_add!(Bstore::Dense{Float64},
                      Astore::Dense{Float64})
  Adata = data(Astore)
  Bdata = data(Bstore)
  Bdata .+= Adata
end

function storage_add!(Bstore::Dense{Float64},
                      Astore::Dense{Float64},
                      α::Real)
  Adata = data(Astore)
  Bdata = data(Bstore)
  Bdata .+= α.*Adata
end

# TODO: optimize this permutation (this does an extra unnecassary permutation
# since permutedims!() doesn't give the option to add the permutation to the original array)
# Maybe wrap the c version?
function storage_add!(Bstore::Dense{Float64},
                      Bdims::NTuple{N,Int},
                      Astore::Dense{Float64},
                      Adims::NTuple{N,Int},
                      perm::Vector{Int}) where {N}
  Astore = storage_permute(Astore,Adims,perm)
  storage_add!(Bstore,Astore)
end

function storage_add!(Bstore::Dense{Float64},
                      Bdims::NTuple{N,Int},
                      Astore::Dense{Float64},
                      Adims::NTuple{N,Int},
                      α::Float64,
                      perm::Vector{Int}) where {N}
  Astore = storage_permute(Astore,Adims,perm)
  storage_add!(Bstore,Astore,α)
end

# TODO: make this a special version of storage_add!()
# Make sure the permutation is optimized
function storage_permute!(Bstore::Dense{Float64},
                          Bis::IndexSet,
                          Astore::Dense{Float64},
                          Ais::IndexSet)
  p = calculate_permutation(Bis,Ais)
  Adata = data(Astore)
  Bdata = data(Bstore)
  if is_trivial_permutation(p)
    Bdata .= Adata
  else
    reshapeBdata = reshape(Bdata,dims(Bis))
    permutedims!(reshapeBdata,reshape(Adata,dims(Ais)),p)
  end
end

function storage_dag(Astore::Dense{Float64},Ais::IndexSet)
  return dag(Ais),storage_conj(Astore)
end

function storage_scalar(D::Dense{Float64})
  if length(D)==1
    return D[1]
  else
    throw(ErrorException("Cannot convert Dense -> Number for length of data greater than 1"))
  end
end

function is_outer(l1::Vector{Int},l2::Vector{Int})
  for l1i in l1
    if l1i < 0
      return false
    end
  end
  for l2i in l2
    if l2i < 0
      return false
    end
  end
  return true
end

function storage_svd(Astore::Dense{Float64},
                     Lis::IndexSet,
                     Ris::IndexSet;
                     kwargs...
                    )
  maxdim::Int = get(kwargs,:maxdim,min(dim(Lis),dim(Ris)))
  mindim::Int = get(kwargs,:mindim,1)
  cutoff::Float64 = get(kwargs,:cutoff,0.0)
  absoluteCutoff::Bool = get(kwargs,:absoluteCutoff,false)
  doRelCutoff::Bool = get(kwargs,:doRelCutoff,true)
  utags::String = get(kwargs,:utags,"Link,u")
  vtags::String = get(kwargs,:vtags,"Link,v")

  MU,MS,MV = svd(reshape(data(Astore),dim(Lis),dim(Ris)))

  sqr(x) = x^2
  P = sqr.(MS)
  #@printf "  Truncating with maxdim=%d cutoff=%.3E\n" maxdim cutoff
  truncate!(P;maxdim=maxdim,
              cutoff=cutoff,
              absoluteCutoff=absoluteCutoff,
              doRelCutoff=doRelCutoff)
  dS = length(P)
  if dS < length(MS)
    MU = MU[:,1:dS]
    resize!(MS,dS)
    MV = MV[:,1:dS]
  end

  u = Index(dS,utags)
  v = settags(u,vtags)
  Uis,Ustore = IndexSet(Lis...,u),Dense{Float64}(vec(MU))
  #TODO: make a diag storage
  Sis,Sstore = IndexSet(u,v),Dense{Float64}(vec(Matrix(Diagonal(MS))))
  Vis,Vstore = IndexSet(Ris...,v),Dense{Float64}(Vector{Float64}(vec(MV)))

  return (Uis,Ustore,Sis,Sstore,Vis,Vstore)
end

function storage_eigen(Astore::Dense{Float64},Lis::IndexSet,Ris::IndexSet,matrixtype::Type{S},truncate::Int,lefttags::String,righttags::String) where {S}
  dim_left = dim(Lis)
  dim_right = dim(Ris)
  MD,MU = eigen(S(reshape(data(Astore),dim_left,dim_right)))

  #TODO: include truncation parameters as keyword arguments
  dim_middle = min(dim_left,dim_right,truncate)
  u = Index(dim_middle,lefttags)
  v = settags(u,righttags)
  Uis,Ustore = IndexSet(Lis...,u),Dense{Float64}(vec(MU[:,1:dim_middle]))
  #TODO: make a diag storage
  Dis,Dstore = IndexSet(u,v),Dense{Float64}(vec(Matrix(Diagonal(MD[1:dim_middle]))))
  return (Uis,Ustore,Dis,Dstore)
end

function polar(A::Matrix{Float64})
  U,S,V = svd(A)
  return U*V',V*Diagonal(S)*V'
end

#TODO: make one generic function storage_factorization(Astore,Lis,Ris,factorization)
function storage_qr(Astore::Dense{Float64},Lis::IndexSet,Ris::IndexSet)
  dim_left = dim(Lis)
  dim_right = dim(Ris)
  MQ,MP = qr(reshape(data(Astore),dim_left,dim_right))
  dim_middle = min(dim_left,dim_right)
  u = Index(dim_middle,"Link,u")
  #Must call Matrix() on MQ since the QR decomposition outputs a sparse
  #form of the decomposition
  Qis,Qstore = IndexSet(Lis...,u),Dense{Float64}(vec(Matrix(MQ)))
  Pis,Pstore = IndexSet(u,Ris...),Dense{Float64}(vec(Matrix(MP)))
  return (Qis,Qstore,Pis,Pstore)
end

function storage_polar(Astore::Dense{Float64},Lis::IndexSet,Ris::IndexSet)
  dim_left = dim(Lis)
  dim_right = dim(Ris)
  MQ,MP = polar(reshape(data(Astore),dim_left,dim_right))
  dim_middle = min(dim_left,dim_right)
  #u = Index(dim_middle,"Link,u")
  Uis = addtags(Ris,"u")
  Qis,Qstore = IndexSet(Lis...,Uis...),Dense{Float64}(vec(MQ))
  Pis,Pstore = IndexSet(Uis...,Ris...),Dense{Float64}(vec(MP))
  return (Qis,Qstore,Pis,Pstore)
end

