using LinearAlgebra:svd, Diagonal
using TensorOperations
using Nabla

@unionise function TRG(K::RT, Dcut::Int, no_iter::Int) where RT<:Real
    D = 2
    inds = 1:D

    M = [sqrt(cosh(K)) sqrt(sinh(K));
         sqrt(cosh(K)) -sqrt(sinh(K))]

    T = zeros(RT, D, D, D, D)
    for i in inds, j in inds, k in inds, l in inds
        for a in inds
            T[i, j, k, l] += M[a, i] * M[a, j] * M[a, k] * M[a, l]
        end
    end

    lnZ = zero(RT)
    for n in 1:no_iter

        #println(n, " ", maximum(T), " ", minimum(T))
        maxval = maximum(T)
        T = T/maxval
        lnZ += 2^(no_iter-n+1)*log(maxval)

        D_new = min(D^2, Dcut)

        Ma = reshape(permutedims(T, (3, 2, 1, 4)),  (D^2, D^2))
        Mb = reshape(permutedims(T, (4, 3, 2, 1)),  (D^2, D^2))

        F = svd(Ma)
        FS = view(F.S, 1:D_new)
        S1 = reshape(view(F.U,:,1:D_new)*Diagonal(sqrt.(FS)), (D, D, D_new))
        S3 = reshape(Diagonal(sqrt.(FS))*view(F.Vt,1:D_new,:), (D_new, D, D))
        F = svd(Mb)
        FS = view(F.S, 1:D_new)
        S2 = reshape(view(F.U,:,1:D_new)*Diagonal(sqrt.(FS)), (D, D, D_new))
        S4 = reshape(Diagonal(sqrt.(FS))*view(F.Vt,1:D_new,:), (D_new, D, D))

        # @tensoropt is much faster than @tensor
        @tensoropt T_new[r, u, l, d] := S1[w, a, r] * S2[a, b, u] * S3[l, b, g] * S4[d, g, w]

        D = D_new
        T = T_new
    end
    trace = zero(RT)
    for i in 1:D
        trace += T[i, i, i, i]
    end
    lnZ += log(trace)
end

#= The Flux Version
Dcut = 24
n = 20

using Flux
K = 0.5
#K = param(0.5)
TRG(K, Dcut, n)
=#

function main()
    Dcut = 24
    n = 20

    K = 0.5
    #TRG(K, Dcut, n)
    ∇(k -> TRG(k, Dcut, n))(K)
end
