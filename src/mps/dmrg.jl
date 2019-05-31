using KrylovKit: Lanczos, eigsolve


function iterEigSolve(PH,
                      phi::ITensor;
                      kwargs...)::Tuple{Number,ITensor}

  #
  # TODO: make version which takes krylovdim, maxiter
  #       was giving energies which randomly go up!
  #
  #tol = get(kwargs,:tol,10*eps(Float64))
  #krylovdim::Int = get(kwargs,:krylovdim,2)
  #maxiter::Int = get(kwargs,:maxiter,1)

  #actualdim = 1
  #for i in inds(phi)
  #  plev(i) == 0 && (actualdim *= dim(i))
  #end
  #if krylovdim > actualdim
  #  @printf "krylovdim=%d > actualdim=%d, resetting" krylovdim actualdim
  #  krylovdim = actualdim
  #end
  #lczos = Lanczos(tol=tol,krylovdim=krylovdim,maxiter=maxiter)
  #vals, vecs, info = eigsolve(PH,phi,1,:SR,lczos)

  vals, vecs, info = eigsolve(PH,phi,1,:SR,ishermitian=true)

  #@show info.normres[1]
  #@show info.numops
  #@show info.numiter
  return vals[1],vecs[1]
end

function dmrg(H::MPO,
              psi0::MPS,
              sweeps::Sweeps;
              kwargs...)::Tuple{Float64,MPS}
  psi = copy(psi0)
  N = length(psi)

  PH = ProjMPO(H)
  position!(PH,psi0,1)
  energy = 0.0

  for sw=1:nsweep(sweeps)
    t = @elapsed begin
    for (b,ha) in sweepnext(N)
      position!(PH,psi,b)

      phi = psi[b]*psi[b+1]

      energy,phi = davidson(PH,phi;kwargs...)

      dir = ha==1 ? "Fromleft" : "Fromright"
      replaceBond!(psi,b,phi,dir;
                   maxdim=maxdim(sweeps,sw),
                   mindim=mindim(sweeps,sw),
                   cutoff=cutoff(sweeps,sw))
    end
    end
    @printf "After sweep %d/%d: energy=%.12f, maxDim=%d, sweep time=%.12f sec\n" sw nsweep(sweeps) energy maxDim(psi) t
  end
  return (energy,psi)
end

