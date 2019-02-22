using ITensors,
      Nabla

include("2d_classical_ising.jl")
include("trg.jl")

function κ(β)
  d = 2
  s = Index(d)
  l = tags(s," -> left")
  r = tags(s," -> right")
  u = tags(s," -> up")
  d = tags(s," -> down")
  T = ising_mpo((l,r),(u,d),β)

  χmax = 20
  nsteps = 20
  κout,T = trg(T;χmax=χmax,nsteps=nsteps)

  return κout
end

function main()
  β = 1.1*βc
  println(≈(κ(β),exp(-β*ising_free_energy(β)),atol=1e-4))

  return ∇(κ)(β)
end

