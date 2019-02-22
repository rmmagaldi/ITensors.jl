using ITensors,
      Nabla

include("2d_classical_ising.jl")
include("trg.jl")

@unionise function κ(β)
  d = 2
  s = Index(d)
  l = tags(s," -> left")
  r = tags(s," -> right")
  u = tags(s," -> up")
  d = tags(s," -> down")
  T = ising_mpo((l,r),(u,d),β;dual_lattice=false)
  χmax = 20
  nsteps = 20
  κout,T = trg(T;χmax=χmax,nsteps=nsteps)
  return κout
end

function main()
  # Test Nabla on SVD
  A = rand(5, 5)
  @unionise f(x) = sum(svd(x).U)
  println("∇(f)(A) = ", ∇(f)(A))

  β = 1.1*βc
  κ_exact = exp(-β*ising_free_energy(β))

  # Test TRG on Ising model
  ε_trg = abs(κ(β)-κ_exact)/κ_exact
  println("ε_trg = ",ε_trg)

  # Test Nabla on TRG
  ∇(κ)(β)

end

