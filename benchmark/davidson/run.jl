using ITensors, Printf, Profile, ProfileView, BenchmarkTools

function main()
  N = 100
  sites = spinOneSites(N)
  H = Heisenberg(sites)
  psi0 = randomMPS(sites)
  sw = Sweeps(5)
  cutoff!(sw,1E-12)

  # Run once to compile
  maxdim!(sw,2)
  energy,psi = dmrg(H,psi0,sw,maxiter=3)

  maxdim!(sw,10,20,100,100,200)
  energy,psi = @time dmrg(H,psi0,sw,maxiter=3)
  @printf "Final energy = %.12f\n" energy

  PH = ProjMPO(H)
  b = Int(N//2)
  position!(psi,b)
  position!(PH,psi,b)
  phi = psi[b]*psi[b+1]
  @time davidson(PH,phi;maxiter=3)

  Profile.clear()
  Profile.init(n = 10^7)
  @profile davidson(PH,phi;maxiter=3)
  ProfileView.view()

  dir = "Fromleft"
  @time replaceBond!(psi,b,phi,dir)

  return
end

