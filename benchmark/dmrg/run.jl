using ITensors, Printf, Profile, ProfileView

function main()
N = 100
sites = spinOneSites(N)
H = Heisenberg(sites)
psi0 = randomMPS(sites)
sw = Sweeps(5)
cutoff!(sw,1E-12)

#G.S. energy for N=10,S=1/2 Heisenberg is -4.258035206805 

# Run once to compile
maxdim!(sw,2)
energy,psi = @time dmrg(H,psi0,sw,maxiter=3)

GC.gc()

maxdim!(sw,10,20,100,100,200)
energy,psi = @time dmrg(H,psi0,sw,maxiter=3)
@printf "Final energy = %.12f\n" energy

Profile.clear()  # in case we have any previous profiling data
Profile.init(n = 10^7)
@profile dmrg(H,psi0,sw,maxiter=3)
ProfileView.view()

return
end

#### RESULTS
# N = 100
# sites = spinOneSites(N)
# H = Heisenberg(sites)
# psi0 = randomMPS(sites)
# sw = Sweeps(5)
#
# cutoff!(sw,1E-12)
#
# In C++: 18.8582 seconds
# In Julia Master: 34.174954 seconds
#
# cutoff!(sw,1E-10)
#
# In C++: 9.0124 seconds
# In Julia Master: 15.994319 seconds
#

