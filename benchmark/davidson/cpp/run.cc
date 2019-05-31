#include "itensor/all.h"
using namespace itensor;

int 
main()
  {
  auto N = 100;
  auto sites = SpinOne(N,{"ConserveQNs=",false}); //make a chain of N spin 1's
  auto ampo = AutoMPO(sites);
  for(auto j : range1(N-1))
      {
      ampo += 0.5,"S+",j,"S-",j+1;
      ampo += 0.5,"S-",j,"S+",j+1;
      ampo +=     "Sz",j,"Sz",j+1;
      }
  auto H = toMPO(ampo);
  auto psi0 = randomMPS(sites);
  auto sweeps = Sweeps(5);
  sweeps.maxdim() = 10,20,100,100,150;
  sweeps.cutoff() = 1E-12;
  sweeps.niter() = 2;
  sweeps.noise() = 1E-7,1E-8,0.0;
  println(sweeps);
  TIMER_START(1);
  auto [energy,psi] = dmrg(H,psi0,sweeps,"Quiet");
  TIMER_STOP(1);

  auto PH = LocalMPO(H);
  auto b = N/2;
  psi.position(b);
  PH.position(b,psi);
  auto phi = psi(b)*psi(b+1);
  TIMER_START(2);
  davidson(PH,phi,{"MaxIter=",3});
  TIMER_STOP(2);

  return 0;
  }

