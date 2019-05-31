#include "itensor/all.h"
using namespace itensor;

int 
main()
  {
  int N = 100;
  auto sites = SpinOne(N,{"ConserveQNs=",false});
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
  sweeps.maxdim() = 10,20,100,100,200;
  sweeps.cutoff() = 1E-12;
  sweeps.niter() = 2;

  TIMER_START(1);
  auto [energy,psi] = dmrg(H,psi0,sweeps,"Quiet");
  TIMER_STOP(1);

  int b = N/2;
  auto PH = LocalMPO(H);
  psi.position(b);
  PH.position(b,psi);
  auto phi = psi(b)*psi(b+1);
  TIMER_START(2);
  energy = davidson(PH,phi,{"MaxIter=",2});
  TIMER_STOP(2);
  TIMER_START(3);
  auto spec = psi.svdBond(b,phi,Fromleft,PH);
  TIMER_STOP(3);

  return 0;
  }

