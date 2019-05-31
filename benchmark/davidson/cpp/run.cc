#include "itensor/all.h"
using namespace itensor;

int 
main()
  {
  auto ldim = 300;
  auto hdim = 5;
  auto sdim = 2;

  auto s2 = Index(sdim,"s2");
  auto s3 = Index(sdim,"s3");

  auto h1 = Index(hdim,"h1");
  auto h2 = Index(hdim,"h2");
  auto h3 = Index(hdim,"h3");

  auto l1 = Index(ldim,"l1");
  auto l2 = Index(ldim,"l2");
  auto l3 = Index(ldim,"l3");

  auto A1 = randomITensor(l1,s2,l2);
  auto A2 = randomITensor(l2,s3,l3);
  auto B0 = A1*A2;

  auto H1 = randomITensor(h1,s2,prime(s2),h2);
  auto H2 = randomITensor(h2,s3,prime(s3),h3);

  auto L = randomITensor(l1,h1,prime(l1));
  auto R = randomITensor(l3,h3,prime(l3));

  TIMER_START(5);
  TIMER_START(1);
  auto B1 = L*B0;
  TIMER_STOP(1);
  TIMER_START(2);
  auto B2 = B1*H1;
  TIMER_STOP(2);
  TIMER_START(3);
  auto B3 = B2*H2;
  TIMER_STOP(3);
  TIMER_START(4);
  auto B4 = B3*R;
  TIMER_STOP(4);
  TIMER_STOP(5);

  return 0;
  }

