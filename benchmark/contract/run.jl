using ITensors, Printf, Profile, ProfileView, BenchmarkTools, TensorOperations

function main()
  ldim = 300
  hdim = 5
  sdim = 2

  s2 = Index(sdim,"s2")
  s3 = Index(sdim,"s3")

  h1 = Index(hdim,"h1")
  h2 = Index(hdim,"h2")
  h3 = Index(hdim,"h3")

  l1 = Index(ldim,"l1")
  l2 = Index(ldim,"l2")
  l3 = Index(ldim,"l3")

  A1 = randomITensor(l1,s2,l2)
  A2 = randomITensor(l2,s3,l3)
  B0 = A1*A2

  H1 = randomITensor(h1,s2,prime(s2),h2)
  H2 = randomITensor(h2,s3,prime(s3),h3)

  L = randomITensor(l1,h1,prime(l1))
  R = randomITensor(l3,h3,prime(l3))

  println("ITensor:")
  @time begin
    @time B1 = L*B0
    @time B2 = B1*H1
    @time B3 = B2*H2
    @time B4 = B3*R
  end

  # TensorOperations
  B0arr = Array(B0)
  Larr = Array(L)
  H1arr = Array(H1)
  H2arr = Array(H2)
  Rarr = Array(R)

  println("TensorOperations:")
  @time begin
    @time @tensor B1arr[h1,l1',s2,s3,l3] := Larr[l1,h1,l1']*B0arr[l1,s2,s3,l3]
    @time @tensor B2arr[l1',s3,l3,s2',h2] := B1arr[h1,l1',s2,s3,l3]*H1arr[h1,s2,s2',h2]
    @time @tensor B3arr[l1',l3,s2',s3',h3] := B2arr[l1',s3,l3,s2',h2]*H2arr[h2,s3,s3',h3]
    @time @tensor B4arr[l1',s2',s3',l3'] := B3arr[l1',l3,s2',s3',h3]*Rarr[l3,h3,l3']
  end

  return
end

