using ITensors, Printf, Profile, ProfileView, TensorOperations

function main()

ldim = 1000
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

#L = permute(L,h1,l1,prime(l1))

#function contract(A,B)
#  return A*B
#end

#function contractH1(B,L)
#  return B*H1
#end
#
#function contractH2(B,L)
#  return B*H2
#end
#
#function contractR(B,R)
#  return B*R
#end

println("ITensor:")
B1 = @time *(L,B0)
B2 = @time *(B1,H1)
B3 = @time *(B2,H2)
B4 = @time *(B3,R)

@show inds(B0)
@show inds(B1)
@show inds(B2)
@show inds(B3)
@show inds(B4)

# TensorOperations
B0arr = Array(B0)
Larr = Array(L)
H1arr = Array(H1)
H2arr = Array(H2)
Rarr = Array(R)

println("TensorOperations:")
@time @tensor B1arr[h1,l1',s2,s3,l3] := Larr[l1,h1,l1']*B0arr[l1,s2,s3,l3]
@time @tensor B2arr[l1',s3,l3,s2',h2] := B1arr[h1,l1',s2,s3,l3]*H1arr[h1,s2,s2',h2]
@time @tensor B3arr[l1',l3,s2',s3',h3] := B2arr[l1',s3,l3,s2',h2]*H2arr[h2,s3,s3',h3]
@time @tensor B4arr[l1',s2',s3',l3'] := B3arr[l1',l3,s2',s3',h3]*Rarr[l3,h3,l3']

#GC.gc()

# Profile
#*(B0,L,H1,H2,R)
#@time *(B0,L,H1,H2,R)
#Profile.clear()  # in case we have any previous profiling data
#Profile.init(n = 10^7, delay = 0.0001)
#@profile *(B0,L,H1,H2,R)
#ProfileView.view()

# Profile
#println("ITensor:")
#*(B1,H1)
#@time *(B1,H1)
#Profile.clear()  # in case we have any previous profiling data
#Profile.init(n = 10^7, delay = 0.0001)
#@profile *(B1,H1)
#ProfileView.view()
#
# Profile
#println("TensorOperations:")
#@tensor B2arr[l1',s3,l3,s2',h2] := B1arr[h1,l1',s2,s3,l3]*H1arr[h1,s2,s2',h2]
#@time @tensor B2arr[l1',s3,l3,s2',h2] := B1arr[h1,l1',s2,s3,l3]*H1arr[h1,s2,s2',h2]
#Profile.clear()  # in case we have any previous profiling data
#Profile.init(n = 10^7, delay = 0.0001)
#@profile @tensor B2arr[l1',s3,l3,s2',h2] := B1arr[h1,l1',s2,s3,l3]*H1arr[h1,s2,s2',h2]
#ProfileView.view()

return
end

