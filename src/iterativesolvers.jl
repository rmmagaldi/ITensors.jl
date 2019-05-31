
function get_vecs(M::Matrix{Float64},
                  V::Vector{ITensor{Dense{Float64}}},
                  AV::Vector{ITensor{Dense{Float64}}},
                  ni::Int)::Tuple{Float64,ITensor{Dense{Float64}},ITensor{Dense{Float64}}}
  F = eigen(Hermitian(M))
  lambda = F.values[1]
  u = F.vectors[:,1]
  phi = u[1]*V[1]
  q = u[1]*AV[1]
  for n=2:ni
    add_noperm!(phi,V[n],u[n])
    add_noperm!(q,AV[n],u[n])
  end
  add_noperm!(q,phi,-lambda)
  #Fix sign
  if real(u[1]) < 0
    scale!(phi,-1.0)
    scale!(q,-1.0)
  end
  return lambda,phi,q
end

function orthogonalize(q::ITensor{Dense{Float64}},
                       V::Vector{ITensor{Dense{Float64}}},
                       ni::Int)::ITensor{Dense{Float64}}
  q0 = q
  q_ortho = copy(q)
  for k=1:ni
    Vqk = dot(V[k],q0)
    add_noperm!(q_ortho,V[k],-Vqk)
  end
  qnrm = norm(q_ortho)
  if qnrm < 1E-10 #orthog failure, try randomizing
    # TODO: put random recovery code here
    error("orthog failure")
  end
  scale!(q_ortho,1.0/qnrm)
  return q_ortho
end

function davidson(A,
                  phi0::ITensor{Dense{Float64}};
                  kwargs...)
  phi = copy(phi0)

  maxiter = get(kwargs,:maxiter,3)
  miniter = get(kwargs,:miniter,1)
  errgoal = get(kwargs,:errgoal,1E-14)
  Northo_pass = get(kwargs,:Northo_pass,2)

  approx0 = 1E-12

  nrm = norm(phi)
  if nrm < 1E-18 
    randn!(phi)
    nrm = norm(phi)
  end
  phi /= nrm

  maxsize = size(A)[1]
  actual_maxiter = min(maxiter,maxsize-1)

  if dim(inds(phi)) != maxsize
    error("linear size of A and dimension of phi should match in davidson")
  end

  V = ITensor{Dense{Float64}}[phi]
  AV = ITensor{Dense{Float64}}[A(phi)]

  last_lambda = NaN
  lambda = dot(V[1],AV[1])
  q::ITensor{Dense{Float64}} = AV[1];
  add_noperm!(q,V[1],-lambda);

  M = fill(lambda,(1,1))

  for ni=1:actual_maxiter+1

    if ni > 1
      lambda,phi,q = get_vecs(M,V,AV,ni)
    end

    qnorm = norm(q)

    errgoal_reached = (qnorm < errgoal && abs(lambda-last_lambda) < errgoal)
    small_qnorm = (qnorm < max(approx0,errgoal*1E-3))
    converged = errgoal_reached || small_qnorm

    if (qnorm < 1E-20) || (converged && ni > miniter_) || (ni >= actual_maxiter)
      #@printf "  done with davidson, ni=%d, qnorm=%.3E\n" ni qnorm
      break
    end

    last_lambda = lambda

    for pass = 1:Northo_pass
      q = orthogonalize(q,V,ni)
    end

    push!(V,q)
    push!(AV,A(q))

    newM = fill(0.0,(ni+1,ni+1))
    newM[1:ni,1:ni] = M
    for k=1:ni+1
      newM[k,ni+1] = dot(V[k],AV[ni+1])
      newM[ni+1,k] = conj(newM[k,ni+1])
    end
    M = newM
  end #for ni=1:actual_maxiter+1

  #phi /= norm(phi)

  return lambda,phi

end

