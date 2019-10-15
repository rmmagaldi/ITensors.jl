export davidson

function get_vecs!((phi,q),M::Matrix,V,AV,ni)
  @show M
  F = eigen(Hermitian(M))
  lambda = F.values[1]
  u = F.vectors[:,1]
  mul!(phi,u[1],V[1])
  mul!(q,u[1],AV[1])
  for n=2:ni
    add!(phi,u[n],V[n])
    add!(q,u[n],AV[n])
  end
  add!(q,-lambda,phi)
  #Fix sign
  if real(u[1]) < 0.0
    scale!(phi,-1)
    scale!(q,-1)
  end
  return lambda
end

function orthogonalize!(q,V,ni)
  q0 = copy(q)
  for k=1:ni
    Vq0k = dot(V[k],q0)
    add!(q,-Vq0k,V[k])
  end
  qnrm = norm(q)
  if qnrm < 1E-10 #orthog failure, try randomizing
    randn!(q)
    qnrm = norm(q)
  end
  scale!(q,1.0/qnrm)
  return 
end

function expand_krylov_space(M::Matrix{ElT},V,AV,ni) where {ElT}
  newM = fill(zero(ElT),(ni+1,ni+1))
  newM[1:ni,1:ni] = M
  for k=1:ni+1
    newM[k,ni+1] = ElT(dot(V[k],AV[ni+1]))
    # TODO: use Hermitian wrapper,
    # setting these elements is not necessary
    newM[ni+1,k] = conj(newM[k,ni+1])
  end
  return newM
end

function davidson(A,
                  phi0::ITensorT;
                  kwargs...) where {ITensorT<:ITensor}

@timeit GLOBAL_TIMER "copy" begin
  phi = copy(phi0)
end

  maxiter = get(kwargs,:maxiter,2)
  miniter = get(kwargs,:maxiter,1)
  errgoal = get(kwargs,:errgoal,1E-14)
  Northo_pass = get(kwargs,:Northo_pass,1)

  approx0 = 1E-12

  nrm = norm(phi)
  if nrm < 1E-18 
    randn!(phi)
    nrm = norm(phi)
  end
  scale!(phi,1.0/nrm)

  maxsize = size(A)[1]
  actual_maxiter = min(maxiter,maxsize-1)

  if dim(inds(phi)) != maxsize
    error("linear size of A and dimension of phi should match in davidson")
  end

@timeit GLOBAL_TIMER "Make V" begin
  V = ITensorT[copy(phi)]
end

@timeit GLOBAL_TIMER "Make AV" begin
  AV = ITensorT[A(phi)]
end

  last_lambda = NaN

@timeit GLOBAL_TIMER "dot" begin
  lambda::Float64 = real(dot(V[1],AV[1]))
end

  q = AV[1] - lambda*V[1];

  M = fill(lambda,(1,1))

  for ni=1:actual_maxiter

    qnorm = norm(q)

    errgoal_reached = (qnorm < errgoal && 
                       abs(lambda-last_lambda) < errgoal)
    small_qnorm = (qnorm < max(approx0,errgoal*1E-3))
    converged = errgoal_reached || small_qnorm

    if (qnorm < 1E-20) || (converged && ni > miniter) #|| (ni >= actual_maxiter)
      #@printf "  done with davidson, ni=%d, qnorm=%.3E\n" ni qnorm
      break
    end

    last_lambda = lambda

    for pass = 1:Northo_pass
@timeit GLOBAL_TIMER "orthogonalize!" begin
      orthogonalize!(q,V,ni)
end
    end

@timeit GLOBAL_TIMER "Add to V" begin
    push!(V,copy(q))
end

@timeit GLOBAL_TIMER "Add to AV" begin
    push!(AV,A(q))
end

@timeit GLOBAL_TIMER "expand_krylov_space" begin
    M = expand_krylov_space(M,V,AV,ni)
end

@timeit GLOBAL_TIMER "get_vecs!" begin
    lambda = get_vecs!((phi,q),M,V,AV,ni+1)
end

  end #for ni=1:actual_maxiter+1

  return lambda,phi

end

