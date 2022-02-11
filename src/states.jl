"""
    calc_HOWF(b, n, x)

Calculate harmonic oscillator wave function.
"""
function calc_HOWF(b, n, x)
    ξ = x/b

    1/√(√π * b * 2^n * factorial(n)) * exp(-0.5ξ*ξ) * hermite(n,ξ)
end


function test_calc_HOWF(param; n=0)
    @unpack ħc, mc², ħω₀, A, ħω₀, Nx, xs, Δx = param 

    b = sqrt(ħc*ħc/(mc²*ħω₀))

    fs = zeros(Float64, Nx)
    for ix in 1:Nx 
        fs[ix] = calc_HOWF(b, n, xs[ix])
    end
    p = plot(xs, fs)
    display(p)
    @show dot(fs, fs)*2Δx 

    return
end


"""
    initial_states(param, β, γ; Nmax=2)

Make initial states as eigenstates of a deformed H.O. potential.
"""
function initial_states(param, β, γ; Nmax=[2,2])
    @assert length(Nmax) === 2

    @unpack ħc, mc², A, ħω₀, Nx, Ny, Nz, xs, ys, zs = param 
    N = 4*Nx*Ny*Nz 

    f(ix,iy,iz,α) = ix + (iy-1)*Nx + (iz-1)*Nx*Ny + (α-1)*Nx*Ny*Nz

    nstates = 0
    for q in 1:2
        nstates += div((Nmax[q]+1)*(Nmax[q]+2)*(Nmax[q]+3), 6) 
    end
    ψs = zeros(Float64, N, nstates) # wave function 
    spEs = zeros(Float64, nstates) # single particle energy
    qnums = Vector{QuantumNumbers}(undef, nstates) # quantum number 
    occ = zeros(Float64, nstates) # occupation number 

    # angular frequency
    ħω₁ = ħω₀*(1 - sqrt(5/4π)*β*cos(γ-2π/3))
    ħω₂ = ħω₀*(1 - sqrt(5/4π)*β*cos(γ+2π/3))
    ħω₃ = ħω₀*(1 - sqrt(5/4π)*β*cos(γ))

    # oscillator length
    b₁ = sqrt(ħc*ħc/(mc²*ħω₁))
    b₂ = sqrt(ħc*ħc/(mc²*ħω₂))
    b₃ = sqrt(ħc*ħc/(mc²*ħω₃))

    istate = 0
    for q in 1:2, nz in 0:Nmax[q], ny in 0:Nmax[q], nx in 0:Nmax[q]
        if (nx + ny + nz > Nmax[q]) continue end
        istate += 1

        α = 0
        if iseven(nx) && iseven(ny)
            α = 1
        end
        if isodd(nx) && isodd(ny)
            α = 2
        end
        if isodd(nx) && iseven(ny)
            α = 3
        end
        if iseven(nx) && isodd(ny)
            α = 4
        end

        for iz in 1:Nz, iy in 1:Ny, ix in 1:Nx 
            i = f(ix,iy,iz,α)
            ψs[i, istate] = calc_HOWF(b₁, nx, xs[ix]) * 
                            calc_HOWF(b₂, ny, ys[iy]) *
                            calc_HOWF(b₃, nz, zs[iz])
        end

        spEs[istate] = ħω₁*(nx+1/2) + ħω₂*(ny+1/2) + ħω₃*(nz+1/2)
        qnums[istate] = QuantumNumbers(Π=(-1)^(nx+ny+nz), q=q)

    end
    
    return SingleParticleStates(nstates, ψs, spEs, qnums, occ)
end



function sort_states!(states)
    @unpack ψs, spEs, qnums = states
    p = sortperm(spEs)

    ψs[:,:] = ψs[:,p]
    spEs[:] = spEs[p]
    qnums[:] = qnums[p]
    return 
end

function calc_occ!(states, param)
    @unpack A, N, Z = param 
    @unpack nstates, qnums, occ = states

    fill!(occ, 0)
    #occupied_states = 0
    n_neut = 0
    n_prot = 0
    for i in 1:nstates 
        @unpack q = qnums[i]
        if q === 1
            if n_neut + 2 ≤ N
                occ[i] = 1
                n_neut += 2
            elseif n_neut < N
                occ[i] = (N - n_neut)/2
                n_neut = N 
            end
        else
            if n_prot + 2 ≤ Z
                occ[i] = 1
                n_prot += 2
            elseif n_prot < N
                occ[i] = (N - n_prot)/2
                n_prot = N 
            end
        end
    end

    @assert n_neut == N && n_prot == Z
    return 
end


function show_states(states)
    @unpack nstates, ψs, spEs, qnums, occ = states
    println("")
    for i in 1:nstates
        println("i = ", i, ": ")
        @show spEs[i] occ[i] qnums[i]
    end
end

function calc_sp_energy(Hmat, ψ)
    dot(ψ, Hmat, ψ)/dot(ψ, ψ)
end


function test_initial_states(param; 
    β=0.0, γ=0.0, Nmax=[2,2], rtol_norm=1e-2, rtol_spE=1e-1)

    @assert length(Nmax) === 2

    @unpack ħc, mc², M, Nx, Ny, Nz, Δx, Δy, Δz, ħω₀, xs, ys, zs = param 

    # angular frequency
    @show sqrt(5/4π)*β*cos(γ)
    ħω₁ = ħω₀*(1 - sqrt(5/4π)*β*cos(γ-2π/3))
    ħω₂ = ħω₀*(1 - sqrt(5/4π)*β*cos(γ+2π/3))
    ħω₃ = ħω₀*(1 - sqrt(5/4π)*β*cos(γ))

    @show ħω₀ ħω₁ ħω₂ ħω₃
    println("")

    # oscillator length
    b₀ = sqrt(ħc*ħc/(mc²*ħω₀))
    b₁ = sqrt(ħc*ħc/(mc²*ħω₁))
    b₂ = sqrt(ħc*ħc/(mc²*ħω₂))
    b₃ = sqrt(ħc*ħc/(mc²*ħω₃))

    @show b₀ b₁ b₂ b₃
    println("")

    @time states = initial_states(param, β, γ; Nmax=Nmax) 
    @unpack nstates, ψs, spEs, qnums, occ = states

    @time sort_states!(states)
    calc_occ!(states, param)

    @time @testset "norm" begin 
        for i in 1:nstates 
            @test dot(ψs[:,i], ψs[:,i])*2Δx*2Δy*2Δz ≈ 1 rtol=rtol_norm
        end
    end


    Vs = zeros(Float64, Nx, Ny, Nz) # harmonic oscillator potential 
    Ws = zeros(Float64, Nx, Ny, Nz, 3) # derivative of h.o. potential

    for iz in 1:Nz, iy in 1:Ny, ix in 1:Nx 
        x = xs[ix]
        y = ys[iy]
        z = zs[iz]
        
        r² = (ħω₁/ħω₀)^2*x*x + (ħω₂/ħω₀)^2*y*y + (ħω₃/ħω₀)^2*z*z 

        Vs[ix,iy,iz] = r²
    end
    @. Vs *= ħω₀*ħω₀/4M


    @time @testset "single particle energy" begin 
        for i in 1:nstates
            Hmat = make_Hamiltonian(param, Vs, Ws, qnums[i])
            E₁ = calc_sp_energy(Hmat, ψs[:,i])
            E₂ = spEs[i]
            @show E₁ E₂
            @test E₁ ≈ E₂ rtol=rtol_spE
        end
    end

    show_states(states)
    
end