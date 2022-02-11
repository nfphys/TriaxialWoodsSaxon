module TriaxialWoodsSaxon

using Plots
using LinearAlgebra
using Parameters
using SparseArrays
using KrylovKit
using StaticArrays
using Test

include("./derivative.jl")
include("./hermite.jl")
include("./potential.jl")
include("./Hamiltonian.jl")
include("./states.jl")
include("./time_reversal.jl")

greet() = print("Hello World!")

@with_kw struct PhysicalParam{T} @deftype Float64
    ħc = 197.
    mc² = 938.
    M = ħc^2/2mc²

    N::Int64 = 8; @assert iseven(N) === true
    Z::Int64 = N; @assert iseven(Z) === true
    A::Int64 = Z + N; @assert A === Z + N

    ħω₀ = 41A^(-1/3)

    V₀ = -42.86 # [MeV]
    r₀ = 1.27 # [fm]
    R₀ = r₀*A^(1/3) # [fm]
    a = 0.67 # [fm]
    κ = 0.44

    Nx::Int64 = 10
    Ny::Int64 = Nx
    Nz::Int64 = Nx

    Δx = 0.9
    Δy = Δx
    Δz = Δx

    xs::T = range((1-1/2)*Δx, (Nx-1/2)*Δx, length=Nx)
    ys::T = range((1-1/2)*Δy, (Ny-1/2)*Δy, length=Ny)
    zs::T = range((1-1/2)*Δz, (Nz-1/2)*Δz, length=Nz)
end

@with_kw struct QuantumNumbers @deftype Int64
    Π = 1; @assert Π === 1 || Π === -1 # parity 
    η = 1; @assert η === 1 || η === -1 # z-signature
    q = 1; @assert q === 1 || q === 2 # q = 1 for neutron, q = 2 for proton
end

@with_kw struct SingleParticleStates 
    nstates::Int64 
    ψs::Matrix{Float64}; @assert size(ψs,2) === nstates
    spEs::Vector{Float64}; @assert length(spEs) === nstates
    qnums::Vector{QuantumNumbers}; @assert length(qnums) === nstates 
    occ::Vector{Float64}; @assert length(occ) === nstates
end




function calc_norm(Δx, Δy, Δz, ψ)
    sqrt(dot(ψ, ψ)*2Δx*2Δy*2Δz)
end


function imaginary_time_evolution!(states, convergences, param, Vs, Ws; 
    Δt=0.1, rtol_spE=1e-2)

    @unpack Nx, Ny, Nz, Δx, Δy, Δz, xs, ys, zs = param 
    @unpack nstates, ψs, spEs, qnums, occ = states

    @views for istate in 1:nstates 
        if convergences[istate] 
            continue
        end

        Hmat = make_Hamiltonian(param, Vs, Ws, qnums[istate]) 

        spE_old = calc_sp_energy(Hmat, ψs[:,istate])
        Δt = 0.1
        if spE_old > 5
            Δt = 0.02
        end

        # imaginary time evolution 
        U₁ = I - 0.5Δt*Hmat
        U₂ = I + 0.5Δt*Hmat
        @time ψs[:,istate] = U₂\(U₁*ψs[:,istate]) 

        # gram schmidt orthogonalization 
        for jstate in 1:istate-1
            if qnums[istate] !== qnums[jstate] continue end
            ψs[:,istate] .-= ψs[:,jstate] .* (dot(ψs[:,jstate], ψs[:,istate])*2Δx*2Δy*2Δz)
        end

        # normalization 
        ψs[:,istate] ./= calc_norm(Δx, Δy, Δz, ψs[:,istate])

        # single particle energy
        spE_new = calc_sp_energy(Hmat, ψs[:,istate])
        if isapprox(spEs[istate], spE_new; rtol=rtol_spE)
            convergences[istate] = true 
        end
        spEs[istate] = spE_new
    end

    return
end


function calc_states(param; 
    β=0.0, γ=0.0, Nmax=[1,1], Δt=0.1, iter_max=10, rtol_spE=1e-2,
    show_result=false)

    @assert length(Nmax) === 2

    @unpack Nx, Ny, Nz, xs, ys, zs = param 

    Vs, Ws = calc_potential(param, β, γ)

    states = initial_states(param, β, γ; Nmax=Nmax)
    sort_states!(states)
    calc_occ!(states, param)
    @unpack nstates, spEs = states

    convergences = zeros(Bool, nstates)

    spEss = zeros(Float64, nstates, iter_max)
    for iter in 1:iter_max
        println("")
        @show iter length(convergences[convergences])

        @time imaginary_time_evolution!(states, convergences, param, Vs, Ws; Δt=Δt, rtol_spE=rtol_spE)

        sort_states!(states)
        spEss[:,iter] = spEs
    end

    if show_result
        p = plot(xlabel="iter", ylabel="single-particle energy [MeV]", legend=false)
        plot!(spEss', marker=:dot)
        display(p)
        
        show_states(states)
    end

    return states
end


function plot_nilsson_diagram(param; 
    β_max=0.4, β_min=-0.4, Δβ=0.1, Nmax=[1,1], iter_max=5)

    @assert length(Nmax) === 2

    nstates = 0
    for q in 1:2
        nstates += div((Nmax[q]+1)*(Nmax[q]+2)*(Nmax[q]+3), 6) 
    end

    βs = β_min:Δβ:β_max
    Nβ = length(βs)

    spEss = zeros(Float64, nstates, Nβ)

    for iβ in 1:Nβ
        β = βs[iβ]
        println("")
        @show β

        @time states = calc_states(param; β=β, Nmax=Nmax, iter_max=iter_max)

        #spEss[:,iβ] = states.spEs 
        for istate in 1:nstates
            if states.spEs[istate] < 0
                spEss[istate,iβ] = states.spEs[istate]
            else
                spEss[istate,iβ] = NaN 
            end
        end
    end

    p = plot(xlabel="β", ylabel="single-particle energy [MeV]", legend=false)
    plot!(βs, spEss', marker=:dot)
    display(p)
end






function make_three_body_Hamiltonian(param, states; Emax=10)
    @unpack Nx, Ny, Nz, Δx, Δy, Δz, xs, ys, zs = param 
    @unpack nstates, ψs, spEs, qnums, occ = states 

    # number of two-particle states
    N = div(2nstates*(2nstates-1), 2)

    # three-body Hamiltonian
    Hmat_3body = zeros(Float64, N, N)

    n₂ = 0
    for i₂ in 1:2nstates, k₂ in 1:i₂-1
        if occ[i₂] == 1.0 || occ[k₂] == 1.0
            continue 
        end
        if qnums[i₂].q ≠ 1 || qnums[k₂].q ≠ 1 
            continue 
        end
        if spEs[i₂] + spEs[k₂] > Emax
            continue
        end

        n₂ += 1

        # single particle energy
        Hmat[n₂, n₂] += spEs[i₂] + spEs[k₂]

        n₁ = 0
        for i₁ in 1:2nstates, k₁ in 1:i₁-1 
            if occ[i₁] == 1.0 || occ[k₁] == 1.0
                continue 
            end
            if qnums[i₁].q ≠ 1 || qnums[k₁].q ≠ 1 
                continue 
            end
            if spEs[i₁] + spEs[k₁] > Emax
                continue
            end

            n₁ += 1 



        end
    end
    return 
end





end # module
