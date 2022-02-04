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

greet() = print("Hello World!")

@with_kw struct PhysicalParam{T} @deftype Float64
    ħc = 197.
    mc² = 938.
    M = ħc^2/2mc²

    Z::Int64 = 8; @assert iseven(Z) === true
    N::Int64 = Z; @assert iseven(N) === true
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

    Δx = 0.8
    Δy = Δx
    Δz = Δx

    xs::T = range((1-1/2)*Δx, (Nx-1/2)*Δx, length=Nx)
    ys::T = range((1-1/2)*Δy, (Ny-1/2)*Δy, length=Ny)
    zs::T = range((1-1/2)*Δz, (Nz-1/2)*Δz, length=Nz)
end

@with_kw struct QuantumNumbers @deftype Int64
    Π = 1; @assert Π === 1 || Π === -1
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

function imaginary_time_evolution!(states, param, Vs, Ws; Δt=0.1)
    @unpack Nx, Ny, Nz, Δx, Δy, Δz, xs, ys, zs = param 
    @unpack nstates, ψs, spEs, qnums, occ = states

    @views for istate in 1:nstates 
        Hmat = make_Hamiltonian(param, Vs, Ws, qnums[istate]) 

        U₁ = I - 0.5Δt*Hmat
        U₂ = I + 0.5Δt*Hmat

        ψs[:,istate] = U₂\(U₁*ψs[:,istate]) # ここがボトルネック

        # gram schmidt orthogonalization 
        for jstate in 1:istate-1
            if qnums[istate] !== qnums[jstate] continue end
            ψs[:,istate] .-= ψs[:,jstate] .* (dot(ψs[:,jstate], ψs[:,istate])*2Δx*2Δy*2Δz)
        end

        # normalization 
        ψs[:,istate] ./= calc_norm(Δx, Δy, Δz, ψs[:,istate])
        spEs[istate] = calc_sp_energy(Hmat, ψs[:,istate])
    end

    return
end


function calc_states(param; β=0.0, γ=0.0, Δt=0.1, iter_max=20)
    @unpack Nx, Ny, Nz, xs, ys, zs = param 
    N = Nx*Ny*Nz

    Vs, Ws = calc_potential(param, β, γ)

    states = initial_states(param, β, γ)
    sort_states!(states)
    calc_occ!(states, param)
    @unpack nstates, spEs = states

    spEss = zeros(Float64, nstates, iter_max)
    for iter in 1:iter_max
        @time imaginary_time_evolution!(states, param, Vs, Ws; Δt=Δt)
        sort_states!(states)
        spEss[:,iter] = spEs
    end

    p = plot(xlabel="iter", ylabel="single-particle energy [MeV]", legend=false)
    plot!(spEss', marker=:dot)
    display(p)
    
    show_states(states)
end





end # module
