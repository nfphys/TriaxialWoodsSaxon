function make_three_body_Hamiltonian(param, states; β=0.0, γ=0.0)
    @unpack mc², ħc, Nx, Ny, Nz, Δx, Δy, Δz, xs, ys, zs, 
    E_cut, v₀, v_rho, R_rho, a_rho = param 

    @unpack nstates, ψs, spEs, qnums, occ = states 

    N_mesh = Nx*Ny*Nz 

    # number of two-particle states
    N_2p = div(2nstates*(2nstates-1), 2)

    # three-body Hamiltonian
    Hmat_3body = zeros(Float64, N_2p, N_2p)

    # single-particle wave functions
    ψ₁ = zeros(Float64, 4N_mesh)
    ψ₂ = zeros(Float64, 4N_mesh)
    ψ₃ = zeros(Float64, 4N_mesh)
    ψ₄ = zeros(Float64, 4N_mesh)

    # effective interaction
    V_nn = zeros(Float64, Nx, Ny, Nz)
    N20 = √(5/16π)
    N22 = √(15/32π)
    βcosγ = β*cos(γ)
    βsinγ = β*sin(γ)
    for iz in 1:Nz, iy in 1:Ny, ix in 1:Nx 
        x = xs[ix]
        y = ys[iy]
        z = zs[iz]

        # polar coordinates
        r = √(x*x + y*y + z*z)
        θ = acos(z/r)
        ϕ = sign(y)*acos(x/√(x*x + y*y))

        # spherical harmonics
        Y20    = N20 * (3cos(θ)^2 - 1)
        Y22_Re = N22 * sin(θ)^2 * cos(2ϕ) # real part of Y22

        # nuclear radius
        R = R_rho*(1 + βcosγ*Y20 + √2*βsinγ*Y22_Re)

        V_nn[ix,iy,iz] += v₀ + v_rho/(1 + exp((r - R)/a_rho))
    end



    f(ix,iy,iz,α) = ix + (iy-1)*Nx + (iz-1)*Nx*Ny + (α-1)*Nx*Ny*Nz

    n₃₄ = 0
    @views for i₃ in 1:2nstates, i₄ in 1:i₃-1 # i₃ > i₄
        if occ[cld(i₃,2)] == 1.0 || occ[cld(i₄,2)] == 1.0
            continue 
        end
        if qnums[cld(i₃,2)].q ≠ 1 || qnums[cld(i₄,2)].q ≠ 1 
            continue 
        end
        if spEs[cld(i₃,2)] + spEs[cld(i₄,2)] > E_cut
            continue
        end

        n₃₄ += 1

        # single particle energy
        Hmat_3body[n₃₄, n₃₄] += spEs[cld(i₃,2)] + spEs[cld(i₄,2)]

        if isodd(i₃)
            ψ₃ = ψs[:,cld(i₃,2)]
        else
            time_reversal!(ψ₃, param, ψs[:,cld(i₃,2)])
        end

        if isodd(i₄)
            ψ₄ = ψs[:,cld(i₄,2)]
        else
            time_reversal!(ψ₄, param, ψs[:,cld(i₄,2)])
        end

        n₁₂ = 0
        for i₁ in 1:2nstates, i₂ in 1:i₁-1  # i₁ > i₂
            if occ[cld(i₁,2)] == 1.0 || occ[cld(i₂,2)] == 1.0
                continue 
            end
            if qnums[cld(i₁,2)].q ≠ 1 || qnums[cld(i₂,2)].q ≠ 1 
                continue 
            end
            if spEs[cld(i₁,2)] + spEs[cld(i₂,2)] > E_cut
                continue
            end

            n₁₂ += 1 

            if isodd(i₁)
                ψ₁ = ψs[:,cld(i₁,2)]
            else
                time_reversal!(ψ₁, param, ψs[:,cld(i₁,2)])
            end

            if isodd(i₂)
                ψ₂ = ψs[:,cld(i₂,2)]
            else
                time_reversal!(ψ₂, param, ψs[:,cld(i₂,2)])
            end

            I = 0.0
            for α₁ in 1:4, α₂ in 1:4, iz in 1:Nz, iy in 1:Ny, ix in 1:Nx
                i = f(ix, iy, iz, α₁)
                j = f(ix, iy, iz, α₂)
                I += V_nn[ix,iy,iz]*ψ₁[i]*ψ₂[j]*(ψ₃[i]*ψ₄[j] - ψ₄[i]*ψ₃[j])
            end

            if n₁₂ == n₃₄
                E₀ = spEs[cld(i₃,2)] + spEs[cld(i₄,2)]
                println("")
                @show E₀ I 
            end

            Hmat_3body[n₁₂, n₃₄] += I

        end
    end

    return Hmat_3body[1:n₃₄, 1:n₃₄]
end

function test_make_three_body_Hamiltonian(param; Nmax=[2,1])
    states = calc_states(param; Nmax=Nmax)
    @time Hmat_3body = make_three_body_Hamiltonian(param, states)
end