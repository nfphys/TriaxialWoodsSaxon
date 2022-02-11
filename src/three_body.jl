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
    @views for i₃ in 1:2nstates, i₄ in 1:i₃-1 # i₃ < i₄
        if occ[i₃] == 1.0 || occ[i₄] == 1.0
            continue 
        end
        if qnums[i₃].q ≠ 1 || qnums[i₄].q ≠ 1 
            continue 
        end
        if spEs[i₃] + spEs[i₄] > E_cut
            continue
        end

        n₃₄ += 1

        # single particle energy
        Hmat_3body[n₃₄, n₃₄] += spEs[i₃] + spEs[i₄]

        if isodd(i₃)
            ψ₃ = ψs[:,i₃]
        else
            time_reversal!(ψ₃, ψs[:,i₃])
        end

        if isood(i₄)
            ψ₄ = ψs[:,i₄]
        else
            time_reversal!(ψ₄, ψs[:,i₄])
        end

        n₁₂ = 0
        for i₁ in 1:2nstates, i₂ in 1:i₁-1  # i₁ < i₂
            if occ[i₁] == 1.0 || occ[i₂] == 1.0
                continue 
            end
            if qnums[i₁].q ≠ 1 || qnums[i₂].q ≠ 1 
                continue 
            end
            if spEs[i₁] + spEs[i₂] > E_cut
                continue
            end

            n₁₂ += 1 

            if isood(i₁)
                ψ₁ = ψs[:,i₁]
            else
                time_reversal!(ψ₁, ψs[:,i₁])
            end

            if isodd(i₂)
                ψ₂ = ψs[:,i₂]
            else
                time_reversal!(ψ₂, ψs[:,i₂])
            end

            I = 0.0
            for α₁ in 1:4, α₂ in 1:4, iz in 1:Nz, iy in 1:Ny, ix in 1:Nx
                i = f(ix, iy, iz, α₁)
                j = f(ix, iy, iz, α₂)
                I += V_nn[ix,iy,iz]*ψ₁[i]*ψ₂[j]*(ψ₃[i]*ψ₄[j] - ψ₄[i]*ψ₃[j])
            end

        end
    end

    return Hmat_3body
end

function test_make_three_body_Hamiltonian(param)
    
end