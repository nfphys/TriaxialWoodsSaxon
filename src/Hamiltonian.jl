function make_Hamiltonian(param, Vs, Ws, qnum)
    @unpack Nx, Ny, Nz, Δx, Δy, Δz, xs, ys, zs, M = param
    N = 4*Nx*Ny*Nz

    f(ix,iy,iz,α) = ix + (iy-1)*Nx + (iz-1)*Nx*Ny + (α-1)*Nx*Ny*Nz

    @unpack Π = qnum 

    N_diff = 2

    Πx = SA[+1, -1, -1, +1]
    Πy = SA[+1, -1, +1, -1]
    Πz = SA[+Π, +Π, -Π, -Π]

    Sx = SA[ 0  0  0  1;
             0  0 -1  0;
             0  1  0  0;
            -1  0  0  0]

    Sy = SA[ 0  0 -1  0;
             0  0  0 -1;
             1  0  0  0;
             0  1  0  0]

    Sz = SA[ 0  1  0  0;
            -1  0  0  0;
             0  0  0 -1;
             0  0  1  0]

    Hmat = zeros(Float64, N, N)
    for α in 1:4, iz in 1:Nz, iy in 1:Ny, ix in 1:Nx
        i = f(ix,iy,iz,α)

        # central term
        Hmat[i,i] += Vs[ix, iy, iz]

        for dx in -N_diff:N_diff
            jx = ix + dx; if !(1 ≤ jx ≤ Nx) continue end
            jy = iy
            jz = iz

            # kinetic term
            j = f(jx,jy,jz,α)
            Hmat[i,j] += -M*second_deriv_coeff(ix, jx, Δx, Nx, Πx[α])

            # ls term
            for β in 1:4
                j = f(jx,jy,jz,β)
                cij_x = first_deriv_coeff(ix, jx, Δx, Nx, Πx[β])
                Hmat[i,j] += 0.5(Sy[α,β]*Ws[ix,iy,iz,3] - Sz[α,β]*Ws[ix,iy,iz,2])*cij_x
            end
        end

        for dy in -N_diff:N_diff
            jx = ix
            jy = iy + dy; if !(1 ≤ jy ≤ Ny) continue end
            jz = iz

            # kinetic term
            j = f(jx,jy,jz,α)
            Hmat[i,j] += -M*second_deriv_coeff(iy, jy, Δy, Ny, Πy[α])

            # ls term
            for β in 1:4
                j = f(jx,jy,jz,β)
                cij_y = first_deriv_coeff(iy, jy, Δy, Ny, Πy[β])
                Hmat[i,j] += 0.5(Sz[α,β]*Ws[ix,iy,iz,1] - Sx[α,β]*Ws[ix,iy,iz,3])*cij_y
            end
        end

        for dz in -N_diff:N_diff
            jx = ix
            jy = iy 
            jz = iz + dz; if !(1 ≤ jz ≤ Nz) continue end 

            # kinetic term
            j = f(jx,jy,jz,α)
            Hmat[i,j] += -M*second_deriv_coeff(iz, jz, Δz, Nz, Πz[α])

            # ls term
            for β in 1:4
                j = f(jx,jy,jz,β)
                cij_z = first_deriv_coeff(iz, jz, Δz, Nz, Πz[β])
                Hmat[i,j] += 0.5(Sx[α,β]*Ws[ix,iy,iz,2] - Sy[α,β]*Ws[ix,iy,iz,1])*cij_z
            end
        end
    end

    return sparse(Hmat)
end




function test_make_Hamiltonian(param; ħω=10, κ=0.1, howmany=10)
    @unpack Nx, Ny, Nz, xs, ys, zs, M = param 
    N = 4*Nx*Ny*Nz 

    Vs = zeros(Float64, Nx, Ny, Nz) # harmonic oscillator potential 
    Ws = zeros(Float64, Nx, Ny, Nz, 3) # derivative of h.o. potential

    for iz in 1:Nz, iy in 1:Ny, ix in 1:Nx 
        x = xs[ix]
        y = ys[iy]
        z = zs[iz]
        
        r² = x*x + y*y + z*z 

        Vs[ix,iy,iz] = r²

        Ws[ix,iy,iz,1] = x
        Ws[ix,iy,iz,2] = y
        Ws[ix,iy,iz,3] = z
    end
    @. Vs *= ħω*ħω/4M
    @. Ws *= κ*ħω

    # calculate exact eigenvalues
    nmax = 10
    lmax = 10
    vals1_exact = Float64[]
    vals2_exact = Float64[]
    for n in 1:nmax, l in 0:lmax, j in 2l+1: -2: max(2l-1, 0), m in j: -2: 0
        ls = (j*(j+2) - 4l*(l+1) - 3)/8
        E = 2(n-1) + l + 1.5 + κ*ls 
        if iseven(l)
            push!(vals1_exact, E)
        else
            push!(vals2_exact, E)
        end
    end
    sort!(vals1_exact)
    sort!(vals2_exact)

    # calculate eigenvalues numerically
    qnum = QuantumNumbers(Π=+1)
    @time Hmat = make_Hamiltonian(param, Vs, Ws, qnum)
    #@show Hmat == Hmat'
    @time vals1, vecs1, info1 = eigsolve(Hmat, howmany, :SR, eltype(Hmat))
    @. vals1 *= 1/ħω

    qnum = QuantumNumbers(Π=-1)
    @time Hmat = make_Hamiltonian(param, Vs, Ws, qnum)
    #@show Hmat == Hmat'
    @time vals2, vecs2, info2 = eigsolve(Hmat, howmany, :SR, eltype(Hmat))
    @. vals2 *= 1/ħω

    # compare exact and numerical eigenvalues
    vals = zeros(Float64, howmany, 4)
    vals[:,1] = vals1[1:howmany]
    @. vals[:,2] = abs(100*vals1[1:howmany]/vals1_exact[1:howmany] - 100)
    vals[:,3] = vals2[1:howmany]
    @. vals[:,4] = abs(100*vals2[1:howmany]/vals2_exact[1:howmany] - 100)

    vals
end