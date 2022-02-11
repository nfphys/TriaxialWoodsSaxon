function time_reversal!(ψ_reversed, param, ψ)
    @unpack Nx, Ny, Nz = param 
    f(ix,iy,iz,α) = ix + (iy-1)*Nx + (iz-1)*Nx*Ny + (α-1)*Nx*Ny*Nz

    for iz in 1:Nz, iy in 1:Ny, ix in 1:Nx 
        i = f(ix,iy,iz,1)
        j = f(ix,iy,iz,3)
        ψ_reversed[i] =  ψ[j]

        i = f(ix,iy,iz,2)
        j = f(ix,iy,iz,4)
        ψ_reversed[i] = -ψ[j]

        i = f(ix,iy,iz,3)
        j = f(ix,iy,iz,1)
        ψ_reversed[i] = -ψ[j]

        i = f(ix,iy,iz,4)
        j = f(ix,iy,iz,2)
        ψ_reversed[i] =  ψ[j]
    end
end


function test_time_reversal(param; Nmax=[1,1])
    @unpack Nx, Ny, Nz = param 
    N = 4*Nx*Ny*Nz

    Vs, Ws = calc_potential(param, 0, 0)

    states = calc_states(param; Nmax=Nmax)
    @unpack nstates, ψs, spEs, qnums = states

    ψ_reversed = zeros(Float64, N)

    @views for istate in 1:nstates 
        ψ = ψs[:,istate]
        spE = spEs[istate]
        qnum = qnums[istate]

        @show 1
        @time time_reversal!(ψ_reversed, param, ψ)
        qnum_reversed = QuantumNumbers(qnum, η=-1)

        Hmat_reversed = make_Hamiltonian(param, Vs, Ws, qnum_reversed)
        spE_reversed = calc_sp_energy(Hmat_reversed, ψ_reversed)

        println("")
        @show spE spE_reversed dot(ψ_reversed, ψ)
    end

end