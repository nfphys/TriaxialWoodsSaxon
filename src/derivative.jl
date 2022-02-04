@inline function second_deriv_coeff(i, j, a, N, Π) 
    d = 0.0 
    if i === 1
        d += ifelse(j===3, -1/12, 0)
        d += ifelse(j===2,   4/3 + Π*(-1/12), 0)
        d += ifelse(j===1,  -5/2 + Π*(4/3), 0)
    elseif i === 2
        d += ifelse(j===4, -1/12, 0)
        d += ifelse(j===3, 4/3, 0)
        d += ifelse(j===2, -5/2, 0)
        d += ifelse(j===1, 4/3 + Π*(-1/12), 0)
    elseif i === N-1
        d += ifelse(j===N, 4/3, 0)
        d += ifelse(j===N-1, -5/2, 0)
        d += ifelse(j===N-2, 4/3, 0)
        d += ifelse(j===N-3, -1/12, 0)
    elseif i === N 
        d += ifelse(j===N, -5/2, 0)
        d += ifelse(j===N-1, 4/3, 0)
        d += ifelse(j===N-2, -1/12, 0)
    else
        d += ifelse(j===i+2, -1/12, 0)
        d += ifelse(j===i+1, 4/3, 0)
        d += ifelse(j===i, -5/2, 0)
        d += ifelse(j===i-1, 4/3, 0)
        d += ifelse(j===i-2, -1/12, 0)
    end
    d /= a*a 
    return d 
end


function first_deriv_coeff(i, j, a, N, Π)
    d = 0.0
    if i === 1
        d += ifelse(j===3, -1/12           , 0)
        d += ifelse(j===2,   2/3 + Π*(1/12), 0)
        d += ifelse(j===1,     0 + Π*(-2/3), 0)
    elseif i === 2
        d += ifelse(j===4, -1/12           , 0)
        d += ifelse(j===3,   2/3           , 0)
        d += ifelse(j===1,  -2/3 + Π*(1/12), 0)
    elseif i === N-1
        d += ifelse(j===N,    2/3, 0)
        d += ifelse(j===N-2, -2/3, 0)
        d += ifelse(j===N-3, 1/12, 0)
    elseif i === N 
        d += ifelse(j===N-1, -2/3, 0)
        d += ifelse(j===N-2, 1/12, 0)
    else
        d += ifelse(j===i+2, -1/12, 0)
        d += ifelse(j===i+1,   2/3, 0)
        d += ifelse(j===i-1,  -2/3, 0)
        d += ifelse(j===i-2,  1/12, 0)
    end
    d /= a
end


function test_first_deriv_coeff(param)
    @unpack Nx, Δx, xs = param 

    fs = @. exp(-xs^2)

    dfs = zeros(Float64, Nx)
    for ix in 1:Nx, dx in -2:2
        jx = ix + dx; if !(1 ≤ jx ≤ Nx) continue end
        dfs[ix] += first_deriv_coeff(ix, jx, Δx, Nx, 1)*fs[jx]
    end

    dfs_exact = @. -2xs*exp(-xs^2)

    p = plot()
    plot!(p, xs, fs)
    plot!(p, xs, dfs)
    plot!(p, xs, dfs_exact)
    display(p)

end


function first_derivative(Nx, Ny, Nz, Δx, Δy, Δz, fs)

    dfs = zeros(Float64, Nx, Ny, Nz, 3)

    N_diff = 2
    for iz in 1:Nz, iy in 1:Ny, ix in 1:Nx 
        # derivative w.r.t x
        for dx in -N_diff:N_diff 
            jx = ix+dx; if !(1 ≤ jx ≤ Nx) continue end
            jy = iy 
            jz = iz 

            c_ij = first_deriv_coeff(ix,jx,Δx,Nx,1)
            dfs[ix,iy,iz,1] += c_ij * fs[jx,jy,jz]
        end

        # derivative w.r.t y 
        for dy in -N_diff:N_diff 
            jx = ix
            jy = iy+dy; if !(1 ≤ jy ≤ Ny) continue end
            jz = iz 

            c_ij = first_deriv_coeff(iy,jy,Δy,Ny,1)
            dfs[ix,iy,iz,2] += c_ij * fs[jx,jy,jz]
        end

        # derivative w.r.t y 
        for dz in -N_diff:N_diff 
            jx = ix
            jy = iy
            jz = iz+dz; if !(1 ≤ jz ≤ Nz) continue end

            c_ij = first_deriv_coeff(iz,jz,Δz,Nz,1)
            dfs[ix,iy,iz,3] += c_ij * fs[jx,jy,jz]
        end
    end

    return dfs
end