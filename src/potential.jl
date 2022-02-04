function calc_potential(param, β, γ)
    @unpack Nx, Ny, Nz, Δx, Δy, Δz, xs, ys, zs, 
    V₀, r₀, R₀, a, κ = param 

    fs = zeros(Float64, Nx, Ny, Nz)

    βcosγ = β*cos(γ)
    βsinγ = β*sin(γ)

    N20 = √(5/16π)
    N22 = √(15/32π)

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
        R = R₀*(1 + βcosγ*Y20 + √2*βsinγ*Y22_Re)

        fs[ix,iy,iz] = 1/(1 + exp((r-R)/a))
    end

    dfs = first_derivative(Nx, Ny, Nz, Δx, Δy, Δz, fs)

    Vs = @. V₀*fs 
    Ws = @. κ*V₀*r₀*r₀*dfs

    return Vs, Ws
end



function test_calc_potential(param, β, γ)
    @unpack xs, ys, zs = param

    @time Vs, Ws = calc_potential(param, β, γ)

    p = heatmap(xs, ys, Vs[:,:,1]', xlabel="x [fm]", ylabel="y [fm]", ratio=:equal)
    display(p)

    p = heatmap(xs, zs, Vs[:,1,:]', xlabel="x [fm]", ylabel="z [fm]", ratio=:equal)
    display(p)

    p = heatmap(ys, zs, Vs[1,:,:]', xlabel="y [fm]", ylabel="z [fm]", ratio=:equal)
    display(p)

    for i in 1:3
        p = heatmap(xs, ys, Ws[:,:,1,i]', xlabel="x [fm]", ylabel="y [fm]", ratio=:equal)
        display(p)

        p = heatmap(xs, zs, Ws[:,1,:,i]', xlabel="x [fm]", ylabel="z [fm]", ratio=:equal)
        display(p)

        p = heatmap(ys, zs, Ws[1,:,:,i]', xlabel="y [fm]", ylabel="z [fm]", ratio=:equal)
        display(p)
    end
end