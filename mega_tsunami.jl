using Pkg
Pkg.activate(".")
Pkg.instantiate()

using CompScienceMeshes
using LinearAlgebra
using SparseArrays

fn = joinpath(@__DIR__, "mesh", "world.msh")
coast  = CompScienceMeshes.read_gmsh_mesh(fn, physical="Coast", dimension=1)
sea    = CompScienceMeshes.read_gmsh_mesh(fn, physical="Sea", dimension=2)

# skeleton creates lower dimensional meshes from a given mesh. With second argument
# zero the returned mesh is simply the cloud of vertices on which the original mesh
# was built.

coast_vertices  = skeleton(coast, 0)
sea_vertices    = skeleton(sea, 0)

# The FEM as presented here solved the homogenous Dirichlet problem for the Laplace
# equations. This means that we will not be associating basis functions with vertices
# on either boundary. After filtering out these vertices we are left with only
# interior vertices.

interior_vertices = submesh(sea_vertices) do v
    v in coast_vertices && return false
    return true
end


"""
Creates the local to global map for FEM assembly.

    localtoglobal(active_vertices, domain) -> gl

The returned map `gl` can be called as

    gl(k,p)

Here, `k` is an index into `domain` (i.e. it refers to a specific element, and
`p` is a local index into a specific element. It ranges from 1 to 3 for triangular
elements and from 1 to 2 for segments. The function returns an index `i` into
`active_vertices` if the i-th active vertex equals the p-th vertex of element k and
`gl` return `nothing` otherwise.
"""
function localtoglobal(active_vertices, domain)
    conn = copy(transpose(connectivity(active_vertices, domain, abs)))
    nz = nonzeros(conn)
    rv = rowvals(conn)
    function gl(k,p)
        for q in nzrange(conn,k)
            nz[q] == p && return rv[q]
        end
        return nothing
    end
    return gl
end


function elementmatrix(mesh, element)
    v1 = mesh.vertices[element[1]]
    v2 = mesh.vertices[element[2]]
    v3 = mesh.vertices[element[3]]
    
    global k # coefficient given by wave equation
    
    normal = (v1-v3) × (v2-v3)
    area = 0.5 * norm(normal)
    
    Δ_det=abs(det([v1 v2 v3]))
    
    # gradient of function
    
    grad1 = (v2 × v3) /  Δ_det
    grad2 = (v3 × v1) /  Δ_det
    grad3 = (v1 × v2) /  Δ_det

    # construct the elemetn matrix
    S = area * ([
        dot(grad1,grad1) dot(grad1,grad2) dot(grad1,grad3)
        dot(grad2,grad1) dot(grad2,grad2) dot(grad2,grad3)
        dot(grad3,grad1) dot(grad3,grad2) dot(grad3,grad3)].-k^2/12)
end

function assemblematrix(mesh, active_vertices)
    n = length(active_vertices)
    S = zeros(n,n)
    gl = localtoglobal(active_vertices, mesh)
    for (k,element) in enumerate(mesh)
        Sel = elementmatrix(mesh, element)
        for p in 1:3
            i = gl(k,p)
            i == nothing && continue
            for q in 1:3
                j = gl(k,q)
                j == nothing && continue
                S[i,j] += Sel[p,q]
            end
        end
    end

    return S
end


function elementvector(f, mesh, element)
    v1 = mesh.vertices[element[1]]
    v2 = mesh.vertices[element[2]]
    v3 = mesh.vertices[element[3]]
    el_size = norm((v1-v3)×(v2-v3))/2
    F = el_size * [
        f(v1)/3
        f(v2)/3
        f(v3)/3]
    return F
end


function assemblevector(f, mesh, active_vertices)
    n = length(active_vertices)
    F = zeros(n)
    gl = localtoglobal(active_vertices, mesh)
    for (k,element) in enumerate(mesh)
        Fel = elementvector(f,mesh,element)
        for p in 1:3
            i = gl(k,p)
            i == nothing && continue
            F[i] += Fel[p]
        end
    end

    return F
end

function latlon2xyz(loc)
    lat, lon = loc
    x = cos(lat)cos(lon)
    y = cos(lat)sin(lon)
    z = sin(lat)
    [x; y; z]
end

function xyz2latlon(coord)
    x, y, z = coord
    lat = asin(z)
    lon = atan(y, x)
    [lat; lon]
end

# function spherical_dist(loc1::Array{Number,1},loc2::Array{Number,1})
#     # loc:{lat ϕ,lon λ}
#     ϕ1,λ1=loc1
#     ϕ2,λ2=loc2
#     Δλ=abs(λ1-λ2)
#     acos(sin(ϕ1)sin(ϕ2)+cos(ϕ1)cos(ϕ2)cos(Δλ))
# end

function spherical_dist(loc1,loc2)
    # loc:{x,y,z}
    abs(acos(dot(normalize(loc1),normalize(loc2))))
end


function f(coord, epicenter, A, σ, R)
    loc1=coord
    loc2= R*latlon2xyz(epicenter)
    d=spherical_dist(loc1,loc2)*R
    
    A*exp(-1/2*d^2/σ^2)
end

f(x)=f(x, epicenter, A, σ, R)


# For the assignment of the lab, i.e. the Helmholtz equations (aka the frequency
# domain wave equation), subject to absorbing boundary conditions, you will also
# have to include a term stemming from boundary integral contributions. For that
# term a different local-to-global matrix is required: one linking segments on the
# boundary to indices of active vertices. You can create this map using the same
# function, i.e. like:
#
#   gl = localtoglobal(active_vertices, border)
#
R=6371*1000
# somewhere at North Atlantic
lat=40.4/180*π
lon=-43.2/180*π

epicenter=[lat,lon]
A=100 # strength of tsunami
σ=100*1000 # decay rate, 400km

k=2π/(4e6)

S = assemblematrix(sea, interior_vertices)
F = assemblevector(f, sea, interior_vertices)
u = S \ F

u_tilda = zeros(length(sea_vertices))
for (j,m) in enumerate(interior_vertices)
    u_tilda[m[1]] = u[j]
end

using Makie

scene= Scene(resolution=(1000,1000),show_axis=false)
Makie.mesh!(scene,vertexarray(sea), cellarray(sea),color=u_tilda)
