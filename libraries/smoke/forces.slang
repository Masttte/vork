#include "smoke/grid.slang"
// Authored smoke_force(position, time, density/heat/fuel, inputs) returns m/s².
// Evaluate at each MAC face with the adjacent cell-centred scalars averaged.
float3 smoke_quantities(int3 p) {
    uint j=smoke_index(p);return float3(d_load(j),t_load(j),fuel_load(j));
}
void smoke_forces(ComputeContext ctx,Inputs inputs) {
    uint id=ctx.id.x;int3 p=smoke_cell(id);float3 velocity=v1_load(id);
    for(int a=0;a<3;a++) {
        int3 e=int3(0);e[a]=1;float3 q=float3(p);q[a]-=.5;
        float3 fields=.5*(smoke_quantities(p)+smoke_quantities(p-e));
        velocity[a]=smoke_wall(p,a)?0.:velocity[a]+ctx.dt*smoke_force(smoke_position(q),ctx.time,fields,inputs)[a];
    }
    v2_store(id,velocity);
}
