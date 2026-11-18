#include "smoke/grid.slang"
void smoke_project(ComputeContext ctx) {
    uint id=ctx.id.x;int3 p=smoke_cell(id);float3 velocity=v2_load(id);
    for(int a=0;a<3;a++) {
        int3 e=int3(0);e[a]=1;
        velocity[a]=smoke_wall(p,a)?0.:velocity[a]-(pressure_load(id)-pressure_load(smoke_index(p-e)))/SMOKE_CELL[a];
    }
    v_store(id,velocity);
}
