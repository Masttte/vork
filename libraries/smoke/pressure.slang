#include "smoke/grid.slang"
// Red/black SOR; a complete dispatch barrier is required between colours.
// Pressure stores dt*p/rho. The Neumann box has a constant nullspace, which
// need not be pinned: projection uses only differences. The RHS sums to zero
// because it is formed from the same wall fluxes as projection.
void smoke_pressure(ComputeContext ctx,int colour,float relaxation) {
    uint id=ctx.id.x;int3 p=smoke_cell(id);
    if(((p.x+p.y+p.z)&1)!=colour)return;
    float sum=0.,diagonal=0.;
    for(int a=0;a<3;a++) {
        int3 e=int3(0);e[a]=1;float w=1./(SMOKE_CELL[a]*SMOKE_CELL[a]);
        if(p[a]>0){sum+=w*pressure_load(smoke_index(p-e));diagonal+=w;}
        if(p[a]+1<SMOKE_DIMS[a]){sum+=w*pressure_load(smoke_index(p+e));diagonal+=w;}
    }
    pressure_store(id,lerp(pressure_load(id),(sum-rhs_load(id))/diagonal,relaxation));
}
