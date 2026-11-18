#include "smoke/grid.slang"
float smoke_forced_face(int3 p,int a) {return smoke_wall(p,a)?0.:v2_load(smoke_index(p))[a];}
void smoke_divergence(ComputeContext ctx) {
    uint id=ctx.id.x;int3 p=smoke_cell(id);float divergence=0.;
    for(int a=0;a<3;a++) {
        int3 e=int3(0);e[a]=1;
        divergence+=(smoke_forced_face(p+e,a)-smoke_forced_face(p,a))/SMOKE_CELL[a];
    }
    rhs_store(id,divergence);pressure_store(id,0.);
}
