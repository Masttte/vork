#include "smoke/grid.slang"
// d,t,f: cell-centred scalar generations. v: negative MAC face velocities.
// d1,t1,f1,v1: distinct outputs. No in-place neighbour updates.
float smoke_face(int3 p,int a) {
    return smoke_wall(p,a)?0.:v_load(smoke_index(p))[a];
}
float smoke_sample_face(float3 q,int a) {
    q[a]+=.5;
    int3 lo=int3(floor(q));float3 f=q-float3(lo);float value=0.;
    for(int z=0;z<2;z++) for(int y=0;y<2;y++) for(int x=0;x<2;x++) {
        int3 b=int3(x,y,z);float3 w=lerp(1.-f,f,float3(b));
        value+=w.x*w.y*w.z*smoke_face(lo+b,a);
    }
    return value;
}
float3 smoke_velocity(float3 q) {
    return float3(smoke_sample_face(q,0),smoke_sample_face(q,1),smoke_sample_face(q,2));
}
float3 smoke_departure(float3 q,float dt) {
    float3 half=q-.5*dt*smoke_velocity(q)/SMOKE_CELL;
    return q-dt*smoke_velocity(half)/SMOKE_CELL;
}
void smoke_advect(ComputeContext ctx) {
    uint id=ctx.id.x;int3 p=smoke_cell(id);
    float3 q=smoke_departure(float3(p),ctx.dt);
    int3 lo=int3(floor(q));float3 f=q-float3(lo),value=0.;
    for(int z=0;z<2;z++) for(int y=0;y<2;y++) for(int x=0;x<2;x++) {
        int3 b=int3(x,y,z);float3 w=lerp(1.-f,f,float3(b));uint j=smoke_index(lo+b);
        value+=w.x*w.y*w.z*float3(d_load(j),t_load(j),fuel_load(j));
    }
    d1_store(id,value.x);t1_store(id,value.y);fuel1_store(id,value.z);
    float3 velocity=0.;
    for(int a=0;a<3;a++) {
        float3 face=float3(p);face[a]-=.5;
        if(!smoke_wall(p,a))velocity[a]=smoke_sample_face(smoke_departure(face,ctx.dt),a);
    }
    v1_store(id,velocity);
}
