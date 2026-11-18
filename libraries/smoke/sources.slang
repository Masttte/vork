#include "smoke/grid.slang"
// The author supplies smoke_source(position, time, dt, density/heat/fuel, inputs).
// It returns the updated scalars; rates and combustion are scene choices.
void smoke_sources(ComputeContext ctx,Inputs inputs) {
    uint id=ctx.id.x;
    float3 value=smoke_source(smoke_position(float3(smoke_cell(id))),ctx.time,ctx.dt,
        float3(d1_load(id),t1_load(id),fuel1_load(id)),inputs);
    d_store(id,value.x);t_store(id,value.y);fuel_store(id,value.z);
}
