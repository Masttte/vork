// Inclusion template: SMOKE_DIMS (int3), SMOKE_CELL (float3 metres),
// SMOKE_ORIGIN (domain minimum, metres). X is contiguous. All grids share dims.
#ifndef SMOKE_GRID_INCLUDED
#define SMOKE_GRID_INCLUDED
uint smoke_index(int3 p) {
    p=clamp(p,int3(0),SMOKE_DIMS-1);
    return uint(p.x+SMOKE_DIMS.x*(p.y+SMOKE_DIMS.y*p.z));
}
int3 smoke_cell(uint id) {
    return int3(id%uint(SMOKE_DIMS.x),(id/uint(SMOKE_DIMS.x))%uint(SMOKE_DIMS.y),id/uint(SMOKE_DIMS.x*SMOKE_DIMS.y));
}
float3 smoke_position(float3 p) {return SMOKE_ORIGIN+(p+.5)*SMOKE_CELL;}
// Face component a is at the negative face of cell p. Positive domain faces
// are implicit zero; the stored negative boundary faces must also be zero.
bool smoke_wall(int3 p,int a) {return p[a]<=0 || p[a]>=SMOKE_DIMS[a];}
#endif
