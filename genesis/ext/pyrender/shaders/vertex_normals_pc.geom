#version 330 core

layout (points) in;

layout (line_strip, max_vertices = 2) out;

in VS_OUT {
    vec3 position;
    vec3 normal;
    mat4 mvp;
} gs_in[];

uniform float normal_magnitude;

void GenerateVertNormal(int index)
{
    vec4 p0 = gs_in[index].mvp * vec4(gs_in[index].position, 1.0);
    vec4 p1 = gs_in[index].mvp * vec4(normal_magnitude * normalize(gs_in[index].normal) + gs_in[index].position, 1.0);
    gl_Position = p0;
    EmitVertex();
    gl_Position = p1;
    EmitVertex();
    EndPrimitive();
}

void main()
{
    GenerateVertNormal(0);
}
