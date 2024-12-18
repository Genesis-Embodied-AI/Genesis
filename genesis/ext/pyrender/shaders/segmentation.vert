#version 330 core
layout(location = 0) in vec3 position;
layout(location = INST_M_LOC) in mat4 inst_m;

uniform mat4 P;
uniform mat4 V;
uniform mat4 M;

void main()
{
    mat4 light_matrix = P * V;
    gl_Position = light_matrix * M * inst_m * vec4(position, 1.0);
}

