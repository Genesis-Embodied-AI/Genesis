#version 330 core
layout(location = 0) in vec3 position;
layout(location = INST_M_LOC) in mat4 inst_m;

uniform mat4 M;

void main()
{
    gl_Position = M * inst_m * vec4(position, 1.0);
}
