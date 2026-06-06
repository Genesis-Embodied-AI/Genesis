#version 330 core

// Vertex Attributes
layout(location = 0) in vec3 position;
layout(location = NORMAL_LOC) in vec3 normal;
layout(location = INST_M_LOC) in mat4 inst_m;

// Uniforms
uniform mat4 M;
uniform mat4 V;
uniform mat4 P;

// Outputs
#ifdef DOUBLE_SIDED
    out vec3 v_frag_position;
    out vec3 v_frag_normal;
#else
    out vec3 frag_position;
    out vec3 frag_normal;
#endif

void main()
{
    gl_Position = P * V * M * inst_m * vec4(position, 1);
    mat4 N = transpose(inverse(M * inst_m));

#ifdef DOUBLE_SIDED
    v_frag_position = vec3(M * inst_m * vec4(position, 1.0));
    v_frag_normal = normalize(vec3(N * vec4(normal, 0.0)));
#else
    frag_position = vec3(M * inst_m * vec4(position, 1.0));
    frag_normal = normalize(vec3(N * vec4(normal, 0.0)));
#endif
}