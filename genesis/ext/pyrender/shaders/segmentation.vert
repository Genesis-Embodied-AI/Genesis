#version 330 core

// Vertex Attributes
layout(location = 0) in vec3 position;
#ifdef NORMAL_LOC
layout(location = NORMAL_LOC) in vec3 normal;
#endif
#ifdef COLOR_0_LOC
layout(location = COLOR_0_LOC) in vec4 color_0;
#endif
layout(location = INST_M_LOC) in mat4 inst_m;

// Uniforms
uniform mat4 P;
uniform mat4 V;
uniform mat4 M;

// Outputs
#ifdef DOUBLE_SIDED
    out vec3 v_frag_position;
    #ifdef NORMAL_LOC
    out vec3 v_frag_normal;
    #endif
    #ifdef COLOR_0_LOC
    out vec4 v_color_multiplier;
    #endif
#endif

void main()
{
    mat4 light_matrix = P * V;
    gl_Position = light_matrix * M * inst_m * vec4(position, 1.0);

#ifdef DOUBLE_SIDED
    v_frag_position = vec3(M * inst_m * vec4(position, 1.0));

    #ifdef NORMAL_LOC
        mat4 N = transpose(inverse(M * inst_m));

        v_frag_normal = normalize(vec3(N * vec4(normal, 0.0)));
    #endif

    #ifdef COLOR_0_LOC
        v_color_multiplier = color_0;
    #endif
#endif
}

