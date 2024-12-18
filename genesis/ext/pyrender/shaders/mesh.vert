#version 330 core

// Vertex Attributes
layout(location = 0) in vec3 position;
#ifdef NORMAL_LOC
layout(location = NORMAL_LOC) in vec3 normal;
#endif
#ifdef TANGENT_LOC
layout(location = TANGENT_LOC) in vec4 tangent;
#endif
#ifdef TEXCOORD_0_LOC
layout(location = TEXCOORD_0_LOC) in vec2 texcoord_0;
#endif
#ifdef TEXCOORD_1_LOC
layout(location = TEXCOORD_1_LOC) in vec2 texcoord_1;
#endif
#ifdef COLOR_0_LOC
layout(location = COLOR_0_LOC) in vec4 color_0;
#endif
#ifdef JOINTS_0_LOC
layout(location = JOINTS_0_LOC) in vec4 joints_0;
#endif
#ifdef WEIGHTS_0_LOC
layout(location = WEIGHTS_0_LOC) in vec4 weights_0;
#endif
layout(location = INST_M_LOC) in mat4 inst_m;

// Uniforms
uniform mat4 M;
uniform mat4 V;
uniform mat4 P;

uniform mat4 reflection_mat;

// Outputs
#ifdef DOUBLE_SIDED
    out vec3 v_frag_position;
    #ifdef NORMAL_LOC
    out vec3 v_frag_normal;
    #endif
    #ifdef HAS_NORMAL_TEX
    #ifdef TANGENT_LOC
    #ifdef NORMAL_LOC
    out mat3 v_tbn;
    #endif
    #endif
    #endif
    #ifdef TEXCOORD_0_LOC
    out vec2 v_uv_0;
    #endif
    #ifdef TEXCOORD_1_LOC
    out vec2 v_uv_1;
    #endif
    #ifdef COLOR_0_LOC
    out vec4 v_color_multiplier;
    #endif
#else
    out vec3 frag_position;
    #ifdef NORMAL_LOC
    out vec3 frag_normal;
    #endif
    #ifdef HAS_NORMAL_TEX
    #ifdef TANGENT_LOC
    #ifdef NORMAL_LOC
    out mat3 tbn;
    #endif
    #endif
    #endif
    #ifdef TEXCOORD_0_LOC
    out vec2 uv_0;
    #endif
    #ifdef TEXCOORD_1_LOC
    out vec2 uv_1;
    #endif
    #ifdef COLOR_0_LOC
    out vec4 color_multiplier;
    #endif
#endif


void main()
{
    gl_Position = P * V * reflection_mat * M * inst_m * vec4(position, 1);

#ifdef DOUBLE_SIDED
    v_frag_position = vec3(M * inst_m * vec4(position, 1.0));

    mat4 N = transpose(inverse(M * inst_m));

    #ifdef NORMAL_LOC
        v_frag_normal = normalize(vec3(N * vec4(normal, 0.0)));
    #endif

    #ifdef HAS_NORMAL_TEX
    #ifdef TANGENT_LOC
    #ifdef NORMAL_LOC
        vec3 normal_w = normalize(vec3(N * vec4(normal, 0.0)));
        vec3 tangent_w = normalize(vec3(N * vec4(tangent.xyz, 0.0)));
        vec3 bitangent_w = cross(normal_w, tangent_w) * tangent.w;
        v_tbn = mat3(tangent_w, bitangent_w, normal_w);
    #endif
    #endif
    #endif
    #ifdef TEXCOORD_0_LOC
        v_uv_0 = texcoord_0;
    #endif
    #ifdef TEXCOORD_1_LOC
        v_uv_1 = texcoord_1;
    #endif
    #ifdef COLOR_0_LOC
        v_color_multiplier = color_0;
    #endif
#else
    frag_position = vec3(M * inst_m * vec4(position, 1.0));

    mat4 N = transpose(inverse(M * inst_m));

    #ifdef NORMAL_LOC
        frag_normal = normalize(vec3(N * vec4(normal, 0.0)));
    #endif

    #ifdef HAS_NORMAL_TEX
    #ifdef TANGENT_LOC
    #ifdef NORMAL_LOC
        vec3 normal_w = normalize(vec3(N * vec4(normal, 0.0)));
        vec3 tangent_w = normalize(vec3(N * vec4(tangent.xyz, 0.0)));
        vec3 bitangent_w = cross(normal_w, tangent_w) * tangent.w;
        tbn = mat3(tangent_w, bitangent_w, normal_w);
    #endif
    #endif
    #endif
    #ifdef TEXCOORD_0_LOC
        uv_0 = texcoord_0;
    #endif
    #ifdef TEXCOORD_1_LOC
        uv_1 = texcoord_1;
    #endif
    #ifdef COLOR_0_LOC
        color_multiplier = color_0;
    #endif
#endif
}
