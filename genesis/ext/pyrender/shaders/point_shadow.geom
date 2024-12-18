#version 330 core
layout (triangles) in;
layout (triangle_strip, max_vertices=18) out;

uniform mat4 light_matrix[6];
out vec3 frag_pos;

void main() {
    for (int i = 0; i < 6; i++) {
        gl_Layer = i;
        for (int j = 0 ; j < 3; j++) {
            frag_pos = gl_in[j].gl_Position.xyz;
            gl_Position = light_matrix[i] * gl_in[j].gl_Position;
            EmitVertex();
        }
        EndPrimitive();
    }
}
