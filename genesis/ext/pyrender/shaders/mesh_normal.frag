#version 330 core

in vec3 frag_position;
in vec3 frag_normal;

out vec4 frag_color;

void main()
{
    vec3 normal = normalize(frag_normal);

    frag_color = vec4(normal * 0.5 + 0.5, 1.0);
}