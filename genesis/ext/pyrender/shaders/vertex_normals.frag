#version 330 core

out vec4 frag_color;

uniform vec4 normal_color;

void main()
{
    frag_color = normal_color;
}
