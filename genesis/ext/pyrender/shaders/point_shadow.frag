#version 330 core

in vec3 frag_pos;
uniform vec3 light_pos;

void main()
{
    float dis = length(frag_pos - light_pos);
    gl_FragDepth = dis/25.0;
}
