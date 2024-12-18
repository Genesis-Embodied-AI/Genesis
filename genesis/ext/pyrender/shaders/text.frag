#version 330 core
in vec2 uv;
out vec4 color;

uniform sampler2D text;
uniform vec4 text_color;

void main()
{
    vec4 sampled = vec4(1.0, 1.0, 1.0, texture(text, uv).r);
    color = text_color * sampled;
}
