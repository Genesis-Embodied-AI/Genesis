#version 330 core
//layout (location = 0) in vec3 aPos;
//layout (location = 1) in vec2 aTexCoords;
//
//out vec2 TexCoords;
//
//void main()
//{
//    TexCoords = aTexCoords;
//    gl_Position = vec4(aPos, 1.0);
//}
//
//
//layout(location = 0) out vec2 uv;

out vec2 TexCoords;

void main() 
{
    float x = float(((uint(gl_VertexID) + 2u) / 3u)%2u); 
    float y = float(((uint(gl_VertexID) + 1u) / 3u)%2u); 

    gl_Position = vec4(-1.0f + x*2.0f, -1.0f+y*2.0f, 0.0f, 1.0f);
    TexCoords = vec2(x, y);
}
