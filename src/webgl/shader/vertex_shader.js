let vertexShaderSource = 
`#version 300 es
precision highp float;

//input position
in vec3 inPosition;
//input texture coordinates
in vec2 inTexCoord;
//output texture coordinates
out vec2 texCoord;

//vertex shader
//The position and texture transfer unchagned
void main() {
    gl_Position = vec4(inPosition, 1.0);
    texCoord = inTexCoord;
}`

export default vertexShaderSource