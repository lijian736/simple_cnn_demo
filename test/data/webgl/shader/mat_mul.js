let source=
`#version 300 es
precision highp float;

//Matrix A
uniform sampler2D A;
//the A width
uniform int aWidth;
//the A height
uniform int aHeight;

//Matrix B
uniform sampler2D B;
//the B width
uniform int bWidth;
//the B height
uniform int bHeight;

//the texture coordinates
in vec2 texCoord;
//the output values
out float outValue;

//Matrix Multiplication:  return A * B
void main() {
    //retrieve the size of texture A
    ivec2 aSize = textureSize(A, 0);
    //retrieve the size of texture B
    ivec2 bSize = textureSize(B, 0);

    //left-bottom is origin, X to right, Y to top
    int outX = int(float(bWidth)  * texCoord.x);
    int outY = int(float(aHeight) * texCoord.y);

    float sum = 0.0;
    for(int i = 0; i < aWidth; ++i) {
        int aRow = (aWidth * (aHeight - outY - 1) + i) / aSize.x;
        int aCol = (aWidth * (aHeight - outY - 1) + i) % aSize.x;

        int bRow = (bWidth * i + outX) / bSize.x;
        int bCol = (bWidth * i + outX) % bSize.x;

        float a = texelFetch(A, ivec2(aCol, aHeight - aRow - 1), 0).r;
        float b = texelFetch(B, ivec2(bCol,               bRow), 0).r;
        sum += a * b;
    }

    outValue = sum;
}`

export default source