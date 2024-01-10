let source=
`#version 300 es
precision highp float;

//Matrix A
uniform sampler2D A;
//the A width
uniform int aWidth;
//Matrix B
uniform sampler2D B;
//the output height
uniform int outHeight;

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
    int outX = int(float(bSize.x) * texCoord.x);
    int outY = int(float(outHeight) * texCoord.y);


    //the B rows
    int bRows = bSize.y;

    float sum = 0.0;
    for(int i = 0; i < bRows; ++i) {
        int aRow = (aWidth * (outHeight - outY - 1) + i) / aSize.x;
        int aCol = (aWidth * (outHeight - outY - 1) + i) % aSize.x;

        float a = texelFetch(A, ivec2(aCol, outHeight - aRow - 1), 0).r;
        float b = texelFetch(B, ivec2(outX, i), 0).r;
        sum += a * b;
    }

    outValue = sum;
}`

export default source