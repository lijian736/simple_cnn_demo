let source=
`#version 300 es
precision highp float;

in vec2 texCoord;
uniform sampler2D A;
out float outValue;

void main() {
  //sampler, lod. return width, height
  ivec2 aSize = textureSize(A, 0);
  int outX = int(float(aSize[0]) * texCoord.x);
  int outY = int(float(aSize[1]) * texCoord.y);

  //sample, position, lod
  float val = texelFetch(A, ivec2(outX, outY), 0).r;

  outValue = val;
}`

export default source
