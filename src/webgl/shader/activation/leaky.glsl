#version 300 es
precision highp float;

//the input position
in vec2 inPosition;
//the texture
uniform sampler2D x;
//the coefficient
uniform float alpha;
//the output
out vec4 outValue;

void main() {
  vec4 v4 = texture(x, vec2(inPosition.x, inPosition.y));
  outValue = max(v4, 0.0) + alpha * min(v4, 0.0);
}