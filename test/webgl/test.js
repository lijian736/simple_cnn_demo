import { webgl2 } from '../../src/webgl/WebGL2.js'
import CNN from '../../src/network.js'
import naiveSource from '../data/webgl/shader/naive.js'
import identitySource from '../data/webgl/shader/identity.js'
import naiveMatMulSource from '../data/webgl/shader/naive_mat_mul.js'
import aMatMulSource from '../data/webgl/shader/a_mat_mul.js'
import matMulSource from '../data/webgl/shader/mat_mul.js'
import matMulOutLayoutSource from '../data/webgl/shader/mat_mul_out_layout.js'


describe("WebGL2 Context", () => {
    it("context", () => {
        console.log('MAX_TEXTURE_SIZE: %d for 1D or 2D textures', webgl2.MAX_TEXTURE_SIZE);
        console.log('MAX_TEXTURE_IMAGE_UNITS: %d', webgl2.MAX_TEXTURE_IMAGE_UNITS);
        console.log('MAX_3D_TEXTURE_SIZE: %d for 3D textures', webgl2.MAX_3D_TEXTURE_SIZE);
        console.log('MAX_ARRAY_TEXTURE_LAYERS: %d for texture arrays', webgl2.MAX_ARRAY_TEXTURE_LAYERS);
        console.log('MAX_VERTEX_UNIFORM_COMPONENTS: %d', webgl2.MAX_VERTEX_UNIFORM_COMPONENTS);
        console.log('MAX_VERTEX_UNIFORM_VECTORS: %d', webgl2.MAX_VERTEX_UNIFORM_VECTORS);
        console.log('MAX_FRAGMENT_UNIFORM_COMPONENTS: %d', webgl2.MAX_FRAGMENT_UNIFORM_COMPONENTS);
        console.log('MAX_FRAGMENT_UNIFORM_VECTORS: %d', webgl2.MAX_FRAGMENT_UNIFORM_VECTORS);
    });

    it("Create Program", () => {
        let convProgram = webgl2.createProgram(naiveSource);
        console.log(convProgram)
    });

    it("Create Texture", () => {
        let outputData = new Float32Array(3 * 3);
        let texture = webgl2.create2DTexture({ textureWidth: 3, textureHeight: 3, textureData: outputData });
    });

    it("Naive Output", () => {
        let naiveProgram = webgl2.createProgram(naiveSource);

        let outputData = webgl2.run({
            program: naiveProgram,
            inputs: [],
            uniforms: [],
            output: { width: 3, height: 3 }
        });

        console.log('output length: %d', outputData.length);
        console.log('output: %f    %f    %f', outputData[0], outputData[1], outputData[2]);
    });

    it("Identity Output", () => {
        let identityProgram = webgl2.createProgram(identitySource);

        let inputData = new Float32Array(4 * 4);
        for(let i = 0; i < inputData.length; ++i){
            inputData[i] = i + 1;
        }

        let inputTexture = webgl2.create2DTexture({ textureWidth: 4, textureHeight: 4, textureData: inputData });
        inputData = null;

        let outputData = webgl2.run({
            program: identityProgram,
            inputs: [{texture: inputTexture, name: 'A'}],
            uniforms: [],
            output: { width: 4, height: 4 }
        });

        console.log('output length: %d', outputData.length);
        console.log('output: %f    %f    %f    %f', outputData[0], outputData[1], outputData[2], outputData[3]);
        console.log('output: %f    %f    %f    %f', outputData[4], outputData[5], outputData[6], outputData[7]);
        console.log('output: %f    %f    %f    %f', outputData[8], outputData[9], outputData[10], outputData[11]);
        console.log('output: %f    %f    %f    %f', outputData[12], outputData[13], outputData[14], outputData[15]);
    });

    it("Naive MatMul Output", () => {
        let naiveMatmulProgram = webgl2.createProgram(naiveMatMulSource);

        let inputData = new Float32Array(3 * 3);
        for(let i = 0; i < inputData.length; ++i){
            inputData[i] = i + 1;
        }

        let inputTexture1 = webgl2.create2DTexture({ textureWidth: 3, textureHeight: 3, textureData: inputData });
        inputData = null;

        inputData = new Float32Array(3 * 3);
        for(let i = 0; i < inputData.length; ++i){
            inputData[i] = i + 1;
        }

        let inputTexture2 = webgl2.create2DTexture({ textureWidth: 3, textureHeight: 3, textureData: inputData });
        inputData = null;

        let outputData = webgl2.run({
            program: naiveMatmulProgram,
            inputs: [{texture: inputTexture1, name: 'A'}, {texture: inputTexture2, name: 'B'}],
            uniforms: [],
            output: { width: 3, height: 3 }
        });

        console.log('output length: %d', outputData.length);
        console.log('output: %f    %f    %f', outputData[0], outputData[1], outputData[2]);
        console.log('output: %f    %f    %f', outputData[3], outputData[4], outputData[5]);
        console.log('output: %f    %f    %f', outputData[6], outputData[7], outputData[8]);
    });

    it("MatMul Output A", () => {
        let matmulProgram = webgl2.createProgram(aMatMulSource);

        let inputData = new Float32Array(3 * 3);
        for(let i = 0; i < inputData.length; ++i){
            inputData[i] = i + 1;
        }

        let inputTexture1 = webgl2.create2DTexture({ textureWidth: 3, textureHeight: 3, textureData: inputData });
        inputData = null;

        inputData = new Float32Array(3 * 3);
        for(let i = 0; i < inputData.length; ++i){
            inputData[i] = i + 1;
        }

        let inputTexture2 = webgl2.create2DTexture({ textureWidth: 3, textureHeight: 3, textureData: inputData });
        inputData = null;

        let outputData = webgl2.run({
            program: matmulProgram,
            inputs: [{texture: inputTexture1, name: 'A'}, {texture: inputTexture2, name: 'B'}],
            uniforms: [{ name: "outHeight", type: "int", data: 3}, { name: "aWidth", type: "int", data: 3}],
            output: { width: 3, height: 3 }
        });

        console.log('output length: %d', outputData.length);
        console.log('output: %f    %f    %f', outputData[0], outputData[1], outputData[2]);
        console.log('output: %f    %f    %f', outputData[3], outputData[4], outputData[5]);
        console.log('output: %f    %f    %f', outputData[6], outputData[7], outputData[8]);
    });

    it("MatMul Output", () => {
        let matmulProgram = webgl2.createProgram(matMulSource);

        let inputData = new Float32Array(4 * 4);
        for(let i = 0; i < inputData.length; ++i){
            inputData[i] = i + 1;
        }

        let inputTexture1 = webgl2.create2DTexture({ textureWidth: 4, textureHeight: 4, textureData: inputData });
        inputData = null;

        inputData = new Float32Array(4 * 4);
        for(let i = 0; i < inputData.length; ++i){
            inputData[i] = i + 1;
        }

        let inputTexture2 = webgl2.create2DTexture({ textureWidth: 4, textureHeight: 4, textureData: inputData });
        inputData = null;

        let outputData = webgl2.run({
            program: matmulProgram,
            inputs: [{texture: inputTexture1, name: 'A'}, {texture: inputTexture2, name: 'B'}],
            uniforms: [{ name: "aWidth", type: "int", data: 4}, { name: "aHeight", type: "int", data: 4},
                       { name: "bWidth", type: "int", data: 4}, { name: "bHeight", type: "int", data: 4}],
            output: { width: 4, height: 4 }
        });

        console.log('output length: %d', outputData.length);
        console.log('output: %f', outputData[0]);
    });

    it("MatMul Out Layout", () => {
        let A = new Float32Array(1);
        A[0] = 2;

        let B = new Float32Array(webgl2.MAX_TEXTURE_SIZE + 1);
        B.forEach((value, index, array)=>{
            array[index] = 1;
        });

        let result = CNN.Math.gemmAccelerate(webgl2, A, B, 1, 1, webgl2.MAX_TEXTURE_SIZE + 1, 1);
        console.log('output length:%d', result.length);
        console.log('result[0]: %f', result[0]);
        console.log('result[-1]: %f', result[webgl2.MAX_TEXTURE_SIZE]);
    });
});

