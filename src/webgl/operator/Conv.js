import ConvShaderSource from '../shader/convolution.glsl'
import ConvShaderSource1 from '../shader/convolution_1.glsl'
import LeakyShaderSource from '../shader/activation/leaky.glsl'

import webgl2 from '../WebGL2';
import Tensor from '../network/Tensor'

/**
 * The WebGL convolution operator
 * Since the WebGL2 texture size constraits,
 * 
 * 1. We describe the input image as a WebGL2 3D texture with size [width, height, depth(Channel)];
 * 2. We describe the kernels as a WebGL2 3D texture with size [kernelWidth * kernelHeight, kernelChannels, filters number];
 * 3. the output was described as a WebGL2 2D texture with size [outputWidth * outputHeight, depth] or [outputWidth, outputHeight * depth];
 *    the depth is filter number.
 */
export default class Conv {
    /**
     * constructor
     */
    constructor() {
        const {
            filters = 1,  //filter number
            kernelSize = [1, 1, 1],  //CHW
            inputSize = [0, 0, 0], //CHW
            strides = [1, 1], //strides in HW
            paddings = [0, 0], //padding in HW
            activation = 'leaky', //activate type
            useBias = false  //bias
        } = options;

        this._filters = filters;
        this._kernelSize = kernelSize;
        this._inputSize = inputSize;
        this._strides = strides;
        this._paddings = paddings;
        this._activation = activation;
        this._useBias = useBias;

        //output image height
        const outHeight = Math.floor((inputSize[1] + 2 * paddings[0] - kernelSize[1]) / strides[0] + 1);
        //output image width
        const outWidth = Math.floor((inputSize[2] + 2 * paddings[1] - kernelSize[2]) / strides[1] + 1);

        //the output shape in CHW format
        this._outputShape = [filters, outHeight, outWidth];

        //WebGL2 context
        this._context = webgl2;
        //create the activation program
        this._activationProgram = webgl2.createProgram(LeakyShaderSource);

        //the convolution output
        this._outputData = new Float32Array(this._outputShape.reduce((prev, curr) => prev * curr, 1));
        //create the output texture
        this._createOutputTexture();

        if(this._outputLayout === "WH_D"){
            //create the convolution WebGL2 program
            this._convProgram = webgl2.createProgram(ConvShaderSource);
        }else{
            this._convProgram = webgl2.createProgram(ConvShaderSource1);
        }
    }

    /**
     * create the convolution output texture
     */
    _createOutputTexture() {
        let gl = this._context;
        let outputWidth = this._outputShape[1] * this._outputShape[2];
        let outputHeight = this._outputShape[0];

        //the output layout, width height and depth
        this._outputLayout = "WH_D";

        if (outputWidth > gl.MAX_TEXTURE_SIZE) {
            outputWidth = this._outputShape[2];
            outputHeight = this._outputShape[0] * this._outputShape[1];
            this._outputLayout = "W_HD";
        }

        if (outputWidth > gl.MAX_TEXTURE_SIZE || outputHeight > gl.MAX_TEXTURE_SIZE) {
            throw new Error('the output texture size exceeds the max size');
        }

        //the output texture width
        this._outputTextureWidth = outputWidth;
        //the output texture height
        this._outputTextureHeight = outputHeight;
        //the output texture
        this._outputTexture = gl.create2DTexture({ textureWidth: outputWidth, textureHeight: outputHeight, textureData: this._outputData });
    }

    /**
     * set weights
     * @param {Tensor} tensor the weights value
     */
    set weights(tensor) {
        //check the input tensor shape
        if (tensor.shape.length !== 4 || tensor.shape[0] !== this._filters || tensor.shape[1] !== this._kernelSize[0] ||
            tensor.shape[2] !== this._kernelSize[1] || tensor.shape[3] !== this._kernelSize[2] || !(tensor.data instanceof Float32Array)) {
            throw new Error('The shape of weights mismatches or the tensor data type mismatches');
        }

        let gl = this._context;
        //create the weights texture
        this._kernelTextures = gl.create2DArrayTexture({ textureWidth: this._kernelSize[1] * this._kernelSize[2], textureHeight: this._kernelSize[0], textureDepth: this._filters, textureData: tensor.data });
    }

    /**
     * set bias
     * @param {Tensor} tensor the bias value
     */
    set bias(tensor) {
        if (this._useBias) {
            if (tensor.shape.length !== 1 || tensor.shape[0] != this._filters) {
                throw new Error('The shape of bias mismatches');
            }

            let gl = this._context;
            //create the bias texture
            this._biasTexture = gl.createTexture({ textureWidth: this._filters, textureHeight: 1, textureData: tensor.data });
        } else {
            throw new Error('The Conv DO NOT need bias');
        }
    }

    /**
     * run the convolution
     * @param {Tensor} tensor the input data 
     * @param {Float32Array} auxSpace the auxiliary memory
     */
    run(tensor, auxSpace) {
        //image to column, and returns the output image width and height
        //[width, height] = CPUMath.im2col(tensor.data, this._inputSize[0], this._inputSize[1], this._inputSize[2], this._kernelSize[1], this._kernelSize[2], this._strides[0], this._strides[1], this._paddings[0], this._paddings[1], auxSpace);
        //the kernel length in WebGL shader
        //const kernelLength = width;
        //the tiling in width
        //let tiling = 1;
        //while(height > gl.MAX_TEXTURE_SIZE){
        // height = Math.ceil(height / 2);
        // width *= 2;
        // tiling *= 2;
        //}

        let gl = this._context;
        //create the texture
        let dataTexture = gl.create2DArrayTexture({ textureWidth: tensor.shape[2], textureHeight: tensor.shape[1], textureDepth: tensor.shape[0], textureData: tensor.data });

        //run the convolution
        gl.run({
            program: this._convProgram,
            inputs: [{ texture: this._kernelTextures, name: 'kernel' }, { texture: dataTexture, name: 'input' }],
            uniforms: [{ name: 'widthTile', type: 'int', data: tiling }, { name: 'kernelLength', type: 'int', data: kernelLength }],
            output: { texture: this._outputTexture, width: this._outputTextureWidth, height: this._outputTextureHeight }
        });

        if (this._activationProgram) {

        }
    }
}