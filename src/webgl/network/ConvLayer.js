import Layer from './Layer'
import Tensor from './Tensor'
import CPUMath from '../backend/cpu/CPUMath'
import { Conv as WebGLConvOp } from '../backend//webgl/operator/Conv'


export default class ConvLayer extends Layer {
    /**
     * constructor
     * @param {Object} options 
     */
    constructor(options = {}) {
        super('conv', options);

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

        //CHW
        this._outputSize = [this._filters, Math.floor((this._inputSize[1] + 2 * this._paddings[0] - this._kernelSize[1]) / this._strides[0] + 1), Math.floor((this._inputSize[2] + 2 * this._paddings[1] - this._kernelSize[2]) / this._strides[1] + 1)];

        //output buffer
        this._output = new Float32Array(this._outputSize.reduce((previous, current, index, array) => { previous * current }, 1));

        //webgl-accelerated computing
        if (this.useWebGL) {
            this._webglConvOp = new WebGLConvOp(options);
        }
    }

    /**
     * set weights
     * @param {Tensor} newValue the Tensor object
     */
    set weights(newValue) {
        if (this.useWebGL) {
            this._webglConvOp.kernels = newValue;
        }
    }

    /**
     * set bias
     * @param {Tensor} newValue the Tensor object
     */
    set bias(newValue) {
        if (this.useWebGL) {
            this._webglConvOp.bias = newValue;
        }
    }

    /**
     * forward
     * @param {Tensor} input the input tensor
     * @param {*} network the CNN network 
     */
    forward(input, network) {
        let B = network.workspace;
        //compute C = A * B
        if (this._size === 1) {
            B = input.data;
        } else {
            const tsStart = Math.round(new Date());
            CPUMath.im2col(input.data, this._inputSize[0], this._inputSize[1], this._inputSize[2], this._kernelSize[1], this._strides[0], this._paddings[0], B);
            const tsEnd = Math.round(new Date());

            console.log('im2col consumes: %d milliseconds', tsEnd - tsStart);
        }

        this._webglConvOp.run(input, this._output);
    }
}