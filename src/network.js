import matMulSource from './webgl/shader/mat_mul.js'

var CNN = CNN || {};

/**
 * Math class for CNN
 */
CNN.Math = class {
    /**
     * Get pixel of the specified image, the image layout is CHW
     * @param {Float32Array} img the image data
     * @param {Number} height the image height(actual height)
     * @param {Number} width the image width(actual width)
     * @param {Number} channels the image channels(actual channels)
     * @param {Number} row the row index(including padding)
     * @param {Number} col the column index, including padding
     * @param {Number} channel the channel index
     * @param {Number} pad the padding value
     * @returns the pixel value
     */
    static im2colGetPixel(img, height, width, channels, row, col, channel, pad) {
        // get the real row and column
        row -= pad;
        col -= pad;

        // if padding ,return 0
        if (row < 0 || col < 0 || row >= height || col >= width) {
            return 0;
        }

        return img[col + width * (row + height * channel)];
    }

    /**
     * image to column.
     * for example:
     * if the input image is(row-first):
     * [ 1 2 3 4
     *   7 9 5 8
     *   6 2 1 7
     *   2 8 9 1]
     * kernel dimension is [1 2 2] in layout CHW.
     * if the padding is 0 and stride is 1,
     * then the output is(row-first):
     * [1 2 3 7 9 5 6 2 1
     *  2 3 4 9 5 8 2 1 7
     *  7 9 5 6 2 1 2 8 9
     *  9 5 8 2 1 7 8 9 1]
     * @param {Float32Array} img the image data in CHW
     * @param {Number} channels the image channels
     * @param {Number} height the image height
     * @param {Number} width the image width
     * @param {Number} kernelSize the kernel size
     * @param {Number} stride the stride
     * @param {Number} pad padding
     * @param {Float32Array} outputData the output float32 array
     */
    static im2col(img, channels, height, width, kernelSize, stride, pad, outputData) {
        //output image height
        const heightCol = Math.floor((height + 2 * pad - kernelSize) / stride + 1);
        //output image width
        const widthCol = Math.floor((width + 2 * pad - kernelSize) / stride + 1);

        //every column-image pixels number
        const columns = channels * kernelSize * kernelSize;
        for (let c = 0; c < columns; c++) {
            //the pixel column offset in the kernel
            const colOffset = c % kernelSize;
            //the pixel row offset in the kernel
            const rowOffset = Math.floor(c / kernelSize) % kernelSize;
            //the pixel channel offset in the kernel
            const channelOffset = Math.floor(c / (kernelSize * kernelSize));

            for (let h = 0; h < heightCol; h++) {
                //the row coordinate in image
                const imgRow = rowOffset + h * stride;

                for (let w = 0; w < widthCol; w++) {
                    //the column coordinate in image
                    const imgCol = colOffset + w * stride;
                    //the output image index, row-first index
                    const outIndex = (c * heightCol + h) * widthCol + w;
                    outputData[outIndex] = CNN.Math.im2colGetPixel(img, height, width, channels, imgRow, imgCol, channelOffset, pad);
                }
            }
        }
    }

    /**
     * General Matrix Multiplication
     * C = alpha * A * B^T
     * A dimensions: M x K
     * B^T dimensions: K x N
     * C dimensions: M x N
     * 
     * @param {Number} M the row of A and C
     * @param {Number} N the column of B^T and C
     * @param {Number} K the row of B^T, or column of A.
     * @param {Number} alpha alpha
     * @param {Float32Array} A matrix A
     * @param {Number} lda the column of A
     * @param {Float32Array} B matrix B
     * @param {Number} ldb the column of B, or rows of B^T
     * @param {Float32Array} C matrix C
     * @param {Number} ldc the column of C
     * @param {Number} aOffset the offset of A
     * @param {Number} bOffset the offset of B
     * @param {Number} cOffset the offset of C
     */
    static gemmNT(M, N, K, alpha, A, lda, B, ldb, C, ldc, aOffset = 0, bOffset = 0, cOffset = 0) {
        for (let i = 0; i < M; ++i) {
            for (let j = 0; j < N; ++j) {
                let sum = 0;
                for (let p = 0; p < K; ++p) {
                    sum += alpha * A[aOffset + i * lda + p] * B[bOffset + j * ldb + p];
                }
                C[cOffset + i * ldc + j] += sum;
            }
        }
    }

    /**
     * General Matrix Multiplication
     * C = alpha * A * B
     * A dimensions: M x K
     * B dimensions: K x N
     * C dimensions: M x N
     * 
     * @param {Number} M the row of matrix A and C
     * @param {Number} N the column of matrix B and C
     * @param {Number} K the row of matrix B, or column of matrix A.
     * @param {Number} alpha alpha
     * @param {Float32Array} A matrix A
     * @param {Number} lda the column of A
     * @param {Float32Array} B matrix B
     * @param {Number} ldb the column of B
     * @param {Float32Array} C matrix C
     * @param {Number} ldc the column of C
     * @param {Number} aOffset the offset of A
     * @param {Number} bOffset the offset of B
     * @param {Number} cOffset the offset of C
     */
    static gemmNN(M, N, K, alpha, A, lda, B, ldb, C, ldc, aOffset = 0, bOffset = 0, cOffset = 0) {
        for (let i = 0; i < M; ++i) {
            for (let p = 0; p < K; ++p) {
                const A_PART = alpha * A[aOffset + i * lda + p];
                for (let j = 0; j < N; ++j) {
                    C[cOffset + i * ldc + j] += A_PART * B[bOffset + p * ldb + j];
                }
            }
        }
    }

    /**
     * General Matrix Multiplication
     * C = alpha * A * B + beta * C, 
     * A dimensions: M x K, 
     * B dimensions: K x N, 
     * C dimensions: M x N
     * 
     * @param {boolean} TA transpose of A
     * @param {boolean} TB transpose of B
     * @param {Number} M the rows of A and C, or rows of A^T if A was transposed
     * @param {Number} N the columns of B and C, or columns of B^T if B was transposed
     * @param {Number} K the row of B, column of A.
     * @param {Number} alpha alpha
     * @param {Float32Array} A matrix A
     * @param {Number} lda the column of A, or rows of A^T if A was transposed
     * @param {Float32Array} B matrix B
     * @param {Number} ldb the column of B, of rows of B^T if B was transposed
     * @param {Float32Array} C matrix C
     * @param {Number} ldc the column of C
     * @param {Number} aOffset the offset of A
     * @param {Number} bOffset the offset of B
     * @param {Number} cOffset the offset of C
     */
    static gemm(TA, TB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, aOffset = 0, bOffset = 0, cOffset = 0) {
        for (let i = 0; i < M; ++i) {
            for (let j = 0; j < N; ++j) {
                C[i * ldc + j] *= beta;
            }
        }

        if (!TA && !TB) {
            CNN.Math.gemmNN(M, N, K, alpha, A, lda, B, ldb, C, ldc, aOffset, bOffset, cOffset);
        } else if (!TA && TB) {
            CNN.Math.gemmNT(M, N, K, alpha, A, lda, B, ldb, C, ldc, aOffset, bOffset, cOffset);
        } else {
            throw new Error('the gemm method was NOT implemented');
        }
    }

    /**
     * Accelerated General Matrix Multiplication
     * @param {WebGL2} webgl2 webgl object in WebGL2.js
     * @param {*} A the matrix A data
     * @param {*} B the matrix B data
     * @param {number} aWidth the matrix A width
     * @param {number} aHeight the matrix A height 
     * @param {number} bWidth the matrix B width
     * @param {number} bHeight the matrix B height
     * @returns the A x B result
     */
    static gemmAccelerate(webgl2, A, B, aWidth, aHeight, bWidth, bHeight){

        if(aWidth > webgl2.MAX_TEXTURE_SIZE && aHeight > webgl2.MAX_TEXTURE_SIZE){
            throw Error('Invalid gemm accelerate arguments');
        }

        if(bWidth > webgl2.MAX_TEXTURE_SIZE && bHeight > webgl2.MAX_TEXTURE_SIZE){
            throw Error('Invalid gemm accelerate arguments');
        }

        let aReLayout = false;
        if(aWidth > webgl2.MAX_TEXTURE_SIZE || aHeight > webgl2.MAX_TEXTURE_SIZE){
            aReLayout = true;
        }

        let bReLayout = false;
        if(bWidth > webgl2.MAX_TEXTURE_SIZE || bHeight > webgl2.MAX_TEXTURE_SIZE){
            bReLayout = true;
        }

        //create webgl program
        let matmulProgram = webgl2.createProgram(matMulSource);

        let inputTexture1 = null;
        let inputTexture2 = null;

        if(!aReLayout){
            inputTexture1 = webgl2.create2DTexture({ textureWidth: aWidth, textureHeight: aHeight, textureData: A });
        } else {
            const aSize = aWidth * aHeight;
            const aLayoutWidth = webgl2.MAX_TEXTURE_SIZE;
            const aLayoutHeight = Math.ceil(aSize / aLayoutWidth);

            let newA = new Float32Array(aLayoutWidth * aLayoutHeight);
            for(let i = 0; i < A.length; ++i){
                newA[i] = A[i];
            }

            inputTexture1 = webgl2.create2DTexture({ textureWidth: aLayoutWidth, textureHeight: aLayoutHeight, textureData: newA });
        }

        if(!bReLayout){
            inputTexture2 = webgl2.create2DTexture({ textureWidth: bWidth, textureHeight: bHeight, textureData: B });
        } else {
            const bSize = bWidth * bHeight;
            const bLayoutWidth = webgl2.MAX_TEXTURE_SIZE;
            const bLayoutHeight = Math.ceil(bSize / bLayoutWidth);

            let newB = new Float32Array(bLayoutWidth * bLayoutHeight);
            for(let i = 0; i < B.length; ++i){
                newB[i] = B[i];
            }

            inputTexture2 = webgl2.create2DTexture({ textureWidth: bLayoutWidth, textureHeight: bLayoutHeight, textureData: newB });
        }

        let outputTexWidth = bWidth;
        let outputTexHeight = aHeight;

        if(bWidth > webgl2.MAX_TEXTURE_SIZE || aHeight > webgl2.MAX_TEXTURE_SIZE){
            const outSize = bWidth * aHeight;
            outputTexWidth = webgl2.MAX_TEXTURE_SIZE;
            outputTexHeight = Math.ceil(outSize / outputTexWidth);
        }

        let outputData = webgl2.run({
            program: matmulProgram,
            inputs: [{texture: inputTexture1, name: 'A'}, {texture: inputTexture2, name: 'B'}],
            uniforms: [{ name: "aWidth", type: "int", data: aWidth},      { name: "aHeight", type: "int", data: aHeight},
                       { name: "bWidth", type: "int", data: bWidth},      { name: "bHeight", type: "int", data: bHeight},
                       { name: "outTexWidth", type: "int", data: outputTexWidth}, { name: "outTexHeight", type: "int", data: outputTexHeight},
                       { name: "outWidth", type: "int", data: bWidth},    { name: "outHeight", type: "int", data: aHeight}],
            output: { width: outputTexWidth, height: outputTexHeight }
        });

        let result = new Float32Array(bWidth * aHeight);
        for(let i = 0; i < result.length; ++i){
            result[i] = outputData[i];
        }

        return result;
    }

    /**
     * compute the IOU(intersection over union)
     * @param {number} x1center center x 
     * @param {number} y1center center y
     * @param {number} w1 width
     * @param {number} h1 height
     * @param {number} x2center center x
     * @param {number} y2center center y
     * @param {number} w2 width
     * @param {number} h2 height
     * @returns the IOU value
     */
    static iou(x1center, y1center, w1, h1, x2center, y2center, w2, h2) {
        let x1l = x1center - w1 / 2;
        let x1r = x1center + w1 / 2;
        let y1t = y1center - h1 / 2;
        let y1b = y1center + h1 / 2;

        let x2l = x2center - w2 / 2;
        let x2r = x2center + w2 / 2;
        let y2t = y2center - h2 / 2;
        let y2b = y2center + h2 / 2;


        let interW = Math.max(0, Math.min(x1r, x2r)) - Math.max(x1l, x2l);
        let interH = Math.max(0, Math.min(y1b, y2b)) - Math.max(y1t, y2t);
        let inter = interW * interH;
        let area1 = w1 * h1;
        let area2 = w2 * h2;
        let union = area1 + area2 - inter;
        return inter / union;
    }

    /**
     * mish activation
     * @param {number} x 
     * @returns 
     */
    static mish(x) {
        const MISH_THRESHOLD = 20;
        let v = 0;
        if (x > MISH_THRESHOLD) {
            v = x;
        } else if (x < -MISH_THRESHOLD) {
            v = Math.exp(x);
        } else {
            v = Math.log(Math.exp(x) + 1);
        }

        return x * (2 / (1 + Math.exp(-2 * v)) - 1);
    }

    /**
     * Calculate the logistic value( 1 / 1 + exp(-x))
     * @param {number} x  
     * @returns 
     */
    static logistic(x) {
        return 1 / (1 + Math.exp(-x));
    }

    /**
     * Calculate each item's logistic in the Array
     * @param {Float32Array} array the float array 
     * @param {number} startIndex the start index in the array
     * @param {number} length the length
     */
    static logisticRange(array, startIndex, length) {
        for (let i = 0; i < length; ++i) {
            array[startIndex + i] = CNN.Math.logistic(array[startIndex + i]);
        }
    }

    /**
     * Calculate the softmax.
     * Get `length` items from `inArray`, these items start at `startIndex` and stride is `stride`.
     * the `outputArray` holds the softmax results
     * @param {Float32Array} inArray the float array
     * @param {number} startIndex the first item index
     * @param {number} length the items count
     * @param {number} stride the stride
     * @param {number} outputArray output parameter. holds the softmax results
     */
    static softmaxRange(inArray, startIndex, length, stride, outputArray) {
        let sum = 0;
        let largest = -Number.MAX_VALUE;

        for (let i = 0; i < length; ++i) {
            if (inArray[startIndex + i * stride] > largest) {
                largest = inArray[startIndex + i * stride];
            }
        }

        for (let i = 0; i < length; ++i) {
            let e = Math.exp(inArray[startIndex + i * stride] - largest);
            sum += e;
            outputArray[startIndex + i * stride] = e;
        }

        for (let i = 0; i < length; ++i) {
            outputArray[startIndex + i * stride] /= sum;
        }
    }
}

/**
 * Layer definition in Convolutional Neural Network
 */
CNN.Layer = class {
    /**
     * constructor
     * @param {String} type the type of layer, like 'conv', 'maxpool' and so on
     */
    constructor(type) {
        this._type = type;
        //the workspace size for intermediate data
        this._workspaceSize = 0;
        this._webgl = null;
    }

    get type() {
        return this._type;
    }
    set type(newType) {
        this._type = newType;
    }

    get workspaceSize() {
        return this._workspaceSize;
    }
    set workspaceSize(newValue) {
        this._workspaceSize = newValue;
    }

    get webgl(){
        return this._webgl;
    }
    set webgl(newValue) {
        this._webgl = newValue;
    }

    get info() {
        return "CNN.Layer";
    }

    /**
     * forward inference
     * @param {*} input the input of the layer
     * @param {*} net the network holding this layer
     */
    forward(input, net) {
        throw new Error('layer ' + this.type + ' forward was not implemented.');
    }
};

/**
 * Convolutional layer
 */
CNN.Convolution = class extends CNN.Layer {
    /**
     * constructor
     * @param {Number} inputH input height
     * @param {Number} inputW input width
     * @param {Number} inputChannels input channels
     * @param {Number} filterNum filters number
     * @param {Number} filterSize filter size
     * @param {Number} stride stride
     * @param {Boolean} isPadding padding enabled
     * @param {String} activation activation type
     * @param {Boolean} batchNormalize batch normalized
     * @param {FunctionStringCallback} infoCallback The info callback function
     */
    constructor(inputH, inputW, inputChannels, filterNum, filterSize, stride, isPadding, activation, batchNormalize, infoCallback) {
        super('conv');

        this._infoCallback = infoCallback || console.info;

        //input info
        this._inputChannels = inputChannels;
        this._inputHeight = inputH;
        this._inputWidth = inputW;
        this._inputSize = inputChannels * inputH * inputW;

        //filter info
        this._stride = stride;
        this._filterSize = filterSize;
        this._padding = isPadding ? Math.floor(filterSize / 2) : 0;
        this._filterNum = filterNum;

        //output info
        this._outputChannels = this._filterNum;
        this._outputHeight = Math.floor((this._inputHeight + 2 * this._padding - this._filterSize) / this._stride + 1);
        this._outputWidth = Math.floor((this._inputWidth + 2 * this._padding - this._filterSize) / this._stride + 1);
        this._outputSize = this._outputChannels * this._outputWidth * this._outputHeight;
        //the output buffer
        this._output = new Float32Array(this._outputSize);

        //activation info
        this._activation = activation;
        this._batchNormalize = batchNormalize;

        //the weights size, NCHW
        this._weightsSize = this._filterNum * this._inputChannels * this._filterSize * this._filterSize;
        //the bias length
        this._biasLength = this._filterNum;

        //the weights data
        this._weights = null;
        //the bias data
        this._bias = null;

        //the workspaceSize is for img2col intermediate buffer
        super.workspaceSize = this._outputHeight * this._outputWidth * this._filterSize * this._filterSize * this._inputChannels;

        if (this._batchNormalize) {
            //each output channel has it's own scale
            this._scales = new Float32Array(this._filterNum);
            //scale initialized to 1
            this._scales.forEach((element, index, array) => { array[index] = 1 });
            this._rollingMean = new Float32Array(this._filterNum);
            this._rollingVariance = new Float32Array(this._filterNum);
        }
    }

    /**
     * load weights, float data type
     * @param {Float32Array} data 
     */
    set weights(data) {
        this._weights = data;
    }
    get weightsLength() {
        return this._weightsSize;
    }

    /**
     * load bias
     * @param {Float32Array} data 
     */
    set bias(data) {
        this._bias = data;
    }
    get biasLength() {
        return this._biasLength;
    }

    /**
     * load scales
     * @param {Float32Array} data 
     */
    set scales(data) {
        this._scales = data;
    }
    get scalesLength() {
        return this._filterNum;
    }

    /**
     * load rollingMean
     * @param {Float32Array} data 
     */
    set rollingMean(data) {
        this._rollingMean = data;
    }
    get rollingMeanLength() {
        return this._filterNum;
    }

    /**
     * load rollingVariance
     * @param {Float32Array} data 
     */
    set rollingVariance(data) {
        this._rollingVariance = data;
    }
    get rollingVarianceLength() {
        return this._filterNum;
    }

    get inputChannels() {
        return this._inputChannels;
    }
    get inputHeight() {
        return this._inputHeight;
    }
    get inputWidth() {
        return this._inputWidth;
    }
    get stride() {
        return this._stride;
    }
    get filterSize() {
        return this._filterSize;
    }
    get padding() {
        return this._padding;
    }
    get filterNum() {
        return this._filterNum;
    }
    get activation() {
        return this._activation;
    }
    get outputChannels() {
        return this._outputChannels;
    }
    get outputWidth() {
        return this._outputWidth;
    }
    get outputHeight() {
        return this._outputHeight;
    }
    get inputSize() {
        return this._inputSize;
    }
    get outputSize() {
        return this._outputSize;
    }
    get output() {
        return this._output;
    }
    get batchNormalize() {
        return this._batchNormalize;
    }
    get info() {
        return `${this.type}   ${this.filterNum}      ${this.filterSize} x ${this.filterSize} / ${this.stride}   ${this.inputWidth} x ${this.inputHeight} x ${this.inputChannels} -> ${this.outputWidth} x ${this.outputHeight} x ${this.outputChannels}`;
    }

    _scaleBias() {
        const channelSize = this._outputWidth * this._outputHeight;
        for (let i = 0; i < this._filterNum; ++i) {
            for (let j = 0; j < channelSize; ++j) {
                this._output[i * channelSize + j] *= this._scales[i];
            }
        }
    }

    _addBias() {
        const channelSize = this._outputWidth * this._outputHeight;
        for (let i = 0; i < this._filterNum; ++i) {
            for (let j = 0; j < channelSize; ++j) {
                this._output[i * channelSize + j] += this._bias[i];
            }
        }
    }

    forward(input, network) {
        let A = this._weights;
        let B = network.workspace;
        let C = this._output;
        //compute C = A * B
        if (this._filterSize === 1) {
            B = input;
        } else {
            const tsStart = performance.now();
            CNN.Math.im2col(input, this._inputChannels, this._inputHeight, this._inputWidth, this._filterSize, this._stride, this._padding, B);
            const tsEnd = performance.now();

            this._infoCallback(`im2col consumes: ${tsEnd - tsStart} milliseconds`);
        }

        const tsStart = performance.now();
        if (super.webgl 
            && this._filterSize * this._filterSize * this._inputChannels * this._filterNum < super.webgl.MAX_TEXTURE_SIZE * super.webgl.MAX_TEXTURE_SIZE 
            && this._filterSize * this._filterSize * this._inputChannels * this._outputWidth * this._outputHeight < super.webgl.MAX_TEXTURE_SIZE * super.webgl.MAX_TEXTURE_SIZE
            && this._outputWidth * this._outputHeight * this._filterNum < super.webgl.MAX_TEXTURE_SIZE * super.webgl.MAX_TEXTURE_SIZE ){
                
            this._output = CNN.Math.gemmAccelerate(super.webgl, A, B, this._filterSize * this._filterSize * this._inputChannels, this._filterNum,
                this._outputWidth * this._outputHeight, this._filterSize * this._filterSize * this._inputChannels);
        } else {
            this._infoCallback(`use cpu to gemm`);
            CNN.Math.gemm(false, false, this._filterNum, this._outputWidth * this._outputHeight, this._inputChannels * this._filterSize * this._filterSize, 1, A, this._inputChannels * this._filterSize * this._filterSize, B, this._outputWidth * this._outputHeight, 0, C, this._outputWidth * this._outputHeight);
        }
        const tsEnd = performance.now();
        this._infoCallback(`gemm consumes: ${tsEnd - tsStart} milliseconds`);

        //batch normalization
        if (this._batchNormalize) {
            const channelSize = this._outputWidth * this._outputHeight;
            this._output.forEach((element, index, array) => {
                const rollingIndex = Math.floor(index / channelSize);
                array[index] = (element - this._rollingMean[rollingIndex]) / (Math.sqrt(this._rollingVariance[rollingIndex]) + 0.000001);
            });

            this._scaleBias();
        }

        this._addBias();

        //activation
        if (this._activation === 'leaky') {
            this._output.forEach((element, index, array) => { array[index] = element > 0 ? element : (element * 0.1) });
        } else if (this._activation === 'linear') {
            //do nothing
        } else if (this._activation === 'mish') {
            this._output.forEach((element, index, array) => { array[index] = CNN.Math.mish(element) });
        } else {
            throw new Error("unsupported activation");
        }
    }
};

/**
 * Max pooling layer
 */
CNN.MaxPool = class extends CNN.Layer {
    /**
     * constructor
     * @param {Number} inputH the input height
     * @param {Number} inputW the input width
     * @param {Number} inputC the input channels
     * @param {Number} stride the stride
     * @param {Number} size the size
     * @param {Number} padding the padding size
     */
    constructor(inputH, inputW, inputC, stride, size, padding) {
        super('maxpool');

        this._channels = inputC;
        this._height = inputH;
        this._width = inputW;

        this._stride = stride || 1;
        this._size = size || this._stride;
        this._padding = padding;

        this._outputHeight = Math.floor((this._height + this._padding - this._size) / this._stride + 1);
        this._outputWidth = Math.floor((this._width + this._padding - this._size) / this._stride + 1);
        this._outputChannels = this._channels;

        this._inputSize = inputC * inputH * inputW;
        this._outputSize = this._outputChannels * this._outputWidth * this._outputHeight;

        this._output = new Float32Array(this._outputSize);
    }

    get channels() {
        return this._channels;
    }
    get height() {
        return this._height;
    }
    get width() {
        return this._width;
    }
    get outputChannels() {
        return this._outputChannels;
    }
    get outputWidth() {
        return this._outputWidth;
    }
    get outputHeight() {
        return this._outputHeight;
    }
    get inputSize() {
        return this._inputSize;
    }
    get outputSize() {
        return this._outputSize;
    }
    get output() {
        return this._output;
    }
    get size() {
        return this._size;
    }
    get stride() {
        return this._stride;
    }

    get info() {
        return `${this.type}   ${this.size} x ${this.size} / ${this.stride}   ${this.width} x ${this.height} x ${this.channels} -> ${this.outputWidth} x ${this.outputHeight} x ${this.outputChannels}`;
    }

    forward(input, network) {
        const wOffset = -Math.floor(this._padding / 2);
        const hOffset = -Math.floor(this._padding / 2);

        //channels
        for (let k = 0; k < this._channels; ++k) {
            //height
            for (let i = 0; i < this._outputHeight; ++i) {
                //width
                for (let j = 0; j < this._outputWidth; ++j) {
                    //the output destination index
                    let outIndex = j + this._outputWidth * (i + this._outputHeight * k);
                    //the max value
                    let max = -Number.MAX_VALUE;
                    //row stride
                    for (let n = 0; n < this._size; ++n) {
                        //column stride
                        for (let m = 0; m < this._size; ++m) {
                            //the original row position
                            let curH = hOffset + i * this._stride + n;
                            //the original column position
                            let curW = wOffset + j * this._stride + m;
                            let index = curW + this._width * (curH + this._height * k);
                            let valid = (curH >= 0 && curH < this._height && curW >= 0 && curW < this._width);
                            let val = valid ? input[index] : (-Number.MAX_VALUE);
                            max = (val > max) ? val : max;
                        }
                    }
                    this._output[outIndex] = max;
                }
            }
        }
    }
};

/**
 * Local Convolution layer
 */
CNN.LocalConv = class extends CNN.Layer {
    /**
     * constructor
     * @param {Number} inputH input height
     * @param {Number} inputW input width
     * @param {Number} inputChannels input channels
     * @param {Number} filterNum filters number
     * @param {Number} filterSize filter size
     * @param {Number} stride stride
     * @param {Number} padding padding
     * @param {String} activation activation type
     */
    constructor(inputH, inputW, inputChannels, filterNum, filterSize, stride, padding, activation) {
        super('local-conv');

        this._inputChannels = inputChannels;
        this._inputHeight = inputH;
        this._inputWidth = inputW;

        this._filterNum = filterNum || 1;
        this._stride = stride || 1;
        this._filterSize = filterSize || 1;
        this._padding = padding || 0;

        this._inputSize = inputChannels * inputH * inputW;

        this._outputHeight = this._computeOutputHeight();
        this._outputWidth = this._computeOutputWidth();
        this._outputChannels = this._filterNum;

        this._outputSize = this._outputChannels * this._outputWidth * this._outputHeight;

        this._locations = this._outputWidth * this._outputHeight;

        this._output = new Float32Array(this._outputSize);

        this._activation = activation;

        this._weights = null;
        this._bias = null;

        super.workspaceSize = this._outputHeight * this._outputWidth * this._filterSize * this._filterSize * this._inputChannels;
    }

    _computeOutputWidth() {
        let w = this._inputWidth;
        if (!this._padding) {
            w -= this._filterSize;
        } else {
            w -= 1;
        }

        return Math.floor(w / this._stride + 1);
    }

    _computeOutputHeight() {
        let h = this._inputHeight;
        if (!this._padding) {
            h -= this._filterSize;
        } else {
            h -= 1;
        }

        return Math.floor(h / this._stride + 1);
    }

    /**
     * load weights, float data type
     * @param {Float32Array} data 
     */
    set weights(data) {
        this._weights = data;
    }
    get weights() {
        return this._weights;
    }

    get weightsLength() {
        return this._filterSize * this._filterSize * this._inputChannels * this._filterNum * this._locations;
    }

    /**
     * load bias
     * @param {Float32Array} data 
     */
    set bias(data) {
        this._bias = data;
    }
    get bias() {
        return this._bias;
    }

    get biasLength() {
        return this._outputSize;
    }

    get channels() {
        return this._inputChannels;
    }

    get height() {
        return this._inputHeight;
    }

    get width() {
        return this._inputWidth;
    }

    get outputChannels() {
        return this._outputChannels;
    }

    get inputSize() {
        return this._inputSize;
    }

    get outputSize() {
        return this._outputSize;
    }

    get output() {
        return this._output;
    }

    get filterNum() {
        return this._filterNum;
    }

    get size() {
        return this._filterSize;
    }

    get stride() {
        return this._stride;
    }

    get info() {
        return `${this.type} ${this.height} x ${this.width} x ${this.channels}  images, ${this.filterNum} filters -> ${this._outputHeight} x ${this._outputWidth} x ${this._filterNum} image`;
    }

    forward(input, network) {
        for (let i = 0; i < this._outputSize; i++) {
            this._output[i] = this._bias[i];
        }

        let netInput = input;
        CNN.Math.im2col(netInput, this._inputChannels, this._inputHeight, this._inputWidth,
            this._filterSize, this._stride, this._padding, network.workspace);

        let a = this._weights;
        let b = network.workspace;
        let c = this._output;

        let m = this._filterNum;
        let n = 1;
        let k = this._filterSize * this._filterSize * this._inputChannels;
        for (let j = 0; j < this._locations; ++j) {
            CNN.Math.gemm(false, false, m, n, k, 1, a, k, b, this._locations, 1, c, this._locations, j * this._filterSize * this._filterSize * this._inputChannels * this._filterNum, j, j);
        }

        //activation
        this._output.forEach((element, index, array) => { array[index] = element > 0 ? element : (element * 0.1) });
    }
};

/**
 * Dropout layer
 */
CNN.Dropout = class extends CNN.Layer {
    /**
     * constructor
     * @param {Number} h height
     * @param {Number} w width
     * @param {Number} c channels
     * @param {Number} probability probability
     * @param {Number} inputSize inputSize
     */
    constructor(h, w, c, probability, inputSize) {
        super('dropout');

        this._probability = probability;
        this._inputSize = inputSize;
        this._outputSize = inputSize;

        this._channels = c;
        this._outputWidth = w;
        this._outputHeight = h;
        this._output = undefined;
    }

    get inputSize() {
        return this._inputSize;
    }
    get outputSize() {
        return this._outputSize;
    }
    get probability() {
        return this._probability;
    }
    get outputChannels() {
        return this._channels;
    }
    get outputWidth() {
        return this._outputWidth;
    }
    get outputHeight() {
        return this._outputHeight;
    }
    get info() {
        return `${this.type}      p = ${this.probability} ${this.inputSize} -> ${this.outputSize}`;
    }

    forward(input, network) {
        //Do nothing in forward
        this._output = input;
    }
};

/**
 * Fully Connected layer
 */
CNN.FullyConnected = class extends CNN.Layer {
    /**
     * constructor
     * @param {Number} inputSize input size
     * @param {Number} outputSize output size
     * @param {String} activation activation
     */
    constructor(inputSize, outputSize, activation) {
        super('fully-connected');

        this._inputSize = inputSize;
        this._outputSize = outputSize;
        this._activation = activation;

        this._channels = inputSize;
        this._width = 1;
        this._height = 1;

        this._outputChannels = outputSize;
        this._outputWidth = 1;
        this._outputHeight = 1;

        this._output = new Float32Array(this._outputSize);
        this._weights = new Float32Array(outputSize * inputSize);
        this._bias = new Float32Array(outputSize);
    }

    /**
     * load weights, float data type
     * @param {Float32Array} data 
     */
    set weights(data) {
        this._weights = data;
    }
    get weights() {
        return this._weights;
    }

    get weightsLength() {
        return this.inputSize * this.outputSize;
    }

    /**
     * load bias
     * @param {Float32Array} data 
     */
    set bias(data) {
        this._bias = data;
    }
    get bias() {
        return this._bias;
    }

    get biasLength() {
        return this.outputSize;
    }

    get channels() {
        return this._channels;
    }

    get height() {
        return this._height;
    }

    get width() {
        return this._width;
    }

    get inputSize() {
        return this._inputSize;
    }

    get outputSize() {
        return this._outputSize;
    }

    get activation() {
        return this._activation;
    }

    get outputChannels() {
        return this._outputChannels;
    }

    get outputWidth() {
        return this._outputWidth;
    }

    get outputHeight() {
        return this._outputHeight;
    }

    get output() {
        return this._output;
    }

    get info() {
        return `${this.type}    ${this.inputSize} -> ${this.outputSize}`;
    }

    _addBias() {
        for (let i = 0; i < this._outputSize; ++i) {
            this._output[i] += this._bias[i];
        }
    }

    forward(input, network) {
        this._output.forEach((element, index, array) => { array[index] = 0; });

        let m = 1;
        let k = this._inputSize;
        let n = this._outputSize;
        let a = input;
        let b = this._weights;
        let c = this._output;
        CNN.Math.gemm(false, true, m, n, k, 1, a, k, b, k, 1, c, n);
        this._addBias();
        //linear activation, do nothing
    }
};

/**
 * Route layer. This layer concatenates multiple layers output into a single output
 */
CNN.Route = class extends CNN.Layer {
    /**
     * constructor
     * @param {Array} inputLayerIndices the layer indices in network
     * @param {CNN.Network} network the network
     */
    constructor(inputLayerIndices, network) {
        super('route');

        //the input layers indices in the network
        this._inputLayersIndices = [];
        //the input layers output size
        this._inputLayersSize = [];

        let inputSize = 0;
        inputLayerIndices.forEach((element, index, array) => {
            inputSize += network.layers[element].outputSize;
            this._inputLayersIndices.push(element);
            this._inputLayersSize.push(network.layers[element].outputSize);
        });

        //the total input size
        this._inputSize = inputSize;
        //the total output size, it's same to input size
        this._outputSize = inputSize;

        //output buffer
        this._output = new Float32Array(this._outputSize);

        let firstConvLayer = network.layers[inputLayerIndices[0]];
        this._outputWidth = firstConvLayer.outputWidth;
        this._outputHeight = firstConvLayer.outputHeight;
        this._outputChannels = firstConvLayer.outputChannels;

        for (let i = 1; i < inputLayerIndices.length; ++i) {
            let nextLayer = network.layers[inputLayerIndices[i]];
            if (nextLayer.outputWidth === firstConvLayer.outputWidth &&
                nextLayer.outputHeight === firstConvLayer.outputHeight) {
                this._outputChannels += nextLayer.outputChannels;
            } else {
                this._outputHeight = 0;
                this._outputWidth = 0;
                this._outputChannels = 0;
            }
        }
    }

    get output() {
        return this._output;
    }

    get outputSize() {
        return this._outputSize;
    }

    get outputChannels() {
        return this._outputChannels;
    }

    get outputWidth() {
        return this._outputWidth;
    }

    get outputHeight() {
        return this._outputHeight;
    }

    get info() {
        let inputs = this._inputLayersSize.join(",");
        let inputsIndices = this._inputLayersIndices.join(",");

        return `${this.type}     input:[${inputsIndices}], size: [${inputs}] -> ${this._outputSize}`;
    }

    /**
     * forward inference
     * @param {*} input the input of the layer
     * @param {*} net the network holding this layer
     */
    forward(input, net) {
        let current = 0;
        for (let i = 0; i < this._inputLayersIndices.length; i++) {
            let prevOutput = net.layers[this._inputLayersIndices[i]].output;
            for (let j = 0; j < this._inputLayersSize[i]; j++) {
                this._output[current] = prevOutput[j];
                current++;
            }
        }
    }
};

/**
 * reorg
 */
CNN.Reorg = class extends CNN.Layer {
    /**
     * constructor
     * @param {Number} inputH input height
     * @param {Number} inputW input width
     * @param {Number} inputChannels input channels
     * @param {Number} stride stride
     */
    constructor(inputH, inputW, inputChannels, stride) {
        super('reorg');

        this._inputWidth = inputW;
        this._inputHeight = inputH;
        this._inputChannels = inputChannels;
        this._stride = stride;

        this._outputWidth = Math.floor(this._inputWidth / this._stride);
        this._outputHeight = Math.floor(this._inputHeight / this._stride);
        this._outputChannels = this._inputChannels * this._stride * this._stride;

        this._inputSize = this._inputWidth * this._inputHeight * this._inputChannels;
        this._outputSize = this._outputWidth * this._outputHeight * this._outputChannels;

        this._output = new Float32Array(this._outputSize);
    }

    get channels() {
        return this._inputChannels;
    }

    get height() {
        return this._inputHeight;
    }

    get width() {
        return this._inputWidth;
    }

    get inputSize() {
        return this._inputSize;
    }

    get outputSize() {
        return this._outputSize;
    }

    get outputChannels() {
        return this._outputChannels;
    }

    get outputWidth() {
        return this._outputWidth;
    }

    get outputHeight() {
        return this._outputHeight;
    }

    get output() {
        return this._output;
    }

    get info() {
        return `${this.type}    /${this._stride} ${this._inputWidth} x ${this._inputHeight} x ${this._inputChannels} -> ${this._outputWidth} x ${this._outputHeight} x ${this._outputChannels}`;
    }

    forward(input, net) {
        let out_c = Math.floor(this._inputChannels / (this._stride * this._stride));

        for (let c = 0; c < this._inputChannels; ++c) {
            for (let h = 0; h < this._inputHeight; ++h) {
                for (let w = 0; w < this._inputWidth; ++w) {
                    let outIndex = w + this._inputWidth * (h + this._inputHeight * c);
                    let c2 = c % out_c;
                    let offset = Math.floor(c / out_c);
                    let w2 = w * this._stride + offset % this._stride;
                    let h2 = h * this._stride + Math.floor(offset / this._stride);
                    let inIndex = w2 + this._inputWidth * this._stride * (h2 + this._inputHeight * this._stride * c2);
                    this._output[outIndex] = input[inIndex];
                }
            }
        }
    }
};

/**
 * Shortcut layer
 */
CNN.Shortcut = class extends CNN.Layer {
    /**
     * constructor
     * @param {Number} inputH input height
     * @param {Number} inputW input width
     * @param {Number} inputChannels input channels
     * @param {number} inputLayerIndex the layer index in network
     * @param {CNN.Network} network the network
     * @param {string} activation type
     */
    constructor(inputH, inputW, inputChannels, inputLayerIndex, network, activation) {
        super('shortcut');

        this._activation = activation;

        this._inputLayerIndex = inputLayerIndex;
        this._inputWidth = network.layers[inputLayerIndex].outputWidth;
        this._inputHeight = network.layers[inputLayerIndex].outputHeight;
        this._inputChannels = network.layers[inputLayerIndex].outputChannels;

        this._outputWidth = inputW;
        this._outputHeight = inputH;
        this._outputChannels = inputChannels;

        //the total output size, it's same to input size
        this._outputSize = this._outputWidth * this._outputHeight * this._outputChannels;
        this._inputSize = this._outputSize;

        //output buffer
        this._output = new Float32Array(this._outputSize);
    }

    get output() {
        return this._output;
    }

    get outputSize() {
        return this._outputSize;
    }

    get outputChannels() {
        return this._outputChannels;
    }

    get outputWidth() {
        return this._outputWidth;
    }

    get outputHeight() {
        return this._outputHeight;
    }

    get info() {
        return `${this.type}  input index:${this._inputLayerIndex}, ${this._inputWidth} x ${this._inputHeight} x ${this._inputChannels} => ${this._outputWidth} x ${this._outputHeight} x ${this._outputChannels}`;
    }

    /**
     * forward inference
     * @param {*} input the input of the layer
     * @param {*} net the network holding this layer
     */
    forward(input, net) {
        for (let i = 0; i < this._output.length; ++i) {
            this._output[i] = input[i];
        }

        let stride = Math.max(Math.floor(this._inputWidth / this._outputWidth), 1);
        let sample = Math.max(Math.floor(this._outputWidth / this._inputWidth), 1);

        let minw = Math.min(this._inputWidth, this._outputWidth);
        let minh = Math.min(this._inputHeight, this._outputHeight);
        let minc = Math.min(this._inputChannels, this._outputChannels);

        for (let k = 0; k < minc; ++k) {
            for (let j = 0; j < minh; ++j) {
                for (let i = 0; i < minw; ++i) {
                    let out_index = i * sample + this._outputWidth * (j * sample + this._outputHeight * k);
                    let add_index = i * stride + this._inputWidth * (j * stride + this._inputHeight * k);
                    this._output[out_index] += net.layers[this._inputLayerIndex].output[add_index];
                }
            }
        }

        if (this._activation === 'linear') {
            //do nothing
        } else if (this._activation === 'leaky') {
            this._output.forEach((element, index, array) => { array[index] = element > 0 ? element : (element * 0.1) });
        } else {
            throw new Error('unsupported activation');
        }
    }
};

/**
 * Up Sample
 */
CNN.UpSample = class extends CNN.Layer {
    /**
     * constructor
     * @param {Number} inputH the input height
     * @param {Number} inputW the input width
     * @param {Number} inputC the input channels
     * @param {Number} stride the stride
     */
    constructor(inputH, inputW, inputC, stride) {
        super('upsample');

        this._channels = inputC;
        this._height = inputH;
        this._width = inputW;

        this._stride = stride || 2;

        this._outputHeight = this._height * this._stride;
        this._outputWidth = this._width * this._stride;
        this._outputChannels = this._channels;

        this._outputSize = this._outputChannels * this._outputWidth * this._outputHeight;

        this._output = new Float32Array(this._outputSize);
    }

    get outputChannels() {
        return this._outputChannels;
    }
    get outputWidth() {
        return this._outputWidth;
    }
    get outputHeight() {
        return this._outputHeight;
    }
    get outputSize() {
        return this._outputSize;
    }
    get output() {
        return this._output;
    }

    get info() {
        return `${this.type}   ${this._stride}   ${this._width} x ${this._height} x ${this._channels} -> ${this._outputWidth} x ${this._outputHeight} x ${this._outputChannels}`;
    }

    forward(input, network) {
        for (let i = 0; i < this._output.length; ++i) {
            this._output[i] = 0;
        }

        for (let k = 0; k < this._channels; ++k) {
            for (let j = 0; j < this._height * this._stride; ++j) {
                for (let i = 0; i < this._width * this._stride; ++i) {

                    let in_index = k * this._width * this._height + Math.floor(j / this._stride) * this._width + Math.floor(i / this._stride);
                    let out_index = k * this._width * this._height * this._stride * this._stride + j * this._width * this._stride + i;
                    this._output[out_index] = input[in_index];
                }
            }
        }
    }
};

/**
 * Global average pooling layer
 */
CNN.GlobalAveragePool = class extends CNN.Layer {
    /**
     * constructor
     * @param {Number} inputH the input height
     * @param {Number} inputW the input width
     * @param {Number} inputC the input channels
     */
    constructor(inputH, inputW, inputC) {
        super('globalavgpool');

        this._channels = inputC;
        this._height = inputH;
        this._width = inputW;

        this._outputHeight = 1;
        this._outputWidth = 1;
        this._outputChannels = this._channels;

        this._inputSize = this._height * this._width * this._channels;
        this._outputSize = this._outputChannels * this._outputWidth * this._outputHeight;

        this._output = new Float32Array(this._outputSize);
    }

    get outputChannels() {
        return this._outputChannels;
    }
    get outputWidth() {
        return this._outputWidth;
    }
    get outputHeight() {
        return this._outputHeight;
    }
    get outputSize() {
        return this._outputSize;
    }
    get output() {
        return this._output;
    }

    get info() {
        return `${this.type}   ${this._width} x ${this._height} x ${this._channels} -> 1 x 1 x ${this._outputChannels}`;
    }

    forward(input, network) {

        for (let k = 0; k < this._channels; ++k) {
            this._output[k] = 0;
            for (let j = 0; j < this._height * this._width; ++j) {
                let inIndex = j + this._height * this._width * k;
                this._output[k] += input[inIndex];
            }
            this._output[k] = this._output[k] / (this._height * this._width);
        }
    }
};

/**
 * Region
 */
CNN.Region = class extends CNN.Layer {
    /**
     * constructor
     * @param {Number} inputH input height
     * @param {Number} inputW input width
     * @param {Number} boxes number of anchor boxes
     * @param {Number} klasses klasses
     * @param {Number} coords coords number\
     * @param {Boolean} is softmax
     */
    constructor(inputH, inputW, boxes, klasses, coords, isSoftmax) {
        super('region');

        this._klasses = klasses;
        this._coords = coords;
        this._isSoftmax = isSoftmax;

        this._inputWidth = inputW;
        this._inputHeight = inputH;
        this._boxes = boxes;

        this._inputChannels = this._boxes * (this._klasses + this._coords + 1);

        this._outputWidth = this._inputWidth;
        this._outputHeight = this._inputHeight;
        this._outputChannels = this._inputChannels;

        this._inputSize = this._inputWidth * this._inputHeight * this._inputChannels;
        this._outputSize = this._outputWidth * this._outputHeight * this._outputChannels;

        this._output = new Float32Array(this._outputSize);
        this._biases = new Float32Array(2 * this._boxes);
    }

    get biases() {
        return this._biases;
    }

    set biases(newValue) {
        this._biases = newValue;
    }

    get boxes() {
        return this._boxes;
    }

    get boxCount() {
        return this._outputWidth * this._outputHeight * this._boxes;
    }

    get klasses() {
        return this._klasses;
    }

    get classes() {
        return this._klasses;
    }

    get coords() {
        return this._coords;
    }

    get channels() {
        return this._inputChannels;
    }

    get height() {
        return this._inputHeight;
    }

    get width() {
        return this._inputWidth;
    }

    get inputSize() {
        return this._inputSize;
    }

    get outputSize() {
        return this._outputSize;
    }

    get outputChannels() {
        return this._outputChannels;
    }

    get outputWidth() {
        return this._outputWidth;
    }

    get outputHeight() {
        return this._outputHeight;
    }

    get output() {
        return this._output;
    }

    get info() {
        return `${this.type}`;
    }

    forward(input, net) {
        for (let i = 0; i < this._output.length; ++i) {
            this._output[i] = input[i];
        }

        for (let i = 0; i < this._boxes; ++i) {
            let index = i * this._inputHeight * this._inputWidth * (this._coords + this._klasses + 1);
            CNN.Math.logisticRange(this._output, index, 2 * this._inputHeight * this._inputWidth);
            index += this._coords * this._inputHeight * this._inputWidth;
            CNN.Math.logisticRange(this._output, index, this._inputHeight * this._inputWidth);
        }

        if (this._isSoftmax) {
            let index = (this._coords + 1) * this._inputHeight * this._inputWidth;

            let offset = this._inputHeight * this._inputWidth * (this._klasses + this._coords + 1);
            for (let b = 0; b < this._boxes; ++b) {
                for (let g = 0; g < this._inputHeight * this._inputWidth; ++g) {
                    CNN.Math.softmaxRange(input, index + b * offset + g, this._klasses, this._inputHeight * this._inputWidth, this._output);
                }
            }
        }
    }
};

/**
 * Detection layer
 */
CNN.Detection = class extends CNN.Layer {
    /**
     * 
     * @param {Number} inputSize input size
     * @param {Number} num the box count in a single grid
     * @param {Number} side the grid count in width and height
     * @param {Number} classes classes number
     * @param {Number} coords coordinates number
     */
    constructor(inputSize, num, side, classes, coords) {
        super('detection');

        this._inputSize = inputSize;
        this._num = num;
        this._side = side;
        this._classes = classes;
        this._coords = coords;

        this._sqrt = false;
    }

    get classes() {
        return this._classes;
    }

    get output() {
        return this._output;
    }

    get boxCount() {
        return this._side * this._side * this._num;
    }

    get side() {
        return this._side;
    }

    get num() {
        return this._num;
    }

    set sqrt(newValue) {
        this._sqrt = newValue;
    }
    get sqrt() {
        return this._sqrt;
    }

    get info() {
        return `${this.type}  Layer`;
    }

    forward(input, network) {
        //Do nothing in forward
        this._output = input;
    }
};

/**
 * Yolo
 */
CNN.Yolo = class extends CNN.Layer {
    /**
     * constructor
     * @param {Number} inputH input height
     * @param {Number} inputW input width
     * @param {Number} boxes number of anchor boxes
     * @param {Number[]} masks the masks
     * @param {Number} klasses klasses
     */
    constructor(inputH, inputW, boxes, masks, klasses) {
        super('yolo');

        this._boxes = boxes;
        this._klasses = klasses;
        this._masks = masks;

        this._inputWidth = inputW;
        this._inputHeight = inputH;
        this._inputChannels = this._boxes * (this._klasses + 4 + 1);

        this._outputWidth = this._inputWidth;
        this._outputHeight = this._inputHeight;
        this._outputChannels = this._inputChannels;

        this._outputSize = this._outputWidth * this._outputHeight * this._outputChannels;
        this._inputSize = this._outputSize;

        this._output = new Float32Array(this._outputSize);
        this._biases = new Float32Array(2 * this._boxes);
    }

    get biases() {
        return this._biases;
    }

    set biases(newValue) {
        this._biases = newValue;
    }

    get masks() {
        return this._masks;
    }

    get boxes() {
        return this._boxes;
    }

    get boxCount() {
        return this._outputWidth * this._outputHeight * this._boxes;
    }

    get klasses() {
        return this._klasses;
    }

    get classes() {
        return this._klasses;
    }

    get channels() {
        return this._inputChannels;
    }

    get height() {
        return this._inputHeight;
    }

    get width() {
        return this._inputWidth;
    }

    get inputSize() {
        return this._inputSize;
    }

    get outputSize() {
        return this._outputSize;
    }

    get outputChannels() {
        return this._outputChannels;
    }

    get outputWidth() {
        return this._outputWidth;
    }

    get outputHeight() {
        return this._outputHeight;
    }

    get output() {
        return this._output;
    }

    get info() {
        return `${this.type}`;
    }

    forward(input, net) {
        for (let i = 0; i < this._output.length; ++i) {
            this._output[i] = input[i];
        }

        for (let i = 0; i < this._boxes; ++i) {
            let index = i * this._inputHeight * this._inputWidth * (4 + this._klasses + 1);
            CNN.Math.logisticRange(this._output, index, 2 * this._inputHeight * this._inputWidth);
            index += 4 * this._inputHeight * this._inputWidth;
            CNN.Math.logisticRange(this._output, index, (this._klasses + 1) * this._inputHeight * this._inputWidth);
        }
    }
};

/**
 * Softmax
 */
CNN.Softmax = class extends CNN.Layer {
    /**
     * constructor
     * @param {Number} inputSize input size
     */
    constructor(inputSize) {
        super('softmax');

        this._outputSize = inputSize;
        this._inputSize = inputSize;

        this._output = new Float32Array(this._outputSize);
    }


    get inputSize() {
        return this._inputSize;
    }

    get outputSize() {
        return this._outputSize;
    }

    get output() {
        return this._output;
    }

    get info() {
        return `${this.type} ${this._inputSize} -> ${this._outputSize}`;
    }

    forward(input, net) {
        CNN.Math.softmaxRange(input, 0, this._inputSize, 1, this._output);
    }
};

/**
 * bounding box definition
 */
CNN.BBox = class {
    /**
     * constructor
     * @param {number} klasses klass counter
     */
    constructor(klasses) {
        this._probabilities = new Float32Array(klasses);

        this._x = undefined;
        this._y = undefined;
        this._w = undefined;
        this._h = undefined;

        this._objectness = 0;
        this._sortClass = 0;
    }

    set sortClass(newValue) {
        this._sortClass = newValue;
    }

    get sortClass() {
        return this._sortClass;
    }

    set x(newValue) {
        this._x = newValue;
    }
    get x() {
        return this._x;
    }

    set y(newValue) {
        this._y = newValue;
    }
    get y() {
        return this._y;
    }

    set w(newValue) {
        this._w = newValue;
    }
    get w() {
        return this._w;
    }

    set h(newValue) {
        this._h = newValue;
    }
    get h() {
        return this._h;
    }

    set objectness(newValue) {
        this._objectness = newValue;
    }

    get objectness() {
        return this._objectness;
    }

    get prob() {
        return this._probabilities;
    }
};

/**
 * Convolutional Neural Network definition
 */
CNN.Network = class {
    constructor() {
        this._layers = [];
        this._input = null;
        //the network input height
        this._height = 0;
        //the network input width
        this._width = 0;
        //the network input channels
        this._channels = 0;
        //the network input size
        this._inputSize = 0;

        //the workspace size for intermedia data
        this._workspaceSize = 0;
        this._workspace = undefined;

        this._output = undefined;
    }

    initWorkspace() {
        this._workspace = new Float32Array(this._workspaceSize);
    }

    set workspaceSize(newValue) {
        this._workspaceSize = newValue;
    }
    get workspaceSize() {
        return this._workspaceSize;
    }

    get workspace() {
        return this._workspace;
    }

    set height(newValue) {
        this._height = newValue;
        this._inputSize = this._height * this._width * this._channels;
    }
    get height() {
        return this._height;
    }

    set width(newValue) {
        this._width = newValue;
        this._inputSize = this._height * this._width * this._channels;
    }
    get width() {
        return this._width;
    }

    set channels(newValue) {
        this._channels = newValue;
        this._inputSize = this._height * this._width * this._channels;
    }
    get channels() {
        return this._channels;
    }

    get inputSize() {
        return this._inputSize;
    }

    get output() {
        return this._output;
    }

    /**
     * add layer to the network
     * @param {CNN.Layer} item 
     */
    pushLayer(item) {
        this._layers.push(item);
    }

    get layers() {
        return this._layers;
    }

    get tailLayer() {
        return this._layers[this._layers.length - 1];
    }

    /**
     * predict
     * @param {Float32Array} input 
     * @param {callback} callback the predict callback
     */
    predict(input, callback) {
        this._input = input;
        const tsInferStart = performance.now();
        for (let i = 0; i < this._layers.length; i++) {
            let layer = this._layers[i];
            
            const tsStart = performance.now();
            layer.forward(this._input, this);
            const tsEnd = performance.now();

            this._input = layer.output;
            if (callback) {
                let status = {};
                status['type'] = 'inferStatus';
                status['current'] = i+1;
                status['total'] = this._layers.length ;
                status['info'] = `layer [${layer.type}] consumes [${tsEnd - tsStart}] milliseconds`;

                callback(status);
            }
        }
        const tsInferEnd = performance.now();
        const duration = tsInferEnd - tsInferStart;

        console.log('%cThe inference consumes %d milliseconds!', 'color: red', duration);

        this._output = this._input;
    }
};

export default CNN