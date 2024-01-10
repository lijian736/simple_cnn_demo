import CNN from './network.js'

/**
 * Binary data reader
 */
class BinaryReader {

    /**
     * constructor
     * @param {ArrayBuffer} data 
     */
    constructor(data) {
        this._buffer = data;
        this._position = 0;
        this._length = this._buffer.byteLength;
        this._view = new DataView(this._buffer);
    }

    get length() {
        return this._length;
    }

    get position() {
        return this._position;
    }

    seek(position) {
        this._position = position >= 0 ? position : this._length + position;
        if (this._position > this._length || this._position < 0) {
            throw new Error('Expected ' + (this._position - this._length) + ' more bytes. The file might be corrupted. Unexpected end of file.');
        }
    }

    skip(offset) {
        this._position += offset;
        if (this._position > this._length) {
            throw new Error('Expected ' + (this._position - this._length) + ' more bytes. The file might be corrupted. Unexpected end of file.');
        }
    }

    align(mod) {
        if (this._position % mod != 0) {
            this.skip(mod - (this._position % mod));
        }
    }

    readAsBuffer(length) {
        if (this._position === 0 && length === undefined) {
            this._position = this._length;
            return this._buffer;
        }
        const position = this._position;
        this.skip(length !== undefined ? length : this._length - this._position);
        return this._buffer.slice(position, this._position);
    }

    /**
     * read as data view
     * @param {Number} length bytes length
     * @returns DataView
     */
    readAsDataView(length) {
        if (this._position === 0 && length === undefined) {
            this._position = this._length;
            return new DataView(this._buffer);
        }
        const position = this._position;
        const skipLength = (length !== undefined ? length : this._length - this._position);
        this.skip(skipLength);
        return new DataView(this._buffer, position, skipLength);
    }

    /**
     * read as Float32Array
     * @param {Number} length float length
     * @returns Float32Array
     */
    readAsFloat32Array(length) {
        if (this._position === 0 && length === undefined) {
            this._position = this._length;
            return new Float32Array(this._buffer);
        }
        const position = this._position;
        const skipLength = (length !== undefined ? length : Math.floor((this._length - this._position) / Float32Array.BYTES_PER_ELEMENT));
        this.skip(skipLength * Float32Array.BYTES_PER_ELEMENT);
        return new Float32Array(this._buffer, position, skipLength);
    }

    int8() {
        const position = this._position;
        this.skip(1);
        return this._view.getInt8(position);
    }

    int16() {
        const position = this._position;
        this.skip(2);
        return this._view.getInt16(position, true);
    }

    int32() {
        const position = this._position;
        this.skip(4);
        return this._view.getInt32(position, true);
    }

    int64() {
        const position = this._position;
        this.skip(8);
        return this._view.getBigInt64(position, true);
    }

    uint8() {
        const position = this._position;
        this.skip(1);
        return this._view.getUint8(position);
    }

    uint16() {
        const position = this._position;
        this.skip(2);
        return this._view.getUint16(position, true);
    }

    uint32() {
        const position = this._position;
        this.skip(4);
        return this._view.getUint32(position, true);
    }

    uint64() {
        const position = this._position;
        this.skip(8);
        return this._view.getBigUint64(position, true);
    }

    float32() {
        const position = this._position;
        this.skip(4);
        return this._view.getFloat32(position, true);
    }

    float64() {
        const position = this._position;
        this.skip(8);
        return this._view.getFloat64(position, true);
    }
};

/**
 * config parser
 */
class ConfigParser {
    /**
     * constructor
     * @param {Text} data 
     */
    constructor(data) {
        this._configs = [];

        let config = {};

        let lines = data.trim().split(/\n/).map((line) => line.split(' ').join('')).filter((line) => line.length > 0 && !line.startsWith("#"));
        for (let line of lines) {
            if (line === '[net]' || line === '[network]') {
                config = {};
                config['type'] = 'net';
            } else if (line === '[convolutional]') {
                this._addItem(config);
                config = {};
                config['type'] = 'conv';
            } else if (line === '[maxpool]') {
                this._addItem(config);
                config = {};
                config['type'] = 'maxpool';
            } else if (line === '[local]') {
                this._addItem(config);
                config = {};
                config['type'] = 'local-conv';
            } else if (line === '[dropout]') {
                this._addItem(config);
                config = {};
                config['type'] = 'dropout';
            } else if (line === '[connected]') {
                this._addItem(config);
                config = {};
                config['type'] = 'fully-connected';
            } else if (line === '[detection]') {
                this._addItem(config);
                config = {};
                config['type'] = 'detection';
            } else if (line === '[reorg]') {
                this._addItem(config);
                config = {};
                config['type'] = 'reorg';
            } else if (line === '[route]') {
                this._addItem(config);
                config = {};
                config['type'] = 'route';
            } else if (line === '[region]') {
                this._addItem(config);
                config = {};
                config['type'] = 'region';
            } else if (line === '[shortcut]') {
                this._addItem(config);
                config = {};
                config['type'] = 'shortcut';
            } else if (line === '[upsample]') {
                this._addItem(config);
                config = {};
                config['type'] = 'upsample';
            } else if (line === '[yolo]') {
                this._addItem(config);
                config = {};
                config['type'] = 'yolo';
            } else if (line === '[globalavgpool]') {
                this._addItem(config);
                config = {};
                config['type'] = 'globalavgpool';
            } else if (line === '[softmax]') {
                this._addItem(config);
                config = {};
                config['type'] = 'softmax';
            } else if (line.startsWith('[')) {
                throw new Error('unsupported layer: ' + line);
            } else if (line.trim().length > 0) {
                let keyValue = line.trim().split('=');
                if (keyValue.length != 2) {
                    throw new Error('Invalid format: ' + line);
                }
                config[keyValue[0]] = keyValue[1];
            }
        }
        this._addItem(config);
    }

    /**
     * add config item
     * @param {Object} item 
     */
    _addItem(item) {
        this._configs.push(item);
    }

    /**
     * get the config layer list, 'net' NOT included
     */
    get layers() {
        return this._configs.filter((config) => config.type !== 'net');
    }

    /**
     * get the 'net' config
     */
    get net() {
        let result = this._configs.filter((config) => config.type === 'net');
        if (result.length > 0) {
            return result[0];
        } else {
            return undefined;
        }
    }
};


/**
* resize image
* @param {Uint8ClampedArray} dataSrc the source image data, RGBA format in HWC layout range from [0, 255]
* @param {number} srcWidth the source image width
* @param {number} srcHeight the source image height
* @param {number} destWidth the destination image width
* @param {number} destHeight the destination image height
* 
* @returns the resized image, RGB format in CHW layout range from [0, 1.0]
*/
export function resizeImage(dataSrc, srcWidth, srcHeight, destWidth, destHeight) {
    if (destWidth === 1 || destHeight === 1) {
        throw new Error('resize image invalid parameters[destWidth, destHeight]');
    }

    const wScale = (srcWidth - 1) / (destWidth - 1);
    const hScale = (srcHeight - 1) / (destHeight - 1);
    const destSize = destWidth * destHeight;

    let result = new Float32Array(destSize * 3); //3 channels, CHW
    result.forEach((value, index, array) => {
        let c = Math.floor(index / destSize);
        let h = Math.floor((index - c * destSize) / destWidth);
        let w = index - c * destSize - h * destWidth;

        if (w !== (destWidth - 1) && srcWidth !== 1 && h !== (destHeight - 1) && srcHeight !== 1) {
            let srcH = h * hScale;
            let srcW = w * wScale;
            let srcHi = Math.floor(srcH);
            let srcWi = Math.floor(srcW);

            let dy = srcH - srcHi;
            let dx = srcW - srcWi;

            let v1 = dataSrc[4 * (srcHi * srcWidth + srcWi) + c];
            let v2 = dataSrc[4 * (srcHi * srcWidth + srcWi + 1) + c];
            let v3 = dataSrc[4 * ((srcHi + 1) * srcWidth + srcWi) + c];
            let v4 = dataSrc[4 * ((srcHi + 1) * srcWidth + srcWi + 1) + c];

            let a1 = (1 - dx) * v1 + dx * v2;
            let b1 = (1 - dx) * v3 + dx * v4;
            array[index] = ((1 - dy) * a1 + dy * b1) / 255;
        }
    });

    return result;
}

/**
* resize image keeping the ratio
* @param {Uint8ClampedArray} dataSrc the source image data, RGBA format in HWC layout range from [0, 255]
* @param {number} srcWidth the source image width
* @param {number} srcHeight the source image height
* @param {number} destWidth the destination image width
* @param {number} destHeight the destination image height
* 
* @returns the resized image, RGB format in CHW layout range from [0, 1.0]
*/
export function resizeImageKeepRatio(dataSrc, srcWidth, srcHeight, destWidth, destHeight) {
    if (destWidth === 1 || destHeight === 1) {
        throw new Error('resize image invalid parameters[destWidth, destHeight]');
    }


    let scale = 1;
    if ((destHeight / srcHeight) < (destWidth / srcWidth)) {
        scale = (destHeight / srcHeight);
    } else {
        scale = (destWidth / srcWidth);
    }

    const width = Math.floor(srcWidth * scale);
    const height = Math.floor(srcHeight * scale);

    let widthScale = width / destWidth;
    let heightScale = height / destHeight;

    const destSize = destWidth * destHeight;
    let result = new Float32Array(destSize * 3); //3 channels, CHW

    const wScale = (srcWidth - 1) / (width - 1);
    const hScale = (srcHeight - 1) / (height - 1);

    for (let c = 0; c < 4; ++c) {
        for (let h = 0; h < height; ++h) {
            for (let w = 0; w < width; ++w) {
                if (w !== (width - 1) && srcWidth !== 1 && h !== (height - 1) && srcHeight !== 1) {
                    let srcH = h * hScale;
                    let srcW = w * wScale;
                    let srcHi = Math.floor(srcH);
                    let srcWi = Math.floor(srcW);

                    let dy = srcH - srcHi;
                    let dx = srcW - srcWi;

                    let v1 = dataSrc[4 * (srcHi * srcWidth + srcWi) + c];
                    let v2 = dataSrc[4 * (srcHi * srcWidth + srcWi + 1) + c];
                    let v3 = dataSrc[4 * ((srcHi + 1) * srcWidth + srcWi) + c];
                    let v4 = dataSrc[4 * ((srcHi + 1) * srcWidth + srcWi + 1) + c];

                    let a1 = (1 - dx) * v1 + dx * v2;
                    let b1 = (1 - dx) * v3 + dx * v4;
                    result[c * destSize + h * destWidth + w] = ((1 - dy) * a1 + dy * b1) / 255;
                }
            }
        }
    }

    return [result, widthScale, heightScale];
}

/**
* crop image
* @param {Uint8ClampedArray} dataSrc the source image data, RGB format in CHW layout range from [0, 1.0]
* @param {number} srcWidth the source image width
* @param {number} srcHeight the source image height
* @param {number} cropStartX the crop x position
* @param {number} cropStartY the crop y position
* @param {number} cropWidth the crop width
* @param {number} cropHeight the crop height
* 
* @returns the croped image, RGB format in CHW layout range from [0, 1.0]
*/
export function cropImage(dataSrc, srcWidth, srcHeight, cropStartX, cropStartY, cropWidth, cropHeight) {
    if (cropStartX < 0 || cropWidth < 0 || cropStartX + cropWidth > srcWidth
        || cropStartY < 0 || cropHeight < 0 || cropStartY + cropHeight > srcHeight) {
        throw new Error('crop image invalid parameters');
    }

    const srcSize = srcWidth * srcHeight;
    const destSize = cropWidth * cropHeight;

    let result = new Float32Array(destSize * 3); //3 channels, CHW
    result.forEach((value, index, array) => {
        let c = Math.floor(index / destSize);
        let h = Math.floor((index - c * destSize) / destWidth);
        let w = index - c * destSize - h * destWidth;

        h += cropStartY;
        w += cropStartX;

        let pos = c * srcSize + h * srcWidth + w;
        array[index] = dataSrc[pos];
    });

    return result;
}

/**
 * filter detection result by the object scroe
 * @param {Array} dets the detection results, BBox array
 * @param {number} num the bbox number
 * @param {number} classes the detection classes
 * @param {number} thresh threshold of confidence
 * @returns the bounding box array
 */
export function filterDetections(dets, num, classes, thresh) {
    let result = [];

    for (let i = 0; i < num; i++) {
        let klass = -1;

        let bbox = {};
        bbox.classes = "";
        bbox.scores = "";
        for (let j = 0; j < classes; j++) {
            if (dets[i].prob[j] > thresh) {
                if (klass < 0) {
                    bbox.classes += `${j}`;
                    bbox.scores += `${dets[i].prob[j]}`;
                    klass = j;
                } else {
                    bbox.classes += `,${j}`;
                    bbox.scores += `,${dets[i].prob[j]}`;
                }
            }
        }

        if (klass >= 0) {
            let left = dets[i].x - dets[i].w / 2;
            let right = dets[i].x + dets[i].w / 2;
            let top = dets[i].y - dets[i].h / 2;
            let bottom = dets[i].y + dets[i].h / 2;

            bbox.left = left;
            bbox.right = right;
            bbox.top = top;
            bbox.bottom = bottom;

            result.push(bbox);
        }
    }

    return result;
}

/**
 * do NMS by the class. Each class will do NMS
 * 
 * @param {Array} dets CNN.BBox Array 
 * @param {number} boxes bbox number
 * @param {number} classes class number
 * @param {number} iouThreshold the IOU threshold
 */
export function doNMS(dets, boxes, classes, iouThreshold) {
    let k = boxes - 1;
    //move the empty bounding box to the end of the array
    for (let i = 0; i <= k; ++i) {
        if (Math.abs(dets[i].objectness) < 0.0001) {
            let swap = dets[i];
            dets[i] = dets[k];
            dets[k] = swap;
            --k;
            --i;
        }
    }
    boxes = k + 1;

    for (let k = 0; k < classes; ++k) {
        for (let i = 0; i < boxes; ++i) {
            dets[i].sortClass = k;
        }
        dets.sort((ele1, ele2) => {
            let diff = 0;
            if (ele2.sortClass >= 0) {
                diff = ele1.prob[ele2.sortClass] - ele2.prob[ele2.sortClass];
            } else {
                diff = ele1.objectness - ele2.objectness;
            }
            if (diff < 0) {
                return 1;
            } else if (diff > 0) {
                return -1;
            } else {
                return 0;
            }
        });

        for (let i = 0; i < boxes; ++i) {
            if (Math.abs(dets[i].prob[k]) < 0.0001) {
                continue;
            }
            for (let j = i + 1; j < boxes; ++j) {
                if (CNN.Math.iou(dets[i].x, dets[i].y, dets[i].w, dets[i].h, dets[j].x, dets[j].y, dets[j].w, dets[j].h) > iouThreshold) {
                    dets[j].prob[k] = 0;
                }
            }
        }
    }
}


/**
 * parse the network config
 * @param {string} config the configure string
 * @param {WebGL} webgl the webgl object
 * @returns CNN.Network 
 */
export function parseConfig(config, webgl) {
    let parser = new ConfigParser(config);

    let network = new CNN.Network();
    network.height = Number.parseInt(parser.net.height);
    network.width = Number.parseInt(parser.net.width);
    network.channels = Number.parseInt(parser.net.channels);

    let params = {};
    params.h = network.height;
    params.w = network.width;
    params.c = network.channels;
    params.inputSize = network.inputSize;

    let workspaceSize = 0;
    //iterate the layers config information
    let layers = parser.layers;
    for (let item of layers) {
        let layer = null;
        if (item.type === 'conv') {
            layer = new CNN.Convolution(params.h, params.w, params.c, Number.parseInt(item.filters), Number.parseInt(item.size), Number.parseInt(item.stride), !!Number.parseInt(item.pad), item.activation, !!Number.parseInt(item.batch_normalize));
            network.pushLayer(layer);
        } else if (item.type === 'maxpool') {
            layer = new CNN.MaxPool(params.h, params.w, params.c, Number.parseInt(item.stride), Number.parseInt(item.size), Number.parseInt(item.size) - 1);
            network.pushLayer(layer);
        } else if (item.type === 'local-conv') {
            layer = new CNN.LocalConv(params.h, params.w, params.c, Number.parseInt(item.filters), Number.parseInt(item.size), Number.parseInt(item.stride), Number.parseInt(item.pad), item.activation);
            network.pushLayer(layer);
        } else if (item.type === 'dropout') {
            layer = new CNN.Dropout(params.h, params.w, params.c, Number.parseFloat(item.probability), params.inputSize);
            layer.output = network.tailLayer.output;
            network.pushLayer(layer);
        } else if (item.type === 'fully-connected') {
            layer = new CNN.FullyConnected(params.inputSize, Number.parseInt(item.output), item.activation);
            network.pushLayer(layer);
        } else if (item.type === 'detection') {
            layer = new CNN.Detection(params.inputSize, Number.parseInt(item.num), Number.parseInt(item.side), Number.parseInt(item.classes), Number.parseInt(item.coords));
            layer.sqrt = !!item.sqrt;
            network.pushLayer(layer);
        } else if (item.type === 'reorg') {
            layer = new CNN.Reorg(params.h, params.w, params.c, Number.parseInt(item.stride));
            network.pushLayer(layer);
        } else if (item.type === 'route') {
            let layersIndices = item.layers.split(",");
            layersIndices.forEach((value, index, arr) => {
                let layerIndex = Number.parseInt(value);
                if (layerIndex < 0) {
                    layerIndex += network.layers.length;
                }
                arr[index] = layerIndex;
            });
            layer = new CNN.Route(layersIndices, network);
            network.pushLayer(layer);
        } else if (item.type === 'region') {
            layer = new CNN.Region(params.h, params.w, Number.parseInt(item.num), Number.parseInt(item.classes), Number.parseInt(item.coords), !!Number.parseInt(item.softmax));

            let biases = item.anchors.split(",").map((anchor) => anchor.split(' ').join('')).map((anchor) => parseFloat(anchor));
            layer.biases = biases;

            network.pushLayer(layer);
        } else if (item.type === 'shortcut') {
            let fromLayerIndex = Number.parseInt(item.from) + network.layers.length;
            layer = new CNN.Shortcut(params.h, params.w, params.c, fromLayerIndex, network, item.activation);
            network.pushLayer(layer);
        } else if (item.type === 'upsample') {
            layer = new CNN.UpSample(params.h, params.w, params.c, Number.parseInt(item.stride));
            network.pushLayer(layer);
        } else if (item.type === 'yolo') {
            let num = Number.parseInt(item.num);
            let masks = item.mask.split(",").map((mask) => mask.split(' ').join('')).map((mask) => parseFloat(mask));
            if (masks.length > 0) {
                num = masks.length;
            }
            layer = new CNN.Yolo(params.h, params.w, num, masks, Number.parseInt(item.classes));

            let biases = item.anchors.split(",").map((anchor) => anchor.split(' ').join('')).map((anchor) => parseFloat(anchor));
            layer.biases = biases;

            network.pushLayer(layer);
        } else if (item.type === 'globalavgpool') {
            layer = new CNN.GlobalAveragePool(params.h, params.w, params.c);
            network.pushLayer(layer);
        } else if (item.type === 'softmax') {
            layer = new CNN.Softmax(params.inputSize);
            network.pushLayer(layer);
        } else {
            throw new Error("Layer '" + item.type + "' not support");
        }

        if (webgl) {
            layer.webgl = webgl;
        }

        if (layer.workspaceSize > workspaceSize) {
            workspaceSize = layer.workspaceSize;
        }

        params.h = layer.outputHeight;
        params.w = layer.outputWidth;
        params.c = layer.outputChannels;
        params.inputSize = layer.outputSize;
    }

    network.workspaceSize = workspaceSize;
    network.initWorkspace();

    console.log("layer     filters    size              input                output");
    network.layers.forEach((item, index, arr) => {
        console.log("%d  %s", index, item.info);
    });


    return network;
}

/**
 * parse weights
 * @param {CNN.Network} network
 * @param {ArrayBuffer} data 
 */
export function parseWeights(network, data) {
    let reader = new BinaryReader(data);
    let length = reader.length;

    let major = reader.uint32();
    let minor = reader.uint32();
    let revision = reader.uint32();

    console.log('major: %d, minor: %d, revison: %d', major, minor, revision);

    if ((major * 10 + minor) >= 2 && major < 1000 && minor < 1000) {
        reader.skip(8);
    } else {
        reader.skip(4);
    }

    network.layers.forEach((item, index, arr) => {
        if (item.type === 'conv') {
            item.bias = reader.readAsFloat32Array(item.biasLength);
            if (item.batchNormalize) {
                item.scales = reader.readAsFloat32Array(item.scalesLength);
                item.rollingMean = reader.readAsFloat32Array(item.rollingMeanLength);
                item.rollingVariance = reader.readAsFloat32Array(item.rollingVarianceLength);
            }
            item.weights = reader.readAsFloat32Array(item.weightsLength);
        } else if (item.type === 'fully-connected') {
            item.bias = reader.readAsFloat32Array(item.biasLength);
            item.weights = reader.readAsFloat32Array(item.weightsLength);
        } else if (item.type === 'local-conv') {
            item.bias = reader.readAsFloat32Array(item.biasLength);
            item.weights = reader.readAsFloat32Array(item.weightsLength);
        }
    });
}