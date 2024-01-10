import { webgl2 } from './webgl/WebGL2.js'
import * as Utils from './utils.js'
import { postProcess as yolov1PostProcess } from './def_yolov1.js'
import { postProcess as yolov2PostProcess } from './def_yolov2.js'
import { postProcess as yolov3PostProcess } from './def_yolov3.js'
import { postProcess as resnet50PostProcess } from './def_resnet50.js'

/**
 * the inference runtime
 */
class Runtime {
    /**
     * constructor
     * @param {canvas} canvas the canvas element
     * @returns
     */
    constructor(canvas) {

        this._palette = [[255, 0, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0]];

        let ctx = canvas.getContext("2d", {willReadFrequently: true});
        if (!ctx) {
            throw Error('canvas 2d is not supported');
        }

        this._canvas = canvas;
        this._canvasCtx = ctx;
        this._image = null;
        this._weights = null;
        this._config = null;
    }

    /**
     * load image by url
     * @param {string} imageURL 
     */
    loadImage(imageURL) {
        let image = new Image();
        image.src = imageURL;
        image.onload = () => {
            console.log('load [%s] successfully', imageURL);

            console.log('image width: [%d]', image.width);
            console.log('image height: [%d]', image.height);

            this._canvas.style = `background-color: #fff;border: 1px solid black;width:${image.width}px; height:${image.height}px;display: block;`;
            this.resizeCanvasToDisplaySize(this._canvas, image.width, image.height);
            this._canvasCtx.drawImage(image, 0, 0);
            this._image = image;
        };
    }

    /**
     * Resize a canvas to match the size its displayed.
     * @param {HTMLCanvasElement} canvas The canvas to resize.
     * @param {number} width the width
     * @param {number} height the height
     * @return {Boolean} true if the canvas was resized.
     */
    resizeCanvasToDisplaySize(canvas, width, height) {
        let resizeWidth = width | canvas.clientWidth | 0;
        let resizeHeight = height | canvas.clientHeight | 0;
        if (canvas.width !== resizeWidth || canvas.height !== resizeHeight) {
            canvas.width = resizeWidth;
            canvas.height = resizeHeight;
            return true;
        }
        return false;
    }

    /**
     * get the bounding box color
     * @param {number} channel the RGB channel
     * @param {number} curr the current class
     * @param {number} max the max classes
     * @returns 
     */
    getColor(channel, curr, max) {
        let ratio = (curr / max) * (this._palette.length - 1);
        let i = Math.floor(ratio);
        let j = Math.ceil(ratio);
        ratio -= i;
        let r = (1 - ratio) * this._palette[i][channel] + ratio * this._palette[j][channel];
        return Math.floor(r);
    }

    /**
     * set labels for yolov1 network
     * @param {string} data
     */
    set labels(data) {
        this._labels = data.split('\n');
        this._labels.forEach((value, index, array) => {
            array[index] = value.trim();
        });
    }

    /**
     * set weights for yolov1 network
     * @param {ArrayBuffer} data 
     */
    set weights(data) {
        this._weights = data;
    }

    /**
     * set config data for yolov1 network
     * @param {string} data
     */
    set networkcfg(data) {
        this._config = data;
    }

    /**
     * start yolov1 inference
     * @param {Boolean} accelerate
     * @param {CallableFunction} callback 
     */
    startYolov1Infer(accelerate, callback) {
        let yolov1Network = Utils.parseConfig(this._config, accelerate ? webgl2 : null);
        Utils.parseWeights(yolov1Network, this._weights);

        let status = {};
        status['type'] = 'weights'
        status['info'] = 'parse weights successfully';
        callback(status);

        const frame = this._canvasCtx.getImageData(0, 0, this._image.width, this._image.height);
        this._imageWidth = frame.width;
        this._imageHeight = frame.height;

        let resized = Utils.resizeImageKeepRatio(frame.data, frame.width, frame.height, yolov1Network.width, yolov1Network.height);

        let inputData = new Float32Array(resized[0]);
        yolov1Network.predict(inputData, callback);

        let scoreThreshold = 0.2;
        let iouThreshold = 0.5;

        let bboxes = yolov1PostProcess(yolov1Network, scoreThreshold, iouThreshold, resized[1], resized[2]);

        let gridNum = yolov1Network.tailLayer.side * yolov1Network.tailLayer.side;
        let boxNum = yolov1Network.tailLayer.num;
        let klasses = yolov1Network.tailLayer.classes;
        let result = Utils.filterDetections(bboxes, gridNum * boxNum, klasses, scoreThreshold);

        this.drawDetections(result);
    }

    /**
     * start yolov2 inference
     * @param {Boolean} accelerate
     * @param {CallableFunction} callback 
     */
    startYolov2Infer(accelerate, callback) {
        let yolov2Network = Utils.parseConfig(this._config, accelerate ? webgl2 : null);
        Utils.parseWeights(yolov2Network, this._weights);

        let status = {};
        status['type'] = 'weights'
        status['info'] = 'parse weights successfully';
        callback(status);

        const frame = this._canvasCtx.getImageData(0, 0, this._image.width, this._image.height);
        this._imageWidth = frame.width;
        this._imageHeight = frame.height;

        let resized = Utils.resizeImageKeepRatio(frame.data, frame.width, frame.height, yolov2Network.width, yolov2Network.height);

        let inputData = new Float32Array(resized[0]);
        yolov2Network.predict(inputData, callback);

        let scoreThreshold = 0.5;
        let iouThreshold = 0.5;

        let bboxes = yolov2PostProcess(yolov2Network, scoreThreshold, iouThreshold, resized[1], resized[2]);

        let gridNum = yolov2Network.tailLayer.outputWidth * yolov2Network.tailLayer.outputHeight;
        let boxNum = yolov2Network.tailLayer.boxes;
        let klasses = yolov2Network.tailLayer.klasses;

        let result = Utils.filterDetections(bboxes, gridNum * boxNum, klasses, 0.4);

        this.drawDetections(result);
    }

    /**
     * start yolov3 inference
     * @param {Boolean} accelerate
     * @param {CallableFunction} callback 
     */
    startYolov3Infer(accelerate, callback) {
        let yolov3Network = Utils.parseConfig(this._config, accelerate ? webgl2 : null);
        Utils.parseWeights(yolov3Network, this._weights);

        let status = {};
        status['type'] = 'weights'
        status['info'] = 'parse weights successfully';
        callback(status);

        const frame = this._canvasCtx.getImageData(0, 0, this._image.width, this._image.height);
        this._imageWidth = frame.width;
        this._imageHeight = frame.height;

        let resized = Utils.resizeImageKeepRatio(frame.data, frame.width, frame.height, yolov3Network.width, yolov3Network.height);

        let inputData = new Float32Array(resized[0]);
        yolov3Network.predict(inputData, callback);

        let scoreThreshold = 0.5;
        let iouThreshold = 0.5;

        let bboxes = yolov3PostProcess(yolov3Network, scoreThreshold, iouThreshold, resized[1], resized[2]);

        let klasses = yolov3Network.tailLayer.klasses;
        let result = Utils.filterDetections(bboxes, bboxes.length, klasses, 0.5);

        this.drawDetections(result);
    }

    /**
     * start ResNet50 inference
     * @param {Boolean} accelerate
     * @param {CallableFunction} callback 
     */
    startResnet50Infer(accelerate, callback) {
        let resnet50Network = Utils.parseConfig(this._config, accelerate ? webgl2 : null);
        Utils.parseWeights(resnet50Network, this._weights);

        let status = {};
        status['type'] = 'weights'
        status['info'] = 'parse weights successfully';
        callback(status);

        const frame = this._canvasCtx.getImageData(0, 0, this._image.width, this._image.height);
        this._imageWidth = frame.width;
        this._imageHeight = frame.height;

        let resized = Utils.resizeImage(frame.data, frame.width, frame.height, resnet50Network.width, resnet50Network.height);

        let inputData = new Float32Array(resized);
        resnet50Network.predict(inputData, callback);

        let bboxes = resnet50PostProcess(resnet50Network, 5);
        this.drawClassifiers(bboxes);
    }

    /**
     * start inference
     * @param {string} networkType 
     * @param {Boolean} accelerate
     * @param {CallableFunction} callback 
     */
    startInfer(networkType, accelerate, callback) {
        if (networkType === 'yolov1') {
            this.startYolov1Infer(accelerate, callback);
        } else if (networkType === 'yolov2') {
            this.startYolov2Infer(accelerate, callback);
        } else if (networkType === 'yolov3' || networkType === 'yolov4') {
            this.startYolov3Infer(accelerate, callback);
        } else if (networkType === 'resnet50') {
            this.startResnet50Infer(accelerate, callback);
        } else {
            throw Error('Unsupported network');
        }
    }

    /**
     * draw detection result
     * @param {Array} dets the detection results
     */
    drawDetections(dets) {
        for (let i = 0; i < dets.length; i++) {
            let top = dets[i].top * this._imageHeight;
            let bottom = dets[i].bottom * this._imageHeight;
            let left = dets[i].left * this._imageWidth;
            let right = dets[i].right * this._imageWidth;

            left = (left < 0) ? 0 : left;
            right = (right > this._imageWidth - 1) ? this._imageWidth - 1 : right;
            top = (top < 0) ? 0 : top;
            bottom = (bottom > this._imageHeight - 1) ? this._imageHeight - 1 : bottom;

            let klasses = dets[i].classes.split(",");
            let scores = dets[i].scores.split(",");
            let labels = "";
            for (let i = 0; i < klasses.length; i++) {
                if (i === 0) {
                    labels = this._labels[parseInt(klasses[i])] + ":" + Math.round(parseFloat(scores[i]) * 100) + "%";
                } else {
                    labels += ("," + this._labels[parseInt(klasses[i])] + ":" + Math.round(parseFloat(scores[i]) * 100) + "%");
                }
            }

            let offset = parseInt(klasses[0]) * 123457 % 20;
            let red = this.getColor(2, offset, 20);
            let green = this.getColor(1, offset, 20);
            let blue = this.getColor(0, offset, 20);

            //draw bounding box
            this._canvasCtx.lineWidth = 3;
            this._canvasCtx.strokeStyle = `rgb(${red}, ${green}, ${blue})`;
            this._canvasCtx.strokeRect(left, top, right - left, bottom - top);
            //draw labels
            this._canvasCtx.font = "25px serif";
            this._canvasCtx.textAlign = "left";
            this._canvasCtx.textBaseline = "bottom";

            //labels background
            const textMetrics = this._canvasCtx.measureText(labels);
            const labelsHeight = textMetrics['actualBoundingBoxAscent'] + textMetrics['actualBoundingBoxDescent'];
            this._canvasCtx.fillStyle = `rgb(${red}, ${green}, ${blue})`;
            this._canvasCtx.fillRect(left, top - labelsHeight - 5, textMetrics.width, labelsHeight + 5);

            //labels text
            this._canvasCtx.fillStyle = "black";
            this._canvasCtx.fillText(labels, left, top);
        }
    }

    /**
     * draw classifier result
     * @param {Array} classifiers the classifier results, {index: ,value:} array
     */
    drawClassifiers(classifiers) {
        let top = 30;
        for (let i = 0; i < classifiers.length; i++) {
            let index = classifiers[i].index;
            let score = classifiers[i].value;

            let labels = this._labels[index] + " " + Math.floor(score * 100) / 100;
            let red = 255;
            let green = 0;
            let blue = 0;

            //draw labels
            this._canvasCtx.font = "25px serif";
            this._canvasCtx.textAlign = "left";
            this._canvasCtx.textBaseline = "bottom";

            //labels text
            this._canvasCtx.fillStyle = `rgb(${red}, ${green}, ${blue})`;
            this._canvasCtx.fillText(labels, 10, top);

            const textMetrics = this._canvasCtx.measureText(labels);
            const labelsHeight = textMetrics['actualBoundingBoxAscent'] + textMetrics['actualBoundingBoxDescent'];
            top += (labelsHeight + 10);
        }
    }
};

export { Runtime }