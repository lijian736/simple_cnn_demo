import * as Utils from './utils.js'
import CNN from './network.js'


/**
 * post process
 *
 * @param {CNN.Network} yolov3Network the yolov3 network
 * @param {number} detectThreshold the threshold of object detection
 * @param {number} iouThreshold the threshold of IOU
 * @param {number} widthScale the input width scale
 * @param {number} heightScale the input height scale
 * 
 * @returns the bounding box array
 */
export function postProcess(yolov3Network, detectThreshold, iouThreshold, widthScale, heightScale) {
    let entryIndex = (layer, location, entry) => {
        let n = Math.floor(location / (layer.outputWidth * layer.outputHeight));
        let loc = location % (layer.outputWidth * layer.outputHeight);
        return n * layer.outputWidth * layer.outputHeight * (4 + layer.klasses + 1) + entry * layer.outputWidth * layer.outputHeight + loc;
    }

    //step 1. get the boxes number
    let totalBoxes = 0;
    for (let layer of yolov3Network.layers) {
        let boxes = 0;
        if (layer.type === 'yolo') {
            for (let i = 0; i < layer.outputWidth * layer.outputHeight; ++i) {
                for (let n = 0; n < layer.boxes; ++n) {
                    let confidenceIndex = entryIndex(layer, n * layer.outputWidth * layer.outputHeight + i, 4);
                    if (layer.output[confidenceIndex] > detectThreshold) {
                        boxes++;
                    }
                }
            }
        }

        totalBoxes += boxes;
    }

    let tailLayer = yolov3Network.tailLayer;
    let dectBoxes = [];
    for (let i = 0; i < totalBoxes; i++) {
        dectBoxes.push(new CNN.BBox(tailLayer.classes));
    }

    let getYoloDetections = (layer, boxes, boxStartIndex) => {
        let predictions = layer.output;
        let count = 0;

        //the grid count
        const gridCount = layer.outputWidth * layer.outputHeight;

        for (let i = 0; i < gridCount; ++i) {
            //the grid row
            let row = Math.floor(i / layer.outputWidth);
            //the grid column
            let col = i % layer.outputWidth;

            for (let n = 0; n < layer.boxes; ++n) {
                let confidenceIndex = entryIndex(layer, n * gridCount + i, 4);
                let confidence = predictions[confidenceIndex];
                if(confidence <= detectThreshold){
                    continue;
                }

                let boxIndex = entryIndex(layer, n * gridCount + i, 0);
                //x range is [0, 1.0]
                boxes[boxStartIndex + count].x = (col + predictions[boxIndex]) / layer.outputWidth;
                //y range is [0, 1.0]
                boxes[boxStartIndex + count].y = (row + predictions[boxIndex + gridCount]) / layer.outputHeight;
                //w range is [0, 1.0]
                boxes[boxStartIndex + count].w = Math.exp(predictions[boxIndex + 2 * gridCount]) * layer.biases[2 * layer.masks[n]] / yolov3Network.width;
                //h range is [0, 1.0]
                boxes[boxStartIndex + count].h = Math.exp(predictions[boxIndex + 3 * gridCount]) * layer.biases[2 * layer.masks[n] + 1] / yolov3Network.height;
                boxes[boxStartIndex + count].objectness = confidence;

                for (let j = 0; j < layer.classes; ++j) {
                    let classIndex = entryIndex(layer, n * gridCount + i, 4 + 1 + j);
                    let prob = confidence * predictions[classIndex];
                    boxes[boxStartIndex + count].prob[j] = (prob > detectThreshold) ? prob : 0;
                }

                count++;
            }
        }

        return count;
    };

    let boxIndex = 0;
    for (let layer of yolov3Network.layers) {
        let boxNum = 0;
        if (layer.type === 'yolo') {
            boxNum = getYoloDetections(layer, dectBoxes, boxIndex);
        }

        boxIndex += boxNum;
    }

    Utils.doNMS(dectBoxes, totalBoxes, tailLayer.classes, iouThreshold);

    dectBoxes.forEach((value, index, array)=>{
        value.y /= heightScale;
        value.h /= heightScale;

        value.x /= widthScale;
        value.w /= widthScale;
    });
    return dectBoxes;
}